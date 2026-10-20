"""
Vector Store — FAISS backend
==============================
Replaces ChromaDB. Uses FAISS IndexFlatIP over L2-normalised embeddings
(dot product == cosine similarity when vectors are normalised).

Two logical collections: "course_material" and "past_papers".
Persists to data/index/:
  chunks.jsonl   — text + metadata per chunk
  embeddings.npy — float32 array, row i corresponds to chunk i
  docs.json      — {collection: {doc_name: doc_info}}
  model.txt      — name of the embedding model used

Incremental add / delete without re-embedding existing chunks.
Thread-safe writes via temp-file-then-rename on Windows.
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

COLLECTIONS = {"course_material", "past_papers"}

# ── In-memory state (loaded once, mutated on add/delete) ──────────────────────
_chunks: list[dict] = []          # [{chunk_id, text, metadata, collection}, ...]
_embeddings: Optional[np.ndarray] = None  # (N, D) float32
_docs: dict[str, dict] = {}       # {collection: {doc_name: {doc_id, chunk_count, ...}}}
_index = None                     # faiss.IndexFlatIP (rebuilt from _embeddings)
_loaded: bool = False


# ── Public API ────────────────────────────────────────────────────────────────

def load_index() -> None:
    """Load all persisted data from disk. Safe to call multiple times."""
    global _chunks, _embeddings, _docs, _index, _loaded

    from core.config import settings
    idx_dir = settings.index_dir

    chunks_path = idx_dir / "chunks.jsonl"
    emb_path = idx_dir / "embeddings.npy"
    docs_path = idx_dir / "docs.json"
    model_path = idx_dir / "model.txt"

    _chunks = []
    _embeddings = None
    _docs = {}

    # Check if model changed → warn but still load
    if model_path.exists():
        from core.retrieval.embedder import get_model_name
        try:
            stored = model_path.read_text(encoding="utf-8").strip()
            current = get_model_name()
            if stored != current:
                logger.warning(
                    f"Embedding model changed: stored={stored}, current={current}. "
                    "Re-indexing required."
                )
                # Flag for the UI to show a banner
                import streamlit as st
                if hasattr(st, "session_state"):
                    st.session_state["model_changed_warning"] = (
                        f"⚠️ Embedding model changed from **{stored}** to **{current}**. "
                        "Please delete and re-upload all documents to rebuild the index."
                    )
        except Exception:
            pass

    if docs_path.exists():
        try:
            _docs = json.loads(docs_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load docs.json: {e}")
            _docs = {}

    if chunks_path.exists():
        try:
            with open(chunks_path, encoding="utf-8") as f:
                _chunks = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load chunks.jsonl: {e}")
            _chunks = []

    if emb_path.exists() and _chunks:
        try:
            _embeddings = np.load(str(emb_path)).astype(np.float32)
            if len(_embeddings) != len(_chunks):
                logger.error("Embedding / chunk count mismatch — rebuilding FAISS index from scratch.")
                _embeddings = None
                _chunks = []
        except Exception as e:
            logger.error(f"Failed to load embeddings.npy: {e}")
            _embeddings = None

    _rebuild_faiss()
    _loaded = True
    logger.info(
        f"Index loaded: {len(_chunks)} chunks across "
        f"{sum(len(v) for v in _docs.values())} documents."
    )


def add_chunks(chunks: list[dict], collection: str) -> int:
    """
    Add chunks (with pre-computed embeddings in chunk['embedding']) to the index.
    Returns number of chunks added.
    """
    global _chunks, _embeddings

    if not chunks:
        return 0
    if collection not in COLLECTIONS:
        raise ValueError(f"Unknown collection: {collection}")

    _ensure_loaded()

    new_texts = [c["text"] for c in chunks]
    from core.retrieval.embedder import embed_texts
    new_embs = embed_texts(new_texts)  # (M, D)

    # Assign IDs if missing
    for c in chunks:
        if "chunk_id" not in c:
            c["chunk_id"] = str(uuid.uuid4())
        c["collection"] = collection

    _chunks.extend(chunks)

    if _embeddings is None or len(_embeddings) == 0:
        _embeddings = new_embs
    else:
        _embeddings = np.vstack([_embeddings, new_embs]).astype(np.float32)

    # Update docs registry
    doc_name = chunks[0]["metadata"]["doc_name"]
    doc_id = chunks[0]["metadata"].get("doc_id", str(uuid.uuid4()))
    _docs.setdefault(collection, {})[doc_name] = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "chunk_count": len(chunks),
        "collection": collection,
    }

    _rebuild_faiss()
    _persist()
    logger.info(f"Added {len(chunks)} chunks for '{doc_name}' to '{collection}'.")
    return len(chunks)


def delete_document(doc_name: str, collection: str) -> int:
    """
    Remove all chunks for doc_name from collection.
    Rebuilds FAISS index from remaining embeddings (no re-embedding needed).
    Returns number of chunks deleted.
    """
    global _chunks, _embeddings

    _ensure_loaded()

    keep_mask = [
        not (c["metadata"].get("doc_name") == doc_name and c.get("collection") == collection)
        for c in _chunks
    ]
    deleted = len(keep_mask) - sum(keep_mask)
    if deleted == 0:
        logger.warning(f"No chunks found for '{doc_name}' in '{collection}'.")
        return 0

    _chunks = [c for c, keep in zip(_chunks, keep_mask) if keep]
    if _embeddings is not None and len(_embeddings) > 0:
        indices = [i for i, keep in enumerate(keep_mask) if keep]
        _embeddings = _embeddings[indices] if indices else np.zeros((0, _embeddings.shape[1]), dtype=np.float32)

    _docs.get(collection, {}).pop(doc_name, None)
    _rebuild_faiss()
    _persist()
    logger.info(f"Deleted {deleted} chunks for '{doc_name}' from '{collection}'.")
    return deleted


def query_dense(
    query_embedding: np.ndarray,
    collection: str,
    top_k: int = 10,
) -> list[dict]:
    """Dense (FAISS) search. Returns top_k results sorted by cosine similarity."""
    _ensure_loaded()

    if _index is None or len(_chunks) == 0:
        return []

    # Filter to collection
    col_indices = [i for i, c in enumerate(_chunks) if c.get("collection") == collection]
    if not col_indices:
        return []

    col_embs = _embeddings[col_indices]  # (M, D)

    # Compute cosine similarity via dot product (vectors are L2-normalised)
    scores = col_embs @ query_embedding  # (M,)
    top_n = min(top_k, len(scores))
    top_local = np.argsort(scores)[::-1][:top_n]

    results = []
    for local_idx in top_local:
        global_idx = col_indices[local_idx]
        chunk = _chunks[global_idx]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": float(scores[local_idx]),
            "collection": collection,
        })
    return results


def get_all_chunks(collection: Optional[str] = None) -> list[dict]:
    """Return all chunks, optionally filtered by collection."""
    _ensure_loaded()
    if collection:
        return [c for c in _chunks if c.get("collection") == collection]
    return list(_chunks)


def get_collection_stats(collection: str) -> dict:
    """Return stats dict for a collection."""
    _ensure_loaded()
    docs = _docs.get(collection, {})
    chunks = [c for c in _chunks if c.get("collection") == collection]
    return {
        "collection": collection,
        "total_documents": len(docs),
        "total_chunks": len(chunks),
        "documents": list(docs.keys()),
    }


def document_exists(doc_name: str, collection: str) -> bool:
    """Check if a document is already indexed."""
    _ensure_loaded()
    return doc_name in _docs.get(collection, {})


def get_document_names(collection: str) -> list[str]:
    """Return sorted list of document names in collection."""
    _ensure_loaded()
    docs = _docs.get(collection, {})
    if docs:
        return sorted(list(docs.keys()))
    chunks = [c for c in _chunks if c.get("collection") == collection]
    names = {c.get("metadata", {}).get("doc_name") for c in chunks if c.get("metadata", {}).get("doc_name")}
    return sorted(list(names))


# ── Private helpers ───────────────────────────────────────────────────────────

def _ensure_loaded():
    if not _loaded:
        load_index()


def _rebuild_faiss():
    global _index
    try:
        import faiss
        if _embeddings is None or len(_embeddings) == 0:
            _index = None
            return
        dim = _embeddings.shape[1]
        _index = faiss.IndexFlatIP(dim)
        _index.add(_embeddings)
    except Exception as e:
        logger.error(f"FAISS rebuild failed: {e}")
        _index = None


def _atomic_write(path: Path, content: str | bytes) -> None:
    """Write via temp file then rename (safe on Windows/OneDrive)."""
    tmp = path.with_suffix(".tmp")
    try:
        if isinstance(content, str):
            tmp.write_text(content, encoding="utf-8")
        else:
            tmp.write_bytes(content)
        shutil.move(str(tmp), str(path))
    except Exception as e:
        logger.error(f"Atomic write failed for {path}: {e}")
        tmp.unlink(missing_ok=True)
        raise


def _persist() -> None:
    """Save all state to disk."""
    from core.config import settings
    from core.retrieval.embedder import get_model_name
    idx_dir = settings.index_dir

    # Write chunks.jsonl
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in _chunks)
    _atomic_write(idx_dir / "chunks.jsonl", lines)

    # Write embeddings.npy
    if _embeddings is not None and len(_embeddings) > 0:
        tmp_emb = idx_dir / "embeddings.tmp.npy"
        np.save(str(tmp_emb), _embeddings)
        shutil.move(str(tmp_emb), str(idx_dir / "embeddings.npy"))
    else:
        (idx_dir / "embeddings.npy").unlink(missing_ok=True)

    # Write docs.json
    _atomic_write(idx_dir / "docs.json", json.dumps(_docs, indent=2, ensure_ascii=False))

    # Write model name
    _atomic_write(idx_dir / "model.txt", get_model_name())
