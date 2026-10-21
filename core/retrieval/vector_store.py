"""
Vector Store & In-Memory Session Storage
========================================
Per-session storage in st.session_state (with non-Streamlit fallback for tests).
FAISS IndexFlatIP over L2-normalized float32 vectors.
Rebuilds FAISS from stored embeddings on document delete without re-embedding.
Rebuilds BM25 on add/delete.
No disk persistence, no chromadb, no data folder dependencies.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import faiss
import numpy as np

from core.config import settings
from core.retrieval.bm25_index import build_bm25_index

logger = logging.getLogger(__name__)

# Fallback store for CLI / unit testing outside Streamlit request context
_fallback_store: dict = {
    "documents": {},
    "chunks": [],
    "embeddings": np.zeros((0, 384), dtype=np.float32),
    "faiss_index": None,
    "bm25_data": {"bm25": None, "chunks": [], "tokenized_corpus": [], "token_sets": []},
}


def _get_store() -> dict:
    """Retrieve session store from st.session_state or fallback in non-Streamlit context."""
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            if "_session_store" not in st.session_state:
                st.session_state["_session_store"] = {
                    "documents": {},
                    "chunks": [],
                    "embeddings": np.zeros((0, 384), dtype=np.float32),
                    "faiss_index": None,
                    "bm25_data": {"bm25": None, "chunks": [], "tokenized_corpus": [], "token_sets": []},
                }
            return st.session_state["_session_store"]
    except Exception:
        pass
    return _fallback_store


def get_documents() -> list[dict]:
    """Return list of indexed documents in the current session."""
    store = _get_store()
    return list(store["documents"].values())


def get_document_names() -> list[str]:
    """Return names of indexed documents."""
    store = _get_store()
    return list(store["documents"].keys())


def get_total_chunks() -> int:
    """Return total number of chunks in the current session."""
    store = _get_store()
    return len(store["chunks"])


def document_exists(doc_name: str) -> bool:
    """Check if document is already indexed in the session."""
    store = _get_store()
    return doc_name in store["documents"]


def add_document(
    doc_name: str,
    chunks: list[dict],
    status_note: str = "Ready",
    total_pages: int = 1,
) -> Tuple[bool, str]:
    """
    Add a document and its chunks to the session store.
    Computes dense embeddings once and indexes with FAISS and BM25.
    Enforces per-session file and chunk limits.
    """
    store = _get_store()

    # 1. Check max files per session
    if len(store["documents"]) >= settings.max_files_per_session and doc_name not in store["documents"]:
        return False, f"Maximum {settings.max_files_per_session} files per session limit reached."

    # 2. Check max chunks per session
    if len(store["chunks"]) + len(chunks) > settings.max_chunks_per_session:
        return False, f"Session limit of {settings.max_chunks_per_session:,} chunks reached. Stop indexing."

    if not chunks:
        store["documents"][doc_name] = {
            "doc_name": doc_name,
            "chunks_count": 0,
            "status_note": status_note,
            "total_pages": total_pages,
        }
        return True, "Indexed 0 chunks."

    # 3. Compute dense embeddings
    from core.retrieval.embedder import embed_texts
    texts = [c["text"] for c in chunks]
    new_embs = embed_texts(texts)  # (M, 384) float32

    # 4. Append to session store
    store["chunks"].extend(chunks)

    current_embs = store["embeddings"]
    if current_embs is None or len(current_embs) == 0:
        store["embeddings"] = new_embs
    else:
        store["embeddings"] = np.vstack([current_embs, new_embs]).astype(np.float32)

    # 5. Record document info
    store["documents"][doc_name] = {
        "doc_name": doc_name,
        "chunks_count": len(chunks),
        "status_note": status_note,
        "total_pages": total_pages,
    }

    # 6. Rebuild FAISS & BM25
    _rebuild_indexes()
    return True, f"Successfully indexed {len(chunks)} chunks."


def delete_document(doc_name: str) -> None:
    """
    Remove a document and its chunks from the session store.
    Rebuilds the FAISS index using stored embeddings without re-embedding.
    Rebuilds BM25 index.
    """
    store = _get_store()
    if doc_name not in store["documents"]:
        return

    del store["documents"][doc_name]

    if not store["chunks"]:
        store["embeddings"] = np.zeros((0, 384), dtype=np.float32)
        store["faiss_index"] = None
        store["bm25_data"] = build_bm25_index([])
        return

    # Identify chunks to keep
    keep_indices = []
    kept_chunks = []
    for idx, c in enumerate(store["chunks"]):
        if c.get("metadata", {}).get("doc_name") != doc_name:
            keep_indices.append(idx)
            kept_chunks.append(c)

    store["chunks"] = kept_chunks

    if keep_indices and len(store["embeddings"]) > 0:
        store["embeddings"] = store["embeddings"][keep_indices]
    else:
        store["embeddings"] = np.zeros((0, 384), dtype=np.float32)

    _rebuild_indexes()


def query_dense(
    query_vector: np.ndarray,
    top_k: int = 8,
    doc_filter: Optional[str] = None,
) -> list[dict]:
    """
    Query the in-memory FAISS IndexFlatIP.
    Returns top_k chunks ranked by dot product (cosine similarity).
    """
    store = _get_store()
    index = store.get("faiss_index")
    chunks = store.get("chunks", [])

    if index is None or len(chunks) == 0 or index.ntotal == 0:
        return []

    # Ensure query shape is (1, D)
    q = query_vector.reshape(1, -1).astype(np.float32)

    search_k = min(index.ntotal, top_k * (4 if doc_filter and doc_filter != "All documents" else 1))
    scores, indices = index.search(q, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]
        if doc_filter and doc_filter != "All documents":
            if chunk.get("metadata", {}).get("doc_name") != doc_filter:
                continue

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "score": float(score),
        })

        if len(results) >= top_k:
            break

    return results


def get_bm25_data() -> dict:
    """Return the current BM25 index data structure."""
    store = _get_store()
    return store.get("bm25_data", {})


def get_all_chunks(doc_filter: Optional[str] = None) -> list[dict]:
    """Return all stored chunks, optionally filtered by document."""
    store = _get_store()
    chunks = store.get("chunks", [])
    if doc_filter and doc_filter != "All documents":
        return [c for c in chunks if c.get("metadata", {}).get("doc_name") == doc_filter]
    return list(chunks)


def _rebuild_indexes() -> None:
    """Internal helper: rebuild FAISS and BM25 indices from stored state."""
    store = _get_store()
    chunks = store.get("chunks", [])
    embs = store.get("embeddings")

    # 1. Rebuild FAISS
    if embs is not None and len(embs) > 0 and len(chunks) == len(embs):
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        store["faiss_index"] = index
    else:
        store["faiss_index"] = None

    # 2. Rebuild BM25
    store["bm25_data"] = build_bm25_index(chunks)
