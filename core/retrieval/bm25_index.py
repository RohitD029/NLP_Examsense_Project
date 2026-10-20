"""
BM25 Sparse Index
=================
Built from chunks.jsonl via rank_bm25. Persisted as bm25_index.pkl.
Rebuilt on every add/delete (fast — in-memory only).
"""
from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","to","of","in","for","on","with","at","by","from","as",
    "into","through","during","before","after","above","below","between",
    "out","off","over","under","again","further","then","once","and","but",
    "or","nor","not","no","so","than","too","very","just","also","it","its",
    "this","that","these","those","he","she","they","we","you","i","me","my",
    "your","his","her","their","our",
}

# module-level store: {collection: {bm25, chunk_ids, chunks}}
_indices: dict[str, dict] = {}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def build_index(chunks: list[dict], collection: str) -> None:
    """Build BM25 index for a collection from a list of chunk dicts."""
    from rank_bm25 import BM25Plus
    from core.config import settings

    if not chunks:
        _indices[collection] = {"bm25": None, "chunk_ids": [], "chunks": [], "token_sets": []}
        _save(collection, settings)
        return

    tokenized = [_tokenize(c["text"]) for c in chunks]
    token_sets = [set(t) for t in tokenized]
    _indices[collection] = {
        "bm25": BM25Plus(tokenized),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "chunks": chunks,
        "token_sets": token_sets,
    }
    _save(collection, settings)
    logger.info(f"BM25 index built for '{collection}': {len(chunks)} docs")


def rebuild_from_store(collection: str) -> None:
    """Rebuild BM25 from the live vector store."""
    from core.retrieval.vector_store import get_all_chunks
    chunks = get_all_chunks(collection)
    build_index(chunks, collection)


def query_index(query: str, collection: str, top_k: int = 10) -> list[dict]:
    """Return top_k BM25 results for query in collection."""
    idx = _indices.get(collection)
    if not idx:
        load_index(collection)
        idx = _indices.get(collection)
    if not idx or idx.get("bm25") is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = idx["bm25"].get_scores(tokens)
    top_n = min(top_k, len(scores))
    top_idxs = scores.argsort()[::-1][:top_n]

    query_tokens = set(tokens)
    token_sets = idx.get("token_sets")

    results = []
    for i in top_idxs:
        has_match = (bool(query_tokens & token_sets[i])) if (token_sets and i < len(token_sets)) else (scores[i] > 0)
        if has_match and scores[i] > 0:
            chunk = idx["chunks"][i]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": float(scores[i]),
                "collection": collection,
            })
    return results


def load_index(collection: str) -> None:
    """Load BM25 index from disk."""
    from core.config import settings
    path = _pkl_path(collection, settings)
    if path.exists():
        try:
            with open(path, "rb") as f:
                _indices[collection] = pickle.load(f)
            logger.info(f"BM25 loaded from {path}")
        except Exception as e:
            logger.error(f"BM25 load failed: {e}")


def _save(collection: str, settings) -> None:
    path = _pkl_path(collection, settings)
    try:
        tmp = path.with_suffix(".tmp.pkl")
        with open(tmp, "wb") as f:
            pickle.dump(_indices[collection], f)
        import shutil
        shutil.move(str(tmp), str(path))
    except Exception as e:
        logger.error(f"BM25 save failed: {e}")


def _pkl_path(collection: str, settings) -> Path:
    return settings.index_dir / f"bm25_{collection}.pkl"
