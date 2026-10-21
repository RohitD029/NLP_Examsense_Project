"""
BM25 Sparse Index
=================
In-memory BM25 index using rank_bm25 and bm25_tokenize.
Per-session only — no disk persistence.
Rebuilt on document add/delete.
"""
from __future__ import annotations

import logging
from typing import Optional

from rank_bm25 import BM25Plus
from core.nlp.preprocess import bm25_tokenize

logger = logging.getLogger(__name__)


def build_bm25_index(chunks: list[dict]) -> dict:
    """
    Build in-memory BM25 index from a list of chunks.

    Returns dict containing:
      bm25: BM25Plus instance (or None)
      chunks: list[dict]
      tokenized_corpus: list[list[str]]
      token_sets: list[set[str]]
    """
    if not chunks:
        return {
            "bm25": None,
            "chunks": [],
            "tokenized_corpus": [],
            "token_sets": [],
        }

    tokenized_corpus = [bm25_tokenize(c["text"]) for c in chunks]
    token_sets = [set(t) for t in tokenized_corpus]

    # Handle cases where all documents have no non-stopword tokens
    non_empty = [t for t in tokenized_corpus if t]
    bm25 = BM25Plus(tokenized_corpus) if non_empty else None

    return {
        "bm25": bm25,
        "chunks": chunks,
        "tokenized_corpus": tokenized_corpus,
        "token_sets": token_sets,
    }


def query_bm25(
    query: str,
    bm25_data: dict,
    top_k: int = 8,
    doc_filter: Optional[str] = None,
) -> list[dict]:
    """
    Query the BM25 index. Returns top_k chunks ranked by BM25 score.
    """
    if not bm25_data or not bm25_data.get("bm25") or not bm25_data.get("chunks"):
        return []

    tokens = bm25_tokenize(query)
    if not tokens:
        return []

    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]
    token_sets = bm25_data.get("token_sets", [])

    scores = bm25.get_scores(tokens)
    query_token_set = set(tokens)

    # Filter by doc_filter if specified
    scored_candidates = []
    for i, chunk in enumerate(chunks):
        if doc_filter and doc_filter != "All documents":
            if chunk.get("metadata", {}).get("doc_name") != doc_filter:
                continue

        score = float(scores[i])
        # Only return items with positive score and token overlap
        has_overlap = bool(query_token_set & token_sets[i]) if (i < len(token_sets)) else (score > 0)
        if has_overlap and score > 0:
            scored_candidates.append((score, chunk))

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in scored_candidates[:top_k]:
        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "score": score,
        })

    return results
