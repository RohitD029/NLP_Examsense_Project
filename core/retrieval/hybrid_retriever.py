"""
Hybrid Retriever
================
Combines dense (FAISS IndexFlatIP) and sparse (BM25Plus) results using Reciprocal Rank Fusion.
Returns ranked chunks with dense_score, bm25_score, rrf_score, and internal confidence.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from core.config import settings
from core.retrieval.bm25_index import query_bm25
from core.retrieval.embedder import embed_query
from core.retrieval.vector_store import get_bm25_data, query_dense

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    top_k_final: int | None = None,
    mode: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid",
    doc_filter: Optional[str] = None,
    **kwargs,  # Accept legacy collection arg if passed
) -> list[dict]:
    """
    Run retrieval (dense, sparse, or hybrid RRF) on the current session store.

    Each result dict has:
      chunk_id, text, metadata,
      dense_score, bm25_score, rrf_score, confidence
    """
    k_final = top_k_final or settings.top_k_final
    cand_mult = 4 if (doc_filter and doc_filter != "All documents") else 1

    dense_results: list[dict] = []
    sparse_results: list[dict] = []

    # 1. Dense retrieval
    if mode in ("hybrid", "dense_only"):
        try:
            q_emb = embed_query(query)
            dense_results = query_dense(
                query_vector=q_emb,
                top_k=settings.top_k_dense * cand_mult,
                doc_filter=doc_filter,
            )
        except Exception as e:
            logger.warning(f"Dense search error: {e}")

    # 2. Sparse (BM25) retrieval
    if mode in ("hybrid", "sparse_only"):
        try:
            bm25_data = get_bm25_data()
            sparse_results = query_bm25(
                query=query,
                bm25_data=bm25_data,
                top_k=settings.top_k_sparse * cand_mult,
                doc_filter=doc_filter,
            )
        except Exception as e:
            logger.warning(f"Sparse search error: {e}")

    # Filter again if not already filtered
    if doc_filter and doc_filter != "All documents":
        dense_results = [
            r for r in dense_results if r.get("metadata", {}).get("doc_name") == doc_filter
        ]
        sparse_results = [
            r for r in sparse_results if r.get("metadata", {}).get("doc_name") == doc_filter
        ]

    if not dense_results and not sparse_results:
        return []

    # 3. Dense-only mode
    if mode == "dense_only":
        results = dense_results[:k_final]
        for r in results:
            r["dense_score"] = r.get("score", 0.0)
            r["bm25_score"] = 0.0
            r["rrf_score"] = r["dense_score"]
        return _add_confidence(results, mode)

    # 4. Sparse-only mode
    if mode == "sparse_only":
        results = sparse_results[:k_final]
        for r in results:
            r["bm25_score"] = r.get("score", 0.0)
            r["dense_score"] = 0.0
            r["rrf_score"] = r["bm25_score"]
        return _add_confidence(results, mode)

    # 5. Hybrid mode (Reciprocal Rank Fusion)
    K = settings.rrf_k
    scores: dict[str, dict] = {}

    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        scores.setdefault(cid, {
            "chunk": r, "dense_rank": None, "sparse_rank": None,
            "dense_score": 0.0, "bm25_score": 0.0,
        })
        scores[cid]["dense_rank"] = rank
        scores[cid]["dense_score"] = r.get("score", 0.0)

    for rank, r in enumerate(sparse_results):
        cid = r["chunk_id"]
        scores.setdefault(cid, {
            "chunk": r, "dense_rank": None, "sparse_rank": None,
            "dense_score": 0.0, "bm25_score": 0.0,
        })
        scores[cid]["sparse_rank"] = rank
        scores[cid]["bm25_score"] = r.get("score", 0.0)

    for cid, d in scores.items():
        rrf = 0.0
        if d["dense_rank"] is not None:
            rrf += 1.0 / (K + d["dense_rank"] + 1)
        if d["sparse_rank"] is not None:
            rrf += 1.0 / (K + d["sparse_rank"] + 1)
        d["rrf_score"] = rrf

    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)[:k_final]

    merged = []
    for d in ranked:
        chunk = dict(d["chunk"])
        chunk.pop("score", None)
        chunk["dense_score"] = d["dense_score"]
        chunk["bm25_score"] = d["bm25_score"]
        chunk["rrf_score"] = d["rrf_score"]
        merged.append(chunk)

    return _add_confidence(merged, mode)


def _add_confidence(results: list[dict], mode: str) -> list[dict]:
    """Compute internal confidence score for retrieval evidence."""
    if not results:
        return results

    max_bm25 = max((r.get("bm25_score", 0.0) for r in results), default=0.0)

    for r in results:
        bm25_norm = (r.get("bm25_score", 0.0) / max_bm25) if max_bm25 > 0 else 0.0
        dense = r.get("dense_score", 0.0)
        if mode == "sparse_only":
            conf = bm25_norm
        elif mode == "dense_only":
            conf = max(0.0, min(1.0, dense))
        else:
            conf = 0.6 * max(0.0, min(1.0, dense)) + 0.4 * bm25_norm
        r["confidence"] = round(conf, 4)

    return results
