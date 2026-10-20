"""
Hybrid Retriever
=================
Combines dense (FAISS) and sparse (BM25) results with Reciprocal Rank Fusion.
Returns top_k_final chunks with dense_score, bm25_score, rrf_score, confidence.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    collection: str = "course_material",
    top_k_final: int | None = None,
    mode: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid",
    doc_filter: Optional[str] = None,
) -> list[dict]:
    """
    Run hybrid retrieval and return merged, ranked results.

    Each result dict has:
      chunk_id, text, metadata, collection,
      dense_score, bm25_score, rrf_score, confidence
    """
    from core.config import settings
    from core.retrieval.embedder import embed_query
    from core.retrieval.vector_store import query_dense
    from core.retrieval.bm25_index import query_index

    k_final = top_k_final or settings.top_k_final

    dense_results: list[dict] = []
    sparse_results: list[dict] = []

    # If filtering by doc, retrieve more initial candidates to avoid empty pools
    cand_mult = 4 if (doc_filter and doc_filter != "All documents") else 1

    if mode in ("hybrid", "dense_only"):
        q_emb = embed_query(query)
        dense_results = query_dense(q_emb, collection, top_k=settings.top_k_dense * cand_mult)

    if mode in ("hybrid", "sparse_only"):
        sparse_results = query_index(query, collection, top_k=settings.top_k_sparse * cand_mult)

    # Apply doc_filter if requested
    if doc_filter and doc_filter != "All documents":
        dense_results = [
            r for r in dense_results if r.get("metadata", {}).get("doc_name") == doc_filter
        ]
        sparse_results = [
            r for r in sparse_results if r.get("metadata", {}).get("doc_name") == doc_filter
        ]

    if not dense_results and not sparse_results:
        return []

    if mode == "dense_only":
        results = dense_results[:k_final]
        for r in results:
            r["dense_score"] = r.pop("score", 0.0)
            r["bm25_score"] = 0.0
            r["rrf_score"] = r["dense_score"]
        return _add_confidence(results, mode)

    if mode == "sparse_only":
        results = sparse_results[:k_final]
        for r in results:
            r["bm25_score"] = r.pop("score", 0.0)
            r["dense_score"] = 0.0
            r["rrf_score"] = r["bm25_score"]
        return _add_confidence(results, mode)

    # ── Hybrid: RRF ──────────────────────────────────────────────────────────
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
    """
    Compute per-result confidence and a top-level confidence for the query.
    Confidence = weighted average of dense cosine (60%) + BM25 rank agreement (40%).
    """
    if not results:
        return results

    max_bm25 = max((r.get("bm25_score", 0.0) for r in results), default=0.0)

    # Normalise BM25 score to [0,1]
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
