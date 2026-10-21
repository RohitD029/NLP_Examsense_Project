"""
Evaluation Runner
=================
Runs test sets across Dense, BM25, and Hybrid retrieval methods without any LLM calls.
Computes Recall@3 and Precision@3.
"""
from __future__ import annotations

import logging
from core.evaluation.metrics import evaluate_retrieval_case
from core.retrieval.hybrid_retriever import hybrid_search

logger = logging.getLogger(__name__)


def run_retrieval_evaluation(
    eval_cases: list[dict],
    k: int = 3,
) -> dict:
    """
    Run evaluation on a test set for three retrieval methods:
    1) Dense Only
    2) BM25 Only
    3) Hybrid (Dense + BM25)

    Each test case must have:
      question: str
      expected_doc: str
      expected_page: int | str | None (optional)

    Returns dict with summary metrics and per-case results.
    """
    if not eval_cases:
        return {
            "summary": {},
            "case_results": [],
            "total_cases": 0,
        }

    methods = ["dense_only", "sparse_only", "hybrid"]
    method_labels = {
        "dense_only": "Dense (FAISS)",
        "sparse_only": "BM25 Only",
        "hybrid": "Hybrid RRF",
    }

    scores = {m: {"precisions": [], "recalls": []} for m in methods}
    detailed_cases = []

    for idx, case in enumerate(eval_cases):
        question = case.get("question", "").strip()
        expected_doc = case.get("expected_doc", "").strip()
        expected_page = case.get("expected_page")

        case_entry = {
            "index": idx + 1,
            "question": question,
            "expected_doc": expected_doc,
            "expected_page": expected_page,
            "results": {},
        }

        for method in methods:
            try:
                retrieved = hybrid_search(
                    query=question,
                    top_k_final=k,
                    mode=method,
                )
            except Exception as e:
                logger.warning(f"Retrieval failed for case {idx} method {method}: {e}")
                retrieved = []

            p, r = evaluate_retrieval_case(
                retrieved_chunks=retrieved,
                expected_doc=expected_doc,
                expected_page=expected_page,
                k=k,
            )

            scores[method]["precisions"].append(p)
            scores[method]["recalls"].append(r)

            case_entry["results"][method] = {
                "precision": p,
                "recall": r,
                "retrieved": [
                    f"{c.get('metadata', {}).get('doc_name')} · p.{c.get('metadata', {}).get('page_or_slide')}"
                    for c in retrieved
                ],
            }

        detailed_cases.append(case_entry)

    summary = {}
    for m in methods:
        label = method_labels[m]
        p_list = scores[m]["precisions"]
        r_list = scores[m]["recalls"]

        avg_p = sum(p_list) / len(p_list) if p_list else 0.0
        avg_r = sum(r_list) / len(r_list) if r_list else 0.0

        summary[label] = {
            "Precision@3": round(avg_p, 4),
            "Recall@3": round(avg_r, 4),
        }

    return {
        "summary": summary,
        "case_results": detailed_cases,
        "total_cases": len(eval_cases),
    }
