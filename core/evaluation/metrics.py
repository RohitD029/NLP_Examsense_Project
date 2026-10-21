"""
Evaluation Metrics
==================
Retrieval-only evaluation metrics (k = 3).
Evaluates retrieval accuracy by matching retrieved chunks against expected document and page/slide.
No LLM calls.
"""
from __future__ import annotations


def evaluate_retrieval_case(
    retrieved_chunks: list[dict],
    expected_doc: str,
    expected_page: int | str | None = None,
    k: int = 3,
) -> tuple[float, float]:
    """
    Compute Precision@K and Recall@K for a single query against expected document and page/slide.

    Returns:
      (precision_at_k, recall_at_k)
    """
    top_k = retrieved_chunks[:k]
    if not top_k:
        return 0.0, 0.0

    hits = 0
    exp_doc_clean = expected_doc.strip().lower()
    exp_page_str = str(expected_page).strip().lower() if expected_page not in (None, "") else None

    for c in top_k:
        meta = c.get("metadata", {})
        doc = str(meta.get("doc_name", "")).strip().lower()
        page = str(meta.get("page_or_slide", "")).strip().lower()

        doc_match = exp_doc_clean in doc or doc in exp_doc_clean
        page_match = True if not exp_page_str else (exp_page_str == page)

        if doc_match and page_match:
            hits += 1

    precision = hits / k
    # For a specific target passage/slide, recall@k is 1.0 if at least one hit was retrieved in top k
    recall = 1.0 if hits > 0 else 0.0

    return round(precision, 4), round(recall, 4)
