"""
Evaluation Metrics
==================
All retrieval metrics run without calling the LLM.
Answer F1 is purely token-level overlap (no LLM).
"""
from __future__ import annotations

import re
from collections import Counter


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    hits = set(retrieved[:k]) & set(relevant)
    return round(len(hits) / len(relevant), 4)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = set(top) & set(relevant)
    return round(len(hits) / len(top), 4)


def f1_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, cid in enumerate(retrieved, 1):
        if cid in relevant_set:
            return round(1.0 / rank, 4)
    return 0.0


def answer_token_f1(generated: str, reference: str) -> float:
    def _tok(text):
        return re.findall(r"\w+", text.lower())

    gen = Counter(_tok(generated))
    ref = Counter(_tok(reference))
    if not gen or not ref:
        return 0.0
    common = sum((gen & ref).values())
    if common == 0:
        return 0.0
    p = common / sum(gen.values())
    r = common / sum(ref.values())
    return round(2 * p * r / (p + r), 4)


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int] = (1, 3, 5),
) -> dict:
    """All retrieval metrics — no LLM needed."""
    m = {}
    for k in k_values:
        m[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        m[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        m[f"f1@{k}"] = f1_at_k(retrieved_ids, relevant_ids, k)
    m["mrr"] = mrr(retrieved_ids, relevant_ids)
    return m


def compute_all_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    generated_answer: str = "",
    reference_answer: str = "",
    k_values: list[int] = (1, 3, 5),
) -> dict:
    """Combines retrieval metrics with token F1 for answer quality."""
    metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k_values)
    if reference_answer:
        metrics["answer_f1"] = answer_token_f1(generated_answer, reference_answer)
    return metrics

