"""
Evaluation Runner
==================
Runs a curated test set through the retrieval + generation pipeline
and computes evaluation metrics. Supports ablation modes.
"""

import logging
import time

from core.retrieval.hybrid_retriever import hybrid_search
from core.generation.answer_generator import answer_query
from core.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def run_evaluation(
    test_set: list[dict],
    collection_name: str = "course_material",
    k_values: list[int] = [1, 3, 5],
    mode: str = "hybrid",
    progress_callback=None,
) -> dict:
    """
    Run evaluation on a curated test set.

    Args:
        test_set: List of test cases:
            [{"question": str, "expected_chunk_ids": [str], "expected_answer": str}]
        collection_name: Collection to evaluate against.
        k_values: K values for retrieval metrics.
        mode: "hybrid", "dense_only", or "sparse_only" for ablation.
        progress_callback: Optional callable(current, total) for progress updates.

    Returns:
        Evaluation results dict.
    """
    logger.info(f"Running evaluation ({mode}) on {len(test_set)} test cases...")
    start_time = time.time()

    results = []
    all_metrics: dict[str, list[float]] = {}
    failed = 0

    for i, test_case in enumerate(test_set):
        question = test_case.get("question", "")
        expected_chunk_ids = test_case.get("expected_chunk_ids", [])
        expected_answer = test_case.get("expected_answer", "")

        if progress_callback:
            progress_callback(i, len(test_set))

        logger.info(f"Evaluating [{i + 1}/{len(test_set)}]: {question[:60]}...")

        try:
            # Run retrieval
            retrieved = hybrid_search(
                query=question,
                collection=collection_name,
                mode=mode,
            )
            retrieved_ids = [r["chunk_id"] for r in retrieved]

            # Run full answer generation
            answer_result = answer_query(
                query=question,
                collection=collection_name,
                mode=mode,
            )
            generated_answer = answer_result.get("answer", "")

            # Compute metrics
            metrics = compute_all_metrics(
                retrieved_ids=retrieved_ids,
                relevant_ids=expected_chunk_ids,
                generated_answer=generated_answer,
                reference_answer=expected_answer,
                k_values=k_values,
            )

            # Accumulate for averages
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

            results.append({
                "question": question,
                "generated_answer": generated_answer[:500],
                "expected_answer": expected_answer[:500],
                "confidence": answer_result.get("confidence", 0),
                "is_grounded": answer_result.get("is_grounded", False),
                "retrieved_chunk_ids": retrieved_ids,
                "expected_chunk_ids": expected_chunk_ids,
                "metrics": metrics,
                "status": "success",
            })

        except Exception as e:
            logger.error(f"Evaluation failed for question '{question[:40]}': {e}")
            failed += 1
            results.append({
                "question": question,
                "status": "error",
                "error": str(e),
                "metrics": {},
            })

    if progress_callback:
        progress_callback(len(test_set), len(test_set))

    elapsed = round(time.time() - start_time, 2)

    # Compute aggregate (average) metrics
    aggregate = {}
    for metric_name, values in all_metrics.items():
        if values:
            aggregate[metric_name] = round(sum(values) / len(values), 4)

    eval_result = {
        "total_cases": len(test_set),
        "completed": len(test_set) - failed,
        "failed": failed,
        "elapsed_seconds": elapsed,
        "aggregate_metrics": aggregate,
        "per_question_results": results,
        "mode": mode,
    }

    logger.info(
        f"Evaluation complete ({mode}): {eval_result['completed']}/{eval_result['total_cases']} "
        f"cases in {elapsed}s. Aggregate: {aggregate}"
    )
    return eval_result
