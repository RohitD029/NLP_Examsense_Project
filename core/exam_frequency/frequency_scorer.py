"""
Frequency Scorer
================
Ties together the exam-frequency pipeline: processes past papers,
extracts questions, clusters them, and scores queries against the
resulting frequency data.
"""

import logging
from typing import Optional

from core.retrieval.vector_store import get_all_chunks
from core.exam_frequency.question_extractor import extract_questions
from core.exam_frequency.question_clusterer import (
    cluster_questions,
    find_matching_cluster,
    get_frequency_summary,
)

logger = logging.getLogger(__name__)


def build_frequency_index() -> dict:
    """
    Build the exam-frequency index from all past papers in the vector store.

    Steps:
    1. Retrieve all chunks from the "past_papers" collection
    2. Extract individual questions from each chunk
    3. Cluster similar questions across papers
    4. Return summary statistics

    Returns:
        Frequency summary dict.
    """
    logger.info("Building exam-frequency index from past papers...")

    chunks = get_all_chunks("past_papers")
    if not chunks:
        logger.warning("No past papers found in the vector store.")
        return get_frequency_summary()

    # Group chunks by source document
    doc_chunks: dict[str, list] = {}
    for chunk in chunks:
        doc_name = chunk["metadata"].get("doc_name", chunk["metadata"].get("source_document", "unknown"))
        if doc_name not in doc_chunks:
            doc_chunks[doc_name] = []
        doc_chunks[doc_name].append(chunk)

    # Extract questions from each document
    all_questions = []
    for doc_name, doc_chunk_list in doc_chunks.items():
        full_text = "\n\n".join(c["text"] for c in doc_chunk_list)
        meta = doc_chunk_list[0].get("metadata", {})
        year = meta.get("year")
        subject = meta.get("subject")

        questions = extract_questions(
            text=full_text,
            source_document=doc_name,
            year=year,
            subject=subject,
        )
        all_questions.extend(questions)

    if not all_questions:
        logger.warning("No questions extracted from past papers.")
        return get_frequency_summary()

    cluster_questions(all_questions)

    summary = get_frequency_summary()
    logger.info(
        f"Exam-frequency index built: {summary['total_questions']} questions "
        f"in {summary['total_clusters']} clusters"
    )
    return summary


def score_query(query: str) -> Optional[dict]:
    """
    Score a query against the exam-frequency index.

    Args:
        query: User's query or answer topic.

    Returns:
        Exam frequency data for the matching topic, or None if no match.
    """
    match = find_matching_cluster(query, threshold=0.4)

    if not match:
        return None

    similar = [
        q["question_text"][:200]
        for q in match.get("questions", [])[:5]
    ]

    return {
        "frequency": match["frequency"],
        "total_occurrences": match["total_occurrences"],
        "years": match["years"],
        "papers": match["papers"],
        "representative_question": match["representative_question"][:300],
        "match_similarity": match.get("match_similarity", 0.0),
        "similar_questions": similar,
    }
