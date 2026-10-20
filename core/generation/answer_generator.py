"""
Answer Generator
=================
Full pipeline: query → hybrid retrieval → confidence check → LLM → citations.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def answer_query(
    query: str,
    collection: str = "course_material",
    top_k: int | None = None,
    mode: str = "hybrid",
    doc_filter: Optional[str] = None,
) -> dict:
    """
    Returns:
      query, answer, confidence, is_grounded, abstained,
      citations, model_info, retrieval_info, error?
    """
    from core.config import settings
    from core.retrieval.hybrid_retriever import hybrid_search
    from core.generation.llm_client import generate_answer, _strip_invalid_citations

    # ── Summary Mode Check ───────────────────────────────────────────────────
    if is_summary_query(query, doc_filter):
        return _handle_summary(query, collection, doc_filter)

    k = top_k or settings.top_k_final

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunks = hybrid_search(
        query=query,
        collection=collection,
        top_k_final=k,
        mode=mode,
        doc_filter=doc_filter,
    )

    if not chunks:
        return _not_found(query, "No matching content in your uploaded documents.")

    top_conf = chunks[0].get("confidence", 0.0)
    top_dense = chunks[0].get("dense_score", 0.0)
    top_bm25 = chunks[0].get("bm25_score", 0.0)

    # Abstain if BOTH dense and BM25 evidence are weak
    dense_weak = top_dense < settings.confidence_threshold
    bm25_weak = top_bm25 < settings.confidence_threshold * 0.5  # BM25 scores differ in scale
    should_abstain = dense_weak and (bm25_weak or mode == "dense_only")

    if should_abstain:
        return _low_confidence(query, chunks, top_conf)

    # ── LLM call ─────────────────────────────────────────────────────────────
    llm = generate_answer(query=query, context_chunks=chunks)

    if llm.get("error") and not llm.get("answer"):
        # No LLM but we have retrieved passages — return retrieval-only result
        return _retrieval_only(query, chunks, llm["error"])

    answer_text = llm["answer"]
    # Strip invalid [N] citations invented by the model
    answer_text = _strip_invalid_citations(answer_text, len(chunks))

    # ── Citations ─────────────────────────────────────────────────────────────
    citations = _build_citations(chunks)

    # ── Confidence ────────────────────────────────────────────────────────────
    llm_conf = llm.get("raw_confidence", 0.5)
    final_conf = round(0.6 * llm_conf + 0.4 * top_conf, 2)

    return {
        "query": query,
        "answer": answer_text,
        "confidence": final_conf,
        "is_grounded": not llm.get("abstained", False),
        "abstained": llm.get("abstained", False),
        "citations": citations,
        "retrieved_chunks": chunks,
        "model_info": {
            "model": llm.get("model_used", ""),
            "provider": llm.get("provider", ""),
        },
        "retrieval_info": {
            "total_candidates": len(chunks),
            "top_dense_score": round(top_dense, 4),
            "top_bm25_score": round(top_bm25, 4),
            "top_confidence": round(top_conf, 4),
        },
    }


def _build_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "index": i + 1,
            "doc_name": c.get("metadata", {}).get("doc_name", "?"),
            "page_or_slide": c.get("metadata", {}).get("page_or_slide", "?"),
            "heading": c.get("metadata", {}).get("heading", ""),
            "text_excerpt": c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
            "dense_score": round(c.get("dense_score", 0.0), 4),
            "confidence": round(c.get("confidence", 0.0), 4),
        }
        for i, c in enumerate(chunks)
    ]


def _not_found(query: str, reason: str) -> dict:
    return {
        "query": query, "answer": "", "confidence": 0.0,
        "is_grounded": False, "abstained": True,
        "citations": [], "retrieved_chunks": [],
        "model_info": {"model": "", "provider": ""},
        "retrieval_info": {"total_candidates": 0, "top_dense_score": 0.0,
                           "top_bm25_score": 0.0, "top_confidence": 0.0},
        "not_found_reason": reason,
    }


def _low_confidence(query: str, chunks: list[dict], conf: float) -> dict:
    base = _not_found(query, "Below confidence threshold.")
    base["confidence"] = round(conf * 0.5, 2)
    base["citations"] = _build_citations(chunks[:3])
    base["retrieved_chunks"] = chunks[:3]
    return base


def _retrieval_only(query: str, chunks: list[dict], error: str) -> dict:
    return {
        "query": query,
        "answer": "",
        "confidence": chunks[0].get("confidence", 0.0) if chunks else 0.0,
        "is_grounded": False,
        "abstained": True,
        "citations": _build_citations(chunks),
        "retrieved_chunks": chunks,
        "model_info": {"model": "none", "provider": "none"},
        "retrieval_info": {
            "total_candidates": len(chunks),
            "top_dense_score": chunks[0].get("dense_score", 0.0) if chunks else 0.0,
            "top_bm25_score": chunks[0].get("bm25_score", 0.0) if chunks else 0.0,
            "top_confidence": chunks[0].get("confidence", 0.0) if chunks else 0.0,
        },
        "error": error,
    }


def is_summary_query(query: str, doc_filter: str | None = None) -> bool:
    """Detect if query is asking for a summary/overview of a document."""
    q = query.lower().strip()
    triggers = [
        "what is this", "what is the document", "about", "summarize", "summarise",
        "summary", "overview", "main topic", "main topics", "main points", "tell me about",
        "what does this", "explain this presentation", "explain this ppt", "explain this document",
        "outline", "key takeaways", "table of contents", "introduction",
    ]
    if any(t in q for t in triggers):
        return True
    if doc_filter and doc_filter != "All documents" and len(q.split()) <= 6:
        return True
    return False


def _handle_summary(
    query: str,
    collection: str,
    doc_filter: str | None,
) -> dict:
    from core.retrieval.vector_store import get_all_chunks, get_document_names
    from core.generation.llm_client import generate_answer, _strip_invalid_citations

    all_col_chunks = get_all_chunks(collection)
    if not all_col_chunks:
        return _not_found(query, "No documents uploaded in this collection.")

    # Determine target document
    target_doc = doc_filter if (doc_filter and doc_filter != "All documents") else None
    if not target_doc:
        doc_names = get_document_names(collection)
        if len(doc_names) == 1:
            target_doc = doc_names[0]
        else:
            # Pick first available or best match
            target_doc = doc_names[0]

    # Get all chunks for this doc
    doc_chunks = [
        c for c in all_col_chunks
        if c.get("metadata", {}).get("doc_name") == target_doc
    ]
    if not doc_chunks:
        doc_chunks = all_col_chunks[:10]

    # Sort chunks by page/slide and chunk index
    doc_chunks.sort(
        key=lambda c: (
            int(c.get("metadata", {}).get("page_or_slide", 0) or 0),
            int(c.get("metadata", {}).get("chunk_index", 0) or 0)
        )
    )

    # Pick a spread of chunks (max ~8)
    if len(doc_chunks) <= 8:
        selected_chunks = doc_chunks
    else:
        selected_chunks = [doc_chunks[0], doc_chunks[1]]
        remaining = doc_chunks[2:]
        step = max(1, len(remaining) // 6)
        for i in range(0, len(remaining), step):
            if len(selected_chunks) < 8:
                selected_chunks.append(remaining[i])

    summary_prompt = (
        "You are ExamSense, an academic study assistant. "
        "The student is asking for a summary/overview of the document. "
        "Summarize what this document is about, its key sections, main points, and takeaways "
        "ONLY from the numbered source passages below. "
        "Cite sources as [1], [2], etc. matching the passage numbers given to you. "
        "Do not invent citations. Keep your answer structured, clear, and informative. "
        "End with **Confidence: 0.95**."
    )

    llm = generate_answer(
        query=query,
        context_chunks=selected_chunks,
        system_prompt=summary_prompt,
    )

    if llm.get("error") and not llm.get("answer"):
        return _retrieval_only(query, selected_chunks, llm["error"])

    answer_text = _strip_invalid_citations(llm.get("answer", ""), len(selected_chunks))
    citations = _build_citations(selected_chunks)

    return {
        "query": query,
        "answer": answer_text,
        "confidence": 0.95,
        "is_grounded": True,
        "abstained": False,
        "citations": citations,
        "retrieved_chunks": selected_chunks,
        "model_info": {
            "model": llm.get("model_used", ""),
            "provider": llm.get("provider", ""),
        },
        "retrieval_info": {
            "total_candidates": len(selected_chunks),
            "top_dense_score": 1.0,
            "top_bm25_score": 1.0,
            "top_confidence": 0.95,
            "is_summary_mode": True,
            "target_document": target_doc,
        },
    }

