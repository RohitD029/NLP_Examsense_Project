"""
Answer Generator
================
Orchestrates retrieval -> evidence check -> Groq generation -> source chips.
Never calls LLM when evidence is weak.
Handles summary queries with spread-out chunks.
Builds source chips from retrieved chunks and strips invented citation markers.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from core.config import settings
from core.generation.llm_client import generate_answer
from core.retrieval.hybrid_retriever import hybrid_search
from core.retrieval.vector_store import get_all_chunks, get_document_names

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I couldn't find this in your documents. Try rephrasing or upload more material."


def strip_invented_citations(text: str, num_chunks: int) -> str:
    """
    Strip any citation markers [n] that the model invents beyond the provided chunks,
    or invalid markers.
    """
    def _replace(match):
        n = int(match.group(1))
        if 1 <= n <= num_chunks:
            return match.group(0)
        return ""

    cleaned = re.sub(r"\[(\d+)\]", _replace, text)
    # Clean up any resulting double spaces
    return re.sub(r" +", " ", cleaned).strip()


def is_summary_query(query: str) -> bool:
    """Check if query is asking for a summary/overview of a document."""
    q = query.lower().strip()
    triggers = [
        "what is this", "what is the document", "about", "summarize", "summarise",
        "summary", "overview", "main topic", "main topics", "main points", "tell me about",
        "what does this", "explain this presentation", "explain this ppt", "explain this document",
        "outline", "key takeaways", "key definitions", "table of contents", "introduction",
    ]
    return any(t in q for t in triggers)


def generate_study_answer(
    query: str,
    doc_filter: Optional[str] = None,
) -> dict:
    """
    Main answering entry point.
    Returns:
      answer: str
      source_chips: list of dicts (label, doc_name, page_or_slide, excerpt, heading)
      not_found: bool
      error: str | None
      is_summary: bool
    """
    # ── Summary Mode ──────────────────────────────────────────────────────────
    if is_summary_query(query):
        return _handle_summary(query, doc_filter)

    # ── Hybrid Retrieval ──────────────────────────────────────────────────────
    retrieved = hybrid_search(
        query=query,
        top_k_final=settings.top_k_final,
        mode="hybrid",
        doc_filter=doc_filter,
    )

    if not retrieved:
        # Nothing retrieved at all -> abstain without calling LLM
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source_chips": [],
            "not_found": True,
            "error": None,
            "is_summary": False,
        }

    # ── Evidence Check ────────────────────────────────────────────────────────
    top_dense = retrieved[0].get("dense_score", 0.0)
    top_bm25 = retrieved[0].get("bm25_score", 0.0)

    dense_weak = top_dense < settings.confidence_threshold
    bm25_weak = top_bm25 <= 0.0

    # Abstain only when BOTH dense and BM25 evidence are weak
    if dense_weak and bm25_weak:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source_chips": [],
            "not_found": True,
            "error": None,
            "is_summary": False,
        }

    # ── Call LLM ──────────────────────────────────────────────────────────────
    top_chunks = retrieved[:settings.top_k_final]
    llm_resp = generate_answer(query=query, context_chunks=top_chunks)

    if llm_resp.get("error"):
        return {
            "answer": "",
            "source_chips": _build_source_chips(top_chunks),
            "not_found": False,
            "error": llm_resp["error"],
            "is_summary": False,
        }

    raw_answer = llm_resp.get("answer", "").strip()

    if llm_resp.get("abstained") or not raw_answer or NOT_FOUND_MESSAGE.lower() in raw_answer.lower():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source_chips": [],
            "not_found": True,
            "error": None,
            "is_summary": False,
        }

    # Strip invented [n] citations
    clean_answer = strip_invented_citations(raw_answer, len(top_chunks))
    source_chips = _build_source_chips(top_chunks)

    return {
        "answer": clean_answer,
        "source_chips": source_chips,
        "not_found": False,
        "error": None,
        "is_summary": False,
    }


def _handle_summary(query: str, doc_filter: Optional[str] = None) -> dict:
    """Handle summary queries: skip evidence check, select spread-out chunks (max 6)."""
    all_chunks = get_all_chunks(doc_filter)
    if not all_chunks:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source_chips": [],
            "not_found": True,
            "error": None,
            "is_summary": True,
        }

    # Target document selection
    target_doc = doc_filter if (doc_filter and doc_filter != "All documents") else None
    if not target_doc:
        doc_names = get_document_names()
        target_doc = doc_names[0] if doc_names else None

    if target_doc:
        doc_chunks = [c for c in all_chunks if c.get("metadata", {}).get("doc_name") == target_doc]
    else:
        doc_chunks = all_chunks

    if not doc_chunks:
        doc_chunks = all_chunks

    # Sort chunks by page/slide and chunk index
    doc_chunks.sort(
        key=lambda c: (
            int(c.get("metadata", {}).get("page_or_slide", 0) or 0),
            int(c.get("metadata", {}).get("chunk_index", 0) or 0),
        )
    )

    # Pick spread-out chunks: up to max 6 chunks
    if len(doc_chunks) <= 6:
        selected_chunks = doc_chunks
    else:
        # Take first 2 chunks + evenly spread remaining up to 6 total
        selected_chunks = [doc_chunks[0], doc_chunks[1]]
        remaining = doc_chunks[2:]
        step = max(1, len(remaining) // 4)
        for i in range(0, len(remaining), step):
            if len(selected_chunks) < 6:
                selected_chunks.append(remaining[i])

    summary_system_prompt = (
        "Answer ONLY from the numbered excerpts. "
        "Summarize the key topics, main definitions, and core concepts. "
        "Be concise, at most 120 words unless asked for more. "
        "Cite like [1]. "
        "If the answer is not in the excerpts, reply exactly: "
        "I couldn't find this in your documents."
    )

    llm_resp = generate_answer(
        query=query,
        context_chunks=selected_chunks,
        system_prompt=summary_system_prompt,
    )

    if llm_resp.get("error"):
        return {
            "answer": "",
            "source_chips": _build_source_chips(selected_chunks),
            "not_found": False,
            "error": llm_resp["error"],
            "is_summary": True,
        }

    raw_answer = llm_resp.get("answer", "").strip()
    if llm_resp.get("abstained") or not raw_answer or NOT_FOUND_MESSAGE.lower() in raw_answer.lower():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "source_chips": [],
            "not_found": True,
            "error": None,
            "is_summary": True,
        }

    clean_answer = strip_invented_citations(raw_answer, len(selected_chunks))
    source_chips = _build_source_chips(selected_chunks)

    return {
        "answer": clean_answer,
        "source_chips": source_chips,
        "not_found": False,
        "error": None,
        "is_summary": True,
    }


def _build_source_chips(chunks: list[dict]) -> list[dict]:
    """
    Build source chips from retrieved chunks.
    Chips show: filename · p.4 (or slide / chapter).
    """
    chips = []
    seen = set()

    for idx, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        doc = meta.get("doc_name", "Document")
        page = meta.get("page_or_slide", 1)
        heading = meta.get("heading", "")

        label = f"{doc} · p.{page}"
        key = (doc, page)
        if key in seen:
            continue
        seen.add(key)

        chips.append({
            "index": idx,
            "label": label,
            "doc_name": doc,
            "page_or_slide": page,
            "heading": heading,
            "excerpt": c["text"],
        })

    return chips
