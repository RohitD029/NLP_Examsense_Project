"""
Ask Page — Chat Q&A with hybrid RAG, confidence gauge, citations, and summary mode.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

from ui.components import (
    hero_header, citation_chips, confidence_gauge,
    exam_frequency_badge, not_found_card, error_card, empty_state,
)
from ui.safe_page import safe_page
from core.generation.answer_generator import answer_query
from core.exam_frequency.frequency_scorer import score_query
from core.generation.llm_client import check_llm_status
from core.retrieval.vector_store import get_collection_stats, get_document_names


# ── Render Helpers (MUST BE DEFINED BEFORE CALLING) ───────────────────────────

def _render_citations(result: dict) -> None:
    citations = result.get("citations", [])
    if not citations:
        return
    st.markdown(citation_chips(citations), unsafe_allow_html=True)
    with st.expander(f"📄 {len(citations)} source(s)", expanded=False):
        for cit in citations:
            score_val = cit.get("dense_score", 0.0)
            score_str = f"score: {score_val:.3f}" if score_val > 0 else "source"
            st.markdown(
                f"""<div class="glass-card" style="padding:.7rem 1rem;margin:.3rem 0">
                <div style="color:#A78BFA;font-weight:600;font-size:.88rem">
                [{cit['index']}] {cit['doc_name']} · p.{cit['page_or_slide']}
                {'· ' + cit['heading'] if cit.get('heading') else ''}
                <span style="float:right;color:#6B6580;font-size:.78rem">{score_str}</span></div>
                <p style="color:#A09CB5;font-size:.82rem;margin:.4rem 0 0">{cit['text_excerpt']}</p>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_passages(result: dict) -> None:
    chunks = result.get("retrieved_chunks", [])
    if not chunks:
        return
    with st.expander(f"📄 {len(chunks)} retrieved passage(s)", expanded=True):
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            st.markdown(
                f"""<div style="border-left:3px solid #7C3AED;padding:.5rem .8rem;
                margin-bottom:.5rem;border-radius:0 6px 6px 0;background:rgba(30,22,56,.4)">
                <div style="color:#A78BFA;font-size:.82rem;font-weight:600">
                [{i}] {meta.get('doc_name','?')} · p.{meta.get('page_or_slide','?')}
                </div>
                <div style="color:#C4C0D8;font-size:.85rem;margin-top:4px">
                {chunk['text'][:400]}{'…' if len(chunk['text'])>400 else ''}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_result(result: dict) -> None:
    """Render an answer result in the current chat_message context. Never leaves bubble blank."""
    conf = result.get("confidence", 0.0)
    freq = result.get("exam_frequency")
    abstained = result.get("abstained", False)
    answer = result.get("answer", "").strip()
    error = result.get("error", "").strip()
    is_summary = result.get("retrieval_info", {}).get("is_summary_mode", False)

    # Confidence + freq row
    gauge = confidence_gauge(conf)
    freq_badge = exam_frequency_badge(
        freq.get("frequency", 0), freq.get("years", [])
    ) if freq else ""
    summary_pill = (
        '<span style="background:rgba(45,212,191,0.15);color:#2DD4BF;padding:3px 10px;'
        'border-radius:12px;font-size:0.75rem;font-weight:600;border:1px solid rgba(45,212,191,0.3)">'
        '📑 Document Summary</span>'
    ) if is_summary else ""

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap">'
        f'{gauge}{freq_badge}{summary_pill}</div>',
        unsafe_allow_html=True,
    )

    if error and not answer:
        st.markdown(error_card("LLM Unavailable", error), unsafe_allow_html=True)
        _render_passages(result)
        return

    if abstained or not answer:
        st.markdown(not_found_card(), unsafe_allow_html=True)
        _render_passages(result)
        return

    st.markdown(answer)
    _render_citations(result)

    model_info = result.get("model_info", {})
    if model_info.get("model"):
        st.caption(f"via {model_info.get('provider', 'Groq')} · {model_info.get('model', '')}")


def _render_answer(msg: dict) -> None:
    """Re-render a stored chat message from session state."""
    meta = msg.get("meta", {})
    if meta:
        _render_result(meta)
    else:
        content = msg.get("content", "").strip()
        if content:
            st.markdown(content)
        else:
            st.markdown("*(No response content recorded)*")


# ── Main Page Logic ───────────────────────────────────────────────────────────

def _run_ask_page():
    st.markdown(
        hero_header("ExamSense Ask", "Grounded, cited answers from your course material."),
        unsafe_allow_html=True,
    )

    # ── LLM Status banner ──
    llm_status = check_llm_status()
    if not llm_status.get("ready"):
        reason = llm_status.get("reason", "API key not configured.")
        st.warning(
            f"⚠️ **LLM Status:** {reason} — "
            "Retrieved passages will still be shown so you can read relevant notes directly."
        )

    # ── Empty library guard ──
    try:
        cm_stats = get_collection_stats("course_material")
        pp_stats = get_collection_stats("past_papers")
        total_docs = cm_stats["total_documents"] + pp_stats["total_documents"]
    except Exception:
        total_docs = 0

    if total_docs == 0:
        st.markdown(
            empty_state("📭", "No Documents Uploaded",
                        "Go to the Library page and upload your course materials first."),
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Settings & Document Filter ──
    with st.expander("⚙️ Query Settings", expanded=False):
        sc1, sc2 = st.columns(2)
        with sc1:
            collection = st.selectbox(
                "Search Collection",
                ["course_material", "past_papers"],
                format_func=lambda x: "📚 Course Material" if x == "course_material" else "📝 Past Papers",
                key="ask_collection",
            )
        with sc2:
            mode = st.selectbox(
                "Retrieval Mode",
                ["hybrid", "dense_only", "sparse_only"],
                format_func=lambda x: {"hybrid": "🔀 Hybrid", "dense_only": "🧲 Dense", "sparse_only": "🔤 BM25"}[x],
                key="ask_mode",
            )

        # Document Filter
        available_docs = ["All documents"] + get_document_names(collection)
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            doc_filter = st.selectbox(
                "Document Filter",
                options=available_docs,
                key="ask_doc_filter",
                help="Restrict search or summarize a single document",
            )
        with fc2:
            top_k = st.slider("Top-K chunks", 2, 8, 4, key="ask_topk")

        show_freq = st.checkbox("Show exam frequency badge", value=True, key="ask_show_freq")

    st.markdown("---")

    # ── Session history ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Example questions ──
    if not st.session_state.chat_history:
        st.markdown("**Try an example question:**")
        examples = [
            "What is this document about?",
            "What is the attention mechanism?",
            "Explain BM25 ranking.",
            "Summarize the key topics.",
        ]
        ex_cols = st.columns(2)
        for i, q in enumerate(examples):
            if ex_cols[i % 2].button(q, key=f"ask_ex_{i}", use_container_width=True):
                st.session_state["ask_pending"] = q
                st.rerun()

    # ── Render chat history ──
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg.get("content", ""))
            else:
                _render_answer(msg)

    # ── Chat input ──
    pending = st.session_state.pop("ask_pending", None)
    user_q = st.chat_input("Ask a question from your course material…") or pending

    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing documents…"):
                try:
                    result = answer_query(
                        query=user_q,
                        collection=collection,
                        top_k=top_k,
                        mode=mode,
                        doc_filter=doc_filter if doc_filter != "All documents" else None,
                    )
                    freq_data = score_query(user_q) if show_freq else None
                    result["exam_frequency"] = freq_data

                    # Render inline
                    _render_result(result)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "meta": result,
                    })

                except Exception as e:
                    logger.error(f"Ask failed: {e}", exc_info=True)
                    st.markdown(error_card("Query Failed", str(e)), unsafe_allow_html=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "",
                        "meta": {
                            "confidence": 0,
                            "is_grounded": False,
                            "citations": [],
                            "abstained": True,
                            "error": str(e),
                        },
                    })

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="ask_clear_btn"):
            st.session_state.chat_history = []
            st.rerun()


# Wrap with safe_page to prevent raw tracebacks
safe_page(_run_ask_page)
