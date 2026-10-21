"""
Ask Page — Chat Q&A with hybrid retrieval, source chips, and summary mode.
All helpers defined above first use. Answer caching in session state.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

from ui.components import hero_header, empty_state, error_card, not_found_card
from ui.safe_page import safe_page
from core.generation.answer_generator import generate_study_answer, NOT_FOUND_MESSAGE
from core.retrieval.vector_store import get_documents, get_document_names


# ── Render Helpers (DEFINED BEFORE ANY USE) ───────────────────────────────────

def _render_source_chips(source_chips: list[dict], message_idx: int) -> None:
    """Render clickable source chips that expand the source excerpt."""
    if not source_chips:
        return

    st.markdown("<div style='margin-top:10px;font-size:0.85rem;color:#A09CB5;font-weight:600'>Sources:</div>", unsafe_allow_html=True)
    for c_idx, chip in enumerate(source_chips):
        label = chip.get("label", "Source")
        heading = chip.get("heading")
        display_label = f"📄 {label}" + (f" · {heading}" if heading else "")
        with st.expander(display_label, expanded=False):
            st.markdown(
                f"<div style='color:#E2E0F0;font-size:0.85rem;line-height:1.5;white-space:pre-wrap;padding:4px 0'>"
                f"{chip.get('excerpt', '')}</div>",
                unsafe_allow_html=True,
            )


def _render_answer_payload(payload: dict, message_idx: int) -> None:
    """
    Render assistant response bubble.
    Guaranteed to never leave the assistant bubble empty.
    """
    error = payload.get("error")
    not_found = payload.get("not_found", False)
    answer = payload.get("answer", "").strip()
    source_chips = payload.get("source_chips", [])

    if error and not answer:
        st.markdown(error_card("Assistant Notice", error), unsafe_allow_html=True)
        if source_chips:
            _render_source_chips(source_chips, message_idx)
        return

    if not_found or (not answer and not error):
        st.markdown(not_found_card(NOT_FOUND_MESSAGE), unsafe_allow_html=True)
        return

    # Render main answer text
    st.markdown(answer)

    # Render source chips
    _render_source_chips(source_chips, message_idx)


def _process_user_question(question: str, selected_doc: str) -> None:
    """Execute question processing, checking session answer cache first."""
    if not question.strip():
        return

    # 1. Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    # 2. Check cache
    cache = st.session_state.setdefault("answer_cache", {})
    cache_key = (question.strip().lower(), selected_doc)

    if cache_key in cache:
        payload = cache[cache_key]
    else:
        doc_filter = None if selected_doc == "All documents" else selected_doc
        payload = generate_study_answer(query=question, doc_filter=doc_filter)
        cache[cache_key] = payload

    # 3. Add assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "payload": payload,
    })


# ── Main Page Logic ───────────────────────────────────────────────────────────

def _run_ask_page():
    st.markdown(
        hero_header("ExamSense Ask", "Grounded, concise answers with direct source citations."),
        unsafe_allow_html=True,
    )

    # ── Check Library ──
    docs = get_documents()
    if not docs:
        st.markdown(
            empty_state("📚", "No Documents Uploaded Yet", "Please upload study files in the Library before asking questions."),
            unsafe_allow_html=True,
        )
        if st.button("➡️ Go to Library", key="btn_goto_library", type="primary"):
            st.switch_page("pages/library.py")
        return

    # ── Document Selection ──
    doc_options = ["All documents"] + get_document_names()
    selected_doc = st.selectbox(
        "Search in document:",
        options=doc_options,
        key="ask_doc_selector",
    )

    # ── Initialize Chat History ──
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Empty Chat Starter Chips ──
    if not st.session_state.chat_history:
        st.markdown("<div style='margin:1.5rem 0 0.5rem 0;color:#A09CB5;font-size:0.9rem'>Get started with a quick prompt:</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        starters = [
            ("Summarize this document", col1, "starter_summarize"),
            ("List the key topics", col2, "starter_topics"),
            ("Explain the important definitions", col3, "starter_definitions"),
        ]
        for prompt_text, col, key in starters:
            with col:
                if st.button(prompt_text, key=key, use_container_width=True):
                    _process_user_question(prompt_text, selected_doc)
                    st.rerun()

    # ── Render Conversation History ──
    for idx, msg in enumerate(st.session_state.chat_history):
        role = msg["role"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(msg["content"])
            else:
                _render_answer_payload(msg.get("payload", {}), idx)

    # ── Chat Input ──
    user_prompt = st.chat_input("Ask a question about your study materials…", key="ask_chat_input")
    if user_prompt:
        _process_user_question(user_prompt, selected_doc)
        st.rerun()


# Wrap page render in safe_page
safe_page(_run_ask_page)
