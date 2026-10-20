"""
ExamSense — Streamlit Application Entry Point
================================================
Single-page-app with navigation via st.navigation / st.Page.
Injects custom CSS, renders sidebar with stats, single source of truth settings,
and connection testing.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import streamlit as st

# ── Ensure project root is on sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Configure logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("examsense.app")

# ── Preload persisted indices ──
try:
    from core.exam_frequency.question_clusterer import load_clusters
    load_clusters()
except Exception as e:
    logger.debug(f"Preload clusters skipped: {e}")

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="ExamSense",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ──
css_path = PROJECT_ROOT / "ui" / "styles.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Import UI helpers ──
from ui.components import status_pill, sidebar_stat
from ui.safe_page import safe_page
from core.generation.llm_client import check_llm_status, test_groq_connection
from core.config import settings

# Pre-populate session state API key from settings/.env before widgets render
if "sidebar_groq_key" not in st.session_state:
    st.session_state["sidebar_groq_key"] = settings.groq_api_key

# ── Sidebar ──
with st.sidebar:
    st.markdown("# 🎓 ExamSense")
    st.caption("AI-Powered Study Assistant")
    st.markdown("---")

    # Collection stats
    try:
        from core.retrieval.vector_store import get_collection_stats
        cm_stats = get_collection_stats("course_material")
        pp_stats = get_collection_stats("past_papers")
    except Exception:
        cm_stats = {"total_documents": 0, "total_chunks": 0}
        pp_stats = {"total_documents": 0, "total_chunks": 0}

    st.markdown(sidebar_stat("📚 Course Docs", cm_stats.get("total_documents", 0)), unsafe_allow_html=True)
    st.markdown(sidebar_stat("📝 Past Papers", pp_stats.get("total_documents", 0)), unsafe_allow_html=True)
    st.markdown(
        sidebar_stat("🧩 Total Chunks", cm_stats.get("total_chunks", 0) + pp_stats.get("total_chunks", 0)),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Model status
    llm_status = check_llm_status()
    if llm_status.get("ready"):
        st.markdown(status_pill(llm_status.get("status_text", "✓ Ready"), is_ready=True), unsafe_allow_html=True)
    else:
        st.markdown(status_pill(llm_status.get("status_text", "✗ Not Ready"), is_ready=False), unsafe_allow_html=True)
        reason = llm_status.get("reason", "")
        if reason:
            st.caption(f"{reason}")

    # Settings expander
    with st.expander("⚙️ Settings", expanded=False):
        def _on_key_change():
            new_key = st.session_state.get("sidebar_groq_key", "").strip()
            settings.groq_api_key = new_key
            st.session_state["groq_conn_status"] = {"key": new_key, "status": test_groq_connection(new_key)}

        st.text_input(
            "Groq API Key",
            type="password",
            key="sidebar_groq_key",
            on_change=_on_key_change,
            help="Get free key at https://console.groq.com/keys",
        )

        if st.button("🔄 Test Connection", key="btn_test_sidebar_conn", use_container_width=True):
            curr_key = st.session_state.get("sidebar_groq_key", "").strip()
            status = test_groq_connection(curr_key)
            st.session_state["groq_conn_status"] = {"key": curr_key, "status": status}
            if status.get("ready"):
                st.success(f"Connected! Model: {status.get('model')}")
            else:
                st.error(status.get("reason"))
            st.rerun()

        st.text_input(
            "Gemini API Key (optional)",
            value=settings.gemini_api_key,
            type="password",
            key="sidebar_gemini_key",
            help="Get free key at https://aistudio.google.com/",
        )

        def _on_threshold_change():
            settings.confidence_threshold = st.session_state.sidebar_threshold

        st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.8,
            value=settings.confidence_threshold,
            step=0.05,
            key="sidebar_threshold",
            on_change=_on_threshold_change,
            help="Below this similarity → abstain from answering",
        )

        st.caption(f"Model: {settings.groq_primary_model}")
        st.caption(f"Embedding: {settings.embedding_model_name}")
        st.caption(f"Chunk size: {settings.max_chunk_chars} chars")


# ── Navigation ──
pages = {
    "📚 Library": st.Page("pages/library.py", title="Library", icon="📚"),
    "💬 Ask": st.Page("pages/ask.py", title="Ask", icon="💬"),
    "📊 Exam Topics": st.Page("pages/exam_topics.py", title="Exam Topics", icon="📊"),
    "🧪 Evaluation": st.Page("pages/evaluation.py", title="Evaluation", icon="🧪"),
}

pg = st.navigation(list(pages.values()))

# Execute navigation run inside safe wrapper
safe_page(pg.run)
