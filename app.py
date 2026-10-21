"""
ExamSense — Streamlit Application Entry Point
================================================
Single-page navigation with dark premium styling.
Sidebar: logo, navigation, API key field + pill with Test button, and Teacher mode toggle.
"""
from __future__ import annotations

import logging
import sys
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
from ui.components import status_pill
from ui.safe_page import safe_page
from core.generation.llm_client import get_active_groq_key, test_groq_connection

# ── Navigation Definition ──
# Library and Ask always; Evaluation only when Teacher mode is ON
nav_pages = [
    st.Page("pages/library.py", title="Library", icon="📚"),
    st.Page("pages/ask.py", title="Ask", icon="💬"),
]
if st.session_state.get("teacher_mode", False):
    nav_pages.append(st.Page("pages/evaluation.py", title="Evaluation", icon="🧪"))

pg = st.navigation(nav_pages)

# ── Sidebar ──
with st.sidebar:
    st.markdown("# 🎓 ExamSense")
    st.caption("AI-Powered Study Assistant")
    st.markdown("---")

    # API key input (optional user key)
    st.text_input(
        "Use your own Groq key (optional)",
        type="password",
        key="sidebar_user_groq_key",
        help="Free key at https://console.groq.com/keys. Overrides shared key and removes question limits.",
    )

    # API key status pill + Test button
    active_key = get_active_groq_key()
    has_key = bool(active_key)
    pill_label = "Ready" if has_key else "Add key"

    col_pill, col_test = st.columns([1, 1])
    with col_pill:
        st.markdown(status_pill(pill_label, is_ready=has_key), unsafe_allow_html=True)
    with col_test:
        if st.button("Test", key="btn_test_sidebar_groq", use_container_width=True):
            test_res = test_groq_connection(active_key)
            if test_res["ready"]:
                st.success("Connected!")
            else:
                st.error(test_res.get("status_text", "Failed"))

    st.markdown("---")

    # Teacher mode toggle (default OFF)
    st.toggle("Teacher mode", key="teacher_mode")

# ── Execute page run inside safe wrapper ──
safe_page(pg.run)
