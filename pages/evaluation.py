"""
Evaluation Page (Teacher Mode Only)
===================================
Measures retrieval quality (Precision@3 and Recall@3) across Dense, BM25, and Hybrid methods.
Zero LLM calls. No Plotly dependencies.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

from ui.components import empty_state, error_card
from ui.safe_page import safe_page
from core.evaluation.eval_runner import run_retrieval_evaluation
from core.retrieval.vector_store import get_document_names, get_documents


# ── Render Helpers (DEFINED BEFORE USE) ────────────────────────────────────────

def _parse_uploaded_eval_file(uploaded_file) -> list[dict]:
    """Parse JSON or CSV upload into a standardized eval test set."""
    cases = []
    fname = uploaded_file.name.lower()
    raw_content = uploaded_file.read().decode("utf-8", errors="replace")

    if fname.endswith(".json"):
        data = json.loads(raw_content)
        if isinstance(data, list):
            for item in data:
                q = item.get("question", "").strip()
                d = item.get("expected_doc", item.get("doc_name", "")).strip()
                p = item.get("expected_page", item.get("page_or_slide"))
                if q and d:
                    cases.append({"question": q, "expected_doc": d, "expected_page": p})
    elif fname.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(raw_content))
        for row in reader:
            q = (row.get("question") or row.get("query") or "").strip()
            d = (row.get("expected_doc") or row.get("doc_name") or row.get("document") or "").strip()
            p = row.get("expected_page") or row.get("page") or row.get("slide")
            if q and d:
                cases.append({"question": q, "expected_doc": d, "expected_page": p})

    return cases


def _build_eval_set_download_payload(eval_cases: list[dict]) -> str:
    """Format eval set as pretty JSON for download."""
    return json.dumps(eval_cases, indent=2)


# ── Main Page Logic ───────────────────────────────────────────────────────────

def _run_evaluation_page():
    # ── Teacher Mode Guard ──
    if not st.session_state.get("teacher_mode", False):
        st.info("Teacher Mode is currently OFF. Enable Teacher Mode in the sidebar to access Evaluation.")
        return

    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <h2 style="background:linear-gradient(135deg,#7C3AED,#2DD4BF);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         font-weight:800;font-size:1.8rem;margin:0">🧪 Retrieval Evaluation</h2>
      <p style="color:#A09CB5;margin-top:0.3rem">
        Measure retrieval quality (Precision@3 and Recall@3) across Dense, BM25, and Hybrid RRF. No LLM calls.
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Check Library Content ──
    docs = get_documents()
    if not docs:
        st.markdown(
            empty_state("📁", "No Documents Uploaded", "Please upload study files in the Library before running retrieval evaluation."),
            unsafe_allow_html=True,
        )
        return

    # Initialize eval set in session state
    if "eval_cases" not in st.session_state:
        st.session_state.eval_cases = []

    # ── Input Section ──
    input_tab, upload_tab = st.tabs(["📝 Form Entry", "📂 Upload JSON / CSV"])

    with input_tab:
        doc_names = get_document_names()
        with st.form("form_add_eval_case", clear_on_submit=True):
            col_q, col_d, col_p = st.columns([3, 2, 1])
            with col_q:
                q_input = st.text_input("Question *", placeholder="e.g. What is gradient descent?")
            with col_d:
                doc_choice = st.selectbox("Expected Document *", options=doc_names)
            with col_p:
                page_input = st.text_input("Expected Page/Slide", placeholder="e.g. 4")

            submitted = st.form_submit_button("➕ Add Test Case")
            if submitted:
                if q_input.strip() and doc_choice:
                    st.session_state.eval_cases.append({
                        "question": q_input.strip(),
                        "expected_doc": doc_choice.strip(),
                        "expected_page": page_input.strip() if page_input.strip() else None,
                    })
                    st.success("Added test case.")
                else:
                    st.error("Question and Expected Document are required.")

    with upload_tab:
        uploaded_eval_file = st.file_uploader(
            "Upload JSON or CSV eval set",
            type=["json", "csv"],
            key="uploader_eval_file",
            help="JSON: [{'question': '...', 'expected_doc': '...', 'expected_page': 1}] or CSV with headers: question, expected_doc, expected_page",
        )
        if uploaded_eval_file:
            if st.button("📥 Load Uploaded Test Cases", key="btn_load_eval_cases"):
                try:
                    loaded = _parse_uploaded_eval_file(uploaded_eval_file)
                    if loaded:
                        st.session_state.eval_cases.extend(loaded)
                        st.success(f"Loaded {len(loaded)} test cases.")
                    else:
                        st.warning("No valid test cases found in uploaded file.")
                except Exception as e:
                    st.markdown(error_card("Parse Error", str(e)), unsafe_allow_html=True)

    # ── Display Current Eval Set ──
    cases = st.session_state.eval_cases
    st.markdown(f"### Current Test Set ({len(cases)} cases)")

    if cases:
        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            json_data = _build_eval_set_download_payload(cases)
            st.download_button(
                "⬇️ Download my eval set",
                data=json_data,
                file_name="examsense_eval_set.json",
                mime="application/json",
                key="btn_download_eval_set",
                use_container_width=True,
            )
        with col_clr:
            if st.button("🗑️ Clear Test Set", key="btn_clear_eval_set", use_container_width=True):
                st.session_state.eval_cases = []
                st.rerun()

        # Small preview
        for i, c in enumerate(cases[:5], 1):
            p_str = f" · p.{c['expected_page']}" if c.get('expected_page') else ""
            st.caption(f"**#{i}** {c['question']} *(Target: {c['expected_doc']}{p_str})*")
        if len(cases) > 5:
            st.caption(f"*(and {len(cases)-5} more...)*")

        st.markdown("---")

        # ── Run Evaluation Button ──
        if st.button("▶️ Run Retrieval Evaluation (k=3)", type="primary", key="btn_run_eval"):
            with st.spinner("Computing retrieval metrics across Dense, BM25, and Hybrid methods…"):
                eval_results = run_retrieval_evaluation(cases, k=3)
                st.session_state["last_eval_results"] = eval_results

    # ── Display Results ──
    results = st.session_state.get("last_eval_results")
    if results and results.get("summary"):
        st.markdown("### Evaluation Results (k = 3)")
        summary = results["summary"]

        # Table of Precision@3 and Recall@3
        table_rows = []
        chart_data = {"Precision@3": {}, "Recall@3": {}}

        for method, metrics in summary.items():
            p3 = metrics["Precision@3"]
            r3 = metrics["Recall@3"]
            table_rows.append({"Method": method, "Precision@3": f"{p3:.2%}", "Recall@3": f"{r3:.2%}"})
            chart_data["Precision@3"][method] = p3
            chart_data["Recall@3"][method] = r3

        st.table(table_rows)

        # Built-in Streamlit bar chart (No Plotly needed)
        st.markdown("#### Performance Comparison")
        import pandas as pd
        df_chart = pd.DataFrame(chart_data)
        st.bar_chart(df_chart)


# Wrap page render in safe_page
safe_page(_run_evaluation_page)
