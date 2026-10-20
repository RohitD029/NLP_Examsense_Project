"""
Evaluation Page — pages/evaluation.py
=======================================
Interactive evaluation harness: run a test set through the retrieval
+ generation pipeline, view aggregate metrics (Precision@K, Recall@K,
MRR, BLEU, ROUGE), and compare retrieval modes (hybrid vs dense vs sparse).
"""
from __future__ import annotations

import sys
import logging
import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from ui.components import metric_card, error_card, empty_state
from ui.safe_page import safe_page
from core.evaluation.eval_runner import run_evaluation


def _run_evaluation_page():
    # ── Page Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <h2 style="background:linear-gradient(135deg,#7C3AED,#2DD4BF);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-weight:800;font-size:1.8rem;margin:0">
            🧪 System Evaluation
        </h2>
        <p style="color:#A09CB5;margin-top:0.3rem">
            Measure retrieval and generation quality with your own test set.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Configuration ──────────────────────────────────────────────────────────
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        eval_collection = st.selectbox(
            "Collection",
            options=["course_material", "past_papers"],
            format_func=lambda x: "📚 Course Material" if x == "course_material" else "📝 Past Papers",
            key="eval_collection",
        )
    with cfg2:
        eval_mode = st.selectbox(
            "Retrieval Mode",
            options=["hybrid", "dense_only", "sparse_only"],
            format_func=lambda x: {
                "hybrid": "🔀 Hybrid",
                "dense_only": "🧲 Dense Only",
                "sparse_only": "🔤 BM25 Only",
            }[x],
            key="eval_mode",
        )
    with cfg3:
        k_values_input = st.multiselect(
            "K Values (Precision/Recall@K)",
            options=[1, 3, 5, 10],
            default=[1, 3, 5],
            key="eval_k_values",
        )
        if not k_values_input:
            k_values_input = [1, 3, 5]

    st.markdown("---")

    # ── Test Set Input ─────────────────────────────────────────────────────────
    st.markdown("### Test Set")

    input_tab, json_tab, demo_tab = st.tabs(["📝 Manual Entry", "📋 Paste JSON", "🎯 Demo Test Set"])

    test_set = []

    # ── Manual Entry ──────────────────────────────────────────────────────────
    with input_tab:
        st.markdown(
            """<p style="color:#A09CB5;font-size:0.88rem">
            Add test cases manually. <em>Expected Chunk IDs</em> are optional — if provided,
            Precision@K and Recall@K are computed. <em>Expected Answer</em> is optional —
            used for token overlap F1 computation.
            </p>""",
            unsafe_allow_html=True,
        )

        if "manual_test_cases" not in st.session_state:
            st.session_state.manual_test_cases = [
                {"question": "", "expected_chunk_ids": "", "expected_answer": ""}
            ]

        for i, tc in enumerate(st.session_state.manual_test_cases):
            with st.expander(f"Test case #{i + 1}" + (f": {tc['question'][:50]}…" if tc["question"] else ""), expanded=i == 0):
                tc["question"] = st.text_input("Question *", value=tc["question"], key=f"tc_q_{i}")
                tc["expected_chunk_ids"] = st.text_input(
                    "Expected Chunk IDs (comma-separated, optional)",
                    value=tc["expected_chunk_ids"],
                    key=f"tc_ids_{i}",
                    placeholder="e.g. chunk_abc123, chunk_def456",
                )
                tc["expected_answer"] = st.text_area(
                    "Expected Answer (optional)",
                    value=tc["expected_answer"],
                    key=f"tc_ans_{i}",
                    height=80,
                )
                if st.button("🗑️ Remove", key=f"rm_tc_{i}"):
                    st.session_state.manual_test_cases.pop(i)
                    st.rerun()

        if st.button("➕ Add Test Case", key="btn_add_tc"):
            st.session_state.manual_test_cases.append(
                {"question": "", "expected_chunk_ids": "", "expected_answer": ""}
            )
            st.rerun()

        # Build test_set from manual entries
        for tc in st.session_state.manual_test_cases:
            if tc["question"].strip():
                ids = [x.strip() for x in tc["expected_chunk_ids"].split(",") if x.strip()]
                test_set.append({
                    "question": tc["question"].strip(),
                    "expected_chunk_ids": ids,
                    "expected_answer": tc["expected_answer"].strip(),
                })

    # ── JSON Paste ────────────────────────────────────────────────────────────
    with json_tab:
        st.markdown(
            """<p style="color:#A09CB5;font-size:0.88rem">
            Paste a JSON array of test cases. Format:
            </p>""",
            unsafe_allow_html=True,
        )
        st.code(
            """[
  {
    "question": "What is attention in transformers?",
    "expected_chunk_ids": ["chunk_abc123"],
    "expected_answer": "Attention allows the model to focus..."
  }
]""",
            language="json",
        )
        json_input = st.text_area(
            "Paste JSON here",
            height=200,
            key="eval_json_input",
            placeholder='[{"question": "...", "expected_chunk_ids": [], "expected_answer": ""}]',
        )
        if json_input.strip():
            try:
                parsed = json.loads(json_input)
                if isinstance(parsed, list):
                    test_set = parsed
                    st.success(f"✅ Loaded {len(parsed)} test case(s) from JSON.")
                else:
                    st.error("JSON must be an array of test case objects.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # ── Demo Test Set ──────────────────────────────────────────────────────────
    with demo_tab:
        st.markdown(
            """<p style="color:#A09CB5;font-size:0.88rem">
            Load a built-in demo test set to verify the pipeline works end-to-end.
            </p>""",
            unsafe_allow_html=True,
        )
        demo_questions = [
            "What is the attention mechanism and how does it work?",
            "Explain the difference between precision and recall.",
            "What is TF-IDF and how is it used in information retrieval?",
            "Describe the transformer architecture.",
            "What is BERT and how is it pre-trained?",
        ]
        st.markdown("**Demo questions:**")
        for q in demo_questions:
            st.markdown(f"- {q}")

        if st.button("📂 Load Demo Test Set", key="btn_load_demo"):
            test_set = [{"question": q, "expected_chunk_ids": [], "expected_answer": ""} for q in demo_questions]
            st.session_state.demo_test_set = test_set
            st.success(f"Loaded {len(test_set)} demo questions.")

        if "demo_test_set" in st.session_state and not test_set:
            test_set = st.session_state.demo_test_set

    st.markdown("---")

    # ── Run Evaluation ─────────────────────────────────────────────────────────
    st.markdown(f"**Ready to evaluate:** {len(test_set)} test case(s)")

    if not test_set:
        st.markdown(
            empty_state("🧪", "No test cases", "Add test cases manually, paste JSON, or load the demo set above."),
            unsafe_allow_html=True,
        )
    else:
        if st.button("▶️ Run Evaluation", type="primary", key="btn_run_eval", disabled=len(test_set) == 0):
            progress_bar = st.progress(0, text="Initialising…")

            def progress_cb(current: int, total: int):
                pct = int((current / max(total, 1)) * 100)
                progress_bar.progress(pct, text=f"Evaluating {current}/{total}…")

            try:
                with st.spinner("Running evaluation pipeline…"):
                    results = run_evaluation(
                        test_set=test_set,
                        collection_name=eval_collection,
                        k_values=k_values_input,
                        mode=eval_mode,
                        progress_callback=progress_cb,
                    )
                progress_bar.progress(100, text="Done!")
                st.session_state.eval_results = results
            except Exception as e:
                logger.error(f"Evaluation failed: {e}", exc_info=True)
                st.markdown(error_card("Evaluation Failed", str(e)), unsafe_allow_html=True)

    # ── Show Results ───────────────────────────────────────────────────────────
    if "eval_results" in st.session_state:
        results = st.session_state.eval_results

        st.markdown("---")
        st.markdown("## 📈 Results")

        total = results.get("total_cases", 0)
        completed = results.get("completed", 0)
        failed = results.get("failed", 0)
        elapsed = results.get("elapsed_seconds", 0)

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(metric_card(str(total), "Total Cases", "📋"), unsafe_allow_html=True)
        with sc2:
            st.markdown(metric_card(str(completed), "Completed", "✅"), unsafe_allow_html=True)
        with sc3:
            st.markdown(metric_card(str(failed), "Failed", "❌"), unsafe_allow_html=True)
        with sc4:
            st.markdown(metric_card(f"{elapsed}s", "Elapsed", "⏱️"), unsafe_allow_html=True)

        st.markdown("---")

        agg = results.get("aggregate_metrics", {})
        if agg:
            st.markdown("### Aggregate Metrics")
            retrieval_metrics = {k: v for k, v in agg.items() if any(x in k for x in ["precision", "recall", "mrr", "ndcg"])}
            generation_metrics = {k: v for k, v in agg.items() if any(x in k for x in ["bleu", "rouge", "f1"])}
            other_metrics = {k: v for k, v in agg.items() if k not in retrieval_metrics and k not in generation_metrics}

            if retrieval_metrics:
                st.markdown("**Retrieval Metrics**")
                rm_cols = st.columns(min(len(retrieval_metrics), 4))
                for idx, (k, v) in enumerate(sorted(retrieval_metrics.items())):
                    label = k.replace("_", " ").title()
                    rm_cols[idx % 4].metric(label, f"{v:.4f}")

            if generation_metrics:
                st.markdown("**Generation Metrics**")
                gm_cols = st.columns(min(len(generation_metrics), 4))
                for idx, (k, v) in enumerate(sorted(generation_metrics.items())):
                    label = k.replace("_", " ").title()
                    gm_cols[idx % 4].metric(label, f"{v:.4f}")

            if other_metrics:
                st.markdown("**Other Metrics**")
                om_cols = st.columns(min(len(other_metrics), 4))
                for idx, (k, v) in enumerate(sorted(other_metrics.items())):
                    label = k.replace("_", " ").title()
                    om_cols[idx % 4].metric(label, f"{v:.4f}")

        # Plotly chart
        if agg:
            try:
                import plotly.graph_objects as go

                labels = list(agg.keys())
                values = [agg[k] for k in labels]
                labels_fmt = [l.replace("_", " ").title() for l in labels]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=labels_fmt,
                    y=values,
                    marker=dict(
                        color=values,
                        colorscale=[[0, "#7C3AED"], [1, "#2DD4BF"]],
                        showscale=False,
                    ),
                    text=[f"{v:.3f}" for v in values],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Aggregate Evaluation Metrics",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,11,30,0.5)",
                    font={"color": "#E2E0F0", "family": "Inter"},
                    title_font_color="#E2E0F0",
                    yaxis=dict(range=[0, max(1.0, max(values) * 1.15)], gridcolor="rgba(124,58,237,0.1)"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        st.markdown("---")
        st.markdown("### Per-Question Results")

        per_q = results.get("per_question_results", [])
        for i, qr in enumerate(per_q):
            status_icon = "✅" if qr["status"] == "success" else "❌"
            confidence = qr.get("confidence", 0)
            is_grounded = qr.get("is_grounded", False)

            with st.expander(f"{status_icon} Q{i+1}: {qr['question'][:80]}…", expanded=False):
                if qr["status"] == "error":
                    st.error(f"Error: {qr.get('error', 'Unknown')}")
                    continue

                pq_cols = st.columns([2, 1])
                with pq_cols[0]:
                    st.markdown("**Generated Answer:**")
                    st.markdown(
                        f'<div style="background:rgba(30,22,56,0.5);border-radius:8px;padding:0.8rem;'
                        f'color:#C4C0D8;font-size:0.88rem">{qr.get("generated_answer", "—")}</div>',
                        unsafe_allow_html=True,
                    )
                    if qr.get("expected_answer"):
                        st.markdown("**Expected Answer:**")
                        st.markdown(
                            f'<div style="background:rgba(30,22,56,0.5);border-radius:8px;padding:0.8rem;'
                            f'color:#A09CB5;font-size:0.88rem">{qr["expected_answer"]}</div>',
                            unsafe_allow_html=True,
                        )

                with pq_cols[1]:
                    st.metric("Confidence", f"{confidence:.2f}")
                    st.metric("Grounded", "Yes" if is_grounded else "No")

                q_metrics = qr.get("metrics", {})
                if q_metrics:
                    st.markdown("**Metrics:**")
                    m_cols = st.columns(min(len(q_metrics), 4))
                    for idx, (mk, mv) in enumerate(sorted(q_metrics.items())):
                        m_cols[idx % 4].metric(mk.replace("_", " ").title(), f"{mv:.4f}")

        st.markdown("---")
        st.download_button(
            "📥 Download Full Results (JSON)",
            data=json.dumps(results, indent=2, default=str),
            file_name="examsense_eval_results.json",
            mime="application/json",
            key="btn_download_eval",
        )


safe_page(_run_evaluation_page)
