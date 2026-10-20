"""
Exam Topics Page — pages/exam_topics.py
========================================
Visualises the exam-frequency index: hot topics, year distribution,
topic clusters sorted by how often they appear in past papers.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from ui.components import metric_card, empty_state, error_card, exam_frequency_badge
from ui.safe_page import safe_page
from core.exam_frequency.frequency_scorer import build_frequency_index
from core.exam_frequency.question_clusterer import get_all_clusters, get_frequency_summary


def _run_exam_topics_page():
    # ── Page Header ──
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <h2 style="background:linear-gradient(135deg,#7C3AED,#2DD4BF);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   font-weight:800;font-size:1.8rem;margin:0">
            📊 Exam Topics & Frequency
        </h2>
        <p style="color:#A09CB5;margin-top:0.3rem">
            Discover which topics appear most frequently in past exam papers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Build / Refresh Index ──
    col_build, _ = st.columns([1, 3])
    with col_build:
        if st.button("🔄 Rebuild Frequency Index", type="primary", key="btn_build_index"):
            with st.spinner("Extracting questions and clustering topics…"):
                try:
                    summary = build_frequency_index()
                    st.success(
                        f"Index built! {summary.get('total_questions', 0)} questions "
                        f"in {summary.get('total_clusters', 0)} clusters."
                    )
                    st.rerun()
                except Exception as e:
                    logger.error(f"Failed to build index: {e}", exc_info=True)
                    st.markdown(error_card("Build Failed", str(e)), unsafe_allow_html=True)

    st.markdown("---")

    # ── Summary Stats ──
    try:
        summary = get_frequency_summary()
    except Exception:
        summary = {}

    total_q = summary.get("total_questions", 0)
    total_c = summary.get("total_clusters", 0)
    top_clusters = summary.get("most_frequent") or summary.get("top_clusters", [])

    if total_q == 0:
        st.markdown(
            empty_state(
                "📊",
                "No Frequency Data",
                "Upload past exam papers in the Library tab, then click 'Rebuild Frequency Index' above.",
            ),
            unsafe_allow_html=True,
        )
        st.stop()

    # Stats cards
    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.markdown(metric_card(str(total_q), "Total Questions", "❓"), unsafe_allow_html=True)
    with stat2:
        st.markdown(metric_card(str(total_c), "Topic Clusters", "🗂️"), unsafe_allow_html=True)
    with stat3:
        max_freq = max((c.get("frequency", 0) for c in top_clusters), default=0)
        st.markdown(metric_card(str(max_freq), "Max Frequency", "🔥"), unsafe_allow_html=True)
    with stat4:
        years_set: set[str] = set()
        for c in top_clusters:
            years_set.update(c.get("years", []))
        st.markdown(metric_card(str(len(years_set)), "Years Covered", "📅"), unsafe_allow_html=True)

    st.markdown("---")

    # ── All Clusters ──
    try:
        clusters = get_all_clusters()
    except Exception as e:
        clusters = []
        st.markdown(error_card("Failed to load clusters", str(e)), unsafe_allow_html=True)

    if not clusters:
        st.info("No topic clusters found. Click 'Rebuild Frequency Index' to analyse past papers.")
        st.stop()

    # ── Filter & Sort Controls ──
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search_filter = st.text_input("🔍 Filter topics", placeholder="e.g. attention, BERT, retrieval…", key="topic_filter")
    with fc2:
        sort_by = st.selectbox("Sort by", ["Frequency ↓", "Occurrences ↓", "Alphabetical ↑"], key="topic_sort")
    with fc3:
        max_cluster_freq = max([c.get("frequency", 1) for c in clusters], default=1)
        if max_cluster_freq <= 1:
            min_freq = 1
            st.caption("Min frequency: 1")
        else:
            min_freq = st.slider("Min frequency", 1, max_cluster_freq, 1, key="min_freq")

    # Apply filters
    filtered = [
        c for c in clusters
        if c.get("frequency", 0) >= min_freq
        and (
            not search_filter
            or search_filter.lower() in c.get("representative_question", "").lower()
        )
    ]

    # Apply sort
    if sort_by == "Frequency ↓":
        filtered.sort(key=lambda c: c.get("frequency", 0), reverse=True)
    elif sort_by == "Occurrences ↓":
        filtered.sort(key=lambda c: c.get("total_occurrences", 0), reverse=True)
    else:
        filtered.sort(key=lambda c: c.get("representative_question", "").lower())

    st.caption(f"Showing {len(filtered)} of {len(clusters)} topic clusters")

    # ── Cluster Cards ──
    for cluster in filtered:
        freq = cluster.get("frequency", 0)
        total_occ = cluster.get("total_occurrences", 0)
        years = cluster.get("years", [])
        rep_q = cluster.get("representative_question", "")[:400]
        q_count = len(cluster.get("questions", []))

        # Color coding by frequency
        if freq >= 4:
            border_color = "rgba(244,63,94,0.4)"   # rose — very hot
            heat_icon = "🔥🔥"
        elif freq >= 2:
            border_color = "rgba(245,158,11,0.35)" # amber — hot
            heat_icon = "🔥"
        else:
            border_color = "rgba(124,58,237,0.25)" # violet — normal
            heat_icon = "📊"

        years_str = ", ".join(sorted(years)) if years else "—"
        badge_html = exam_frequency_badge(freq, years)

        with st.container():
            st.markdown(
                f"""<div class="glass-card" style="border-color:{border_color};margin-bottom:1rem">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.6rem">
                        <div style="flex:1">
                            <span style="font-size:1rem;font-weight:600;color:#E2E0F0">{heat_icon} {rep_q[:160]}{"..." if len(rep_q) > 160 else ""}</span>
                        </div>
                        <div style="margin-left:12px">{badge_html}</div>
                    </div>
                    <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:0.82rem;color:#6B6580">
                        <span>&#128197; Years: <strong style="color:#A09CB5">{years_str}</strong></span>
                        <span>&#128221; Papers: <strong style="color:#A09CB5">{total_occ}</strong></span>
                        <span>&#10067; Variants: <strong style="color:#A09CB5">{q_count}</strong></span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            with st.expander(f"View {q_count} question variant(s)", expanded=False):
                for q_item in cluster.get("questions", [])[:10]:
                    src = q_item.get("source_document", "?")
                    yr = q_item.get("year", "")
                    q_text = q_item.get("question_text", "")[:300]
                    st.markdown(
                        f"""<div style="padding:0.5rem 0.8rem;border-left:3px solid rgba(124,58,237,0.4);
                            margin-bottom:0.5rem;border-radius:0 4px 4px 0;background:rgba(30,22,56,0.4)">
                            <div style="font-size:0.82rem;color:#6B6580;margin-bottom:3px">
                                📄 {src}{f' [{yr}]' if yr else ''}
                            </div>
                            <div style="color:#C4C0D8;font-size:0.88rem">{q_text}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    # ── Year Distribution Chart ──
    st.markdown("---")
    st.markdown("### 📅 Year Distribution")

    year_counts: dict[str, int] = {}
    for cluster in clusters:
        for yr in cluster.get("years", []):
            year_counts[yr] = year_counts.get(yr, 0) + cluster.get("frequency", 1)

    if year_counts:
        try:
            import plotly.express as px
            import pandas as pd

            df = pd.DataFrame(
                [{"Year": k, "Topics": v} for k, v in sorted(year_counts.items())]
            )
            fig = px.bar(
                df, x="Year", y="Topics",
                color="Topics",
                color_continuous_scale=["#7C3AED", "#2DD4BF"],
                title="Topics per Year",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,11,30,0.5)",
                font={"color": "#E2E0F0", "family": "Inter"},
                title_font_color="#E2E0F0",
                coloraxis_showscale=False,
            )
            fig.update_traces(marker_line_color="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            for yr, cnt in sorted(year_counts.items()):
                st.write(f"**{yr}**: {cnt} topics")


safe_page(_run_exam_topics_page)
