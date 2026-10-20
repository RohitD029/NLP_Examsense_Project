"""
Library Page — upload, browse, delete documents.
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

from ui.components import metric_card, doc_tile, empty_state, error_card
from ui.safe_page import safe_page
from core.ingestion.document_processor import (
    process_document, save_uploaded_file,
    get_supported_extensions, detect_paper_metadata,
)
from core.retrieval.vector_store import (
    add_chunks, get_collection_stats, get_all_chunks,
    delete_document, document_exists,
)
from core.retrieval.bm25_index import rebuild_from_store
from core.ocr_helper import tesseract_available, tesseract_install_instructions


def _run_library_page():
    # ── Header ──
    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <h2 style="background:linear-gradient(135deg,#7C3AED,#2DD4BF);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         font-weight:800;font-size:1.8rem;margin:0">📚 Document Library</h2>
      <p style="color:#A09CB5;margin-top:0.3rem">
        Upload and manage your course materials and past papers.</p>
    </div>""", unsafe_allow_html=True)

    if not tesseract_available():
        with st.expander("⚠️ OCR unavailable — scanned PDFs will yield no text", expanded=False):
            st.markdown(tesseract_install_instructions())

    # ── Stats ──
    try:
        cm = get_collection_stats("course_material")
        pp = get_collection_stats("past_papers")
    except Exception:
        cm = pp = {"total_documents": 0, "total_chunks": 0}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card(str(cm.get("total_documents", 0)), "Course Docs", "📚"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card(str(pp.get("total_documents", 0)), "Past Papers", "📝"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card(str(cm.get("total_chunks", 0)), "Course Chunks", "🧩"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card(str(pp.get("total_chunks", 0)), "Paper Chunks", "🗂️"), unsafe_allow_html=True)

    st.markdown("---")

    upload_tab, browse_tab = st.tabs(["⬆️ Upload", "📂 Browse Library"])

    # ════════════════════════════════════════════════════════════════════════
    # UPLOAD TAB
    # ════════════════════════════════════════════════════════════════════════
    with upload_tab:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            collection_choice = st.radio(
                "Collection", ["course_material", "past_papers"],
                format_func=lambda x: "📚 Course Material" if x == "course_material" else "📝 Past Papers",
                horizontal=True, key="lib_collection_radio",
            )
        with col_r:
            if collection_choice == "past_papers":
                year_input = st.text_input("Year (optional)", placeholder="2023", key="lib_year_input")
                subject_input = st.text_input("Subject (optional)", placeholder="NLP", key="lib_subject_input")
            else:
                year_input = subject_input = ""

        exts = get_supported_extensions()
        uploaded = st.file_uploader(
            f"Supported: {', '.join(exts)}",
            type=[e.lstrip(".") for e in exts],
            accept_multiple_files=True,
            key="lib_file_uploader",
        )

        if uploaded:
            if st.button("🚀 Process & Index", type="primary", key="lib_btn_process"):
                prog = st.progress(0, text="Starting…")
                success, errors = 0, 0

                for idx, uf in enumerate(uploaded):
                    fname = uf.name
                    prog.progress(int(idx / len(uploaded) * 85), text=f"Processing {fname}…")

                    try:
                        # Duplicate check
                        if document_exists(fname, collection_choice):
                            st.info(f"ℹ️ **{fname}** already indexed — skipping (delete first in Browse to re-upload).")
                            continue

                        content = uf.read()
                        saved = save_uploaded_file(content, fname)

                        year = year_input or None
                        subject = subject_input or None
                        if collection_choice == "past_papers" and not year:
                            meta = detect_paper_metadata(fname)
                            year = year or meta.get("year")
                            subject = subject or meta.get("subject")

                        report = process_document(
                            file_path=saved,
                            original_filename=fname,
                            collection=collection_choice,
                            year=year,
                            subject=subject,
                        )

                        # Show extraction warnings
                        for w in report.get("extraction_warnings", []):
                            st.warning(w)

                        total_pages = report.get("total_pages", 0)
                        total_chunks = report.get("total_chunks", 0)
                        chars_ext = report.get("chars_extracted", 0)
                        pages_ocr = report.get("pages_ocrd", 0)

                        if chars_ext < 100:
                            st.error(
                                f"⚠️ **Low text warning for {fname}**: Only {chars_ext} characters extracted. "
                                "This document may be an image scan without OCR."
                            )

                        if report.get("status") in ("processed",) and report.get("chunks_list"):
                            add_chunks(report["chunks_list"], collection_choice)
                            rebuild_from_store(collection_choice)
                            success += 1
                            st.markdown(
                                f"""<div class="glass-card" style="padding:.7rem 1rem;margin:.3rem 0;
                                border-color:rgba(16,185,129,.3)">
                                <span style="color:#10B981;font-weight:600">✓ {fname}</span>
                                <span style="color:#A09CB5;font-size:.83rem;margin-left:10px">
                                📊 Report: <strong>{total_pages}</strong> pages/slides &middot;
                                <strong>{chars_ext:,}</strong> chars &middot;
                                <strong>{total_chunks}</strong> chunks &middot;
                                <strong>{pages_ocr}</strong> OCR pages</span></div>""",
                                unsafe_allow_html=True,
                            )
                        else:
                            errors += 1
                            st.markdown(
                                error_card(f"Empty or Unreadable: {fname}", report.get("message", "")),
                                unsafe_allow_html=True,
                            )

                    except Exception as e:
                        errors += 1
                        logger.error(f"Upload failed {fname}: {e}", exc_info=True)
                        st.markdown(error_card(f"Failed: {fname}", str(e)), unsafe_allow_html=True)

                prog.progress(100, text="Done!")
                if success:
                    st.success(f"✅ {success} document(s) successfully indexed.")
                if errors:
                    st.error(f"❌ {errors} document(s) could not be processed.")
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # BROWSE TAB
    # ════════════════════════════════════════════════════════════════════════
    with browse_tab:
        col_cm, col_pp = st.columns(2)

        for col_widget, coll_name, coll_label in [
            (col_cm, "course_material", "📚 Course Material"),
            (col_pp, "past_papers", "📝 Past Papers"),
        ]:
            with col_widget:
                st.markdown(f"#### {coll_label}")
                try:
                    chunks = get_all_chunks(coll_name)
                    docs: dict[str, list] = {}
                    for c in chunks:
                        dn = c.get("metadata", {}).get("doc_name", "Unknown")
                        docs.setdefault(dn, []).append(c)

                    if not docs:
                        st.markdown(
                            empty_state("📭", "Empty", "Upload documents in the Upload tab."),
                            unsafe_allow_html=True,
                        )
                    else:
                        for doc_name, doc_chunks in sorted(docs.items()):
                            ft = doc_name.rsplit(".", 1)[-1] if "." in doc_name else "doc"
                            meta0 = doc_chunks[0].get("metadata", {})
                            yr = meta0.get("year", "")
                            label = doc_name + (f" [{yr}]" if yr else "")
                            st.markdown(doc_tile(label, ft, len(doc_chunks)), unsafe_allow_html=True)

                            # Clean unique expander key
                            exp_key = f"exp_{coll_name}_{doc_name}"
                            with st.expander(f"⚙️ Details: {doc_name}", expanded=False):
                                if coll_name == "past_papers":
                                    mc1, mc2 = st.columns(2)
                                    mc1.metric("Year", yr or "—")
                                    mc2.metric("Subject", meta0.get("subject", "—"))
                                st.caption(f"{len(doc_chunks)} chunks · method: {meta0.get('extraction_method','?')}")
                                del_key = f"del_{coll_name}_{doc_name}"
                                if st.button("🗑️ Delete Document", key=del_key):
                                    try:
                                        delete_document(doc_name, coll_name)
                                        rebuild_from_store(coll_name)
                                        st.success(f"Deleted '{doc_name}'.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete failed: {e}")

                except Exception as e:
                    st.markdown(error_card("Error loading library", str(e)), unsafe_allow_html=True)


safe_page(_run_library_page)
