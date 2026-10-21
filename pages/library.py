"""
Library Page — Upload and manage session study documents.
Files are kept in-memory for the current session only.
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

from ui.components import doc_tile, empty_state, error_card
from ui.safe_page import safe_page
from core.config import settings
from core.ingestion.document_processor import (
    process_uploaded_file,
    get_supported_extensions,
)
from core.retrieval.vector_store import (
    add_document,
    delete_document,
    document_exists,
    get_documents,
    get_total_chunks,
)


def _handle_upload_processing(uploaded_files: list) -> None:
    """Process uploaded files from memory into session storage."""
    current_docs = get_documents()
    current_chunks = get_total_chunks()

    if len(current_docs) >= settings.max_files_per_session:
        st.warning(f"Session file limit reached (max {settings.max_files_per_session} files). Delete a document to upload another.")
        return

    progress_bar = st.progress(0, text="Indexing files…")
    total_files = len(uploaded_files)

    for idx, uf in enumerate(uploaded_files):
        fname = uf.name
        progress_bar.progress(int((idx / total_files) * 90), text=f"Processing {fname}…")

        # 1. Duplicate check
        if document_exists(fname):
            st.info(f"ℹ️ **{fname}** is already uploaded in this session.")
            continue

        # 2. Check session limits before processing
        if len(get_documents()) >= settings.max_files_per_session:
            st.warning(f"Session limit reached (max {settings.max_files_per_session} files). Skipping remaining files.")
            break

        # 3. Read bytes directly from memory (no disk saving)
        file_bytes = uf.read()

        # 4. Process document
        report = process_uploaded_file(file_bytes=file_bytes, filename=fname)

        if not report["success"]:
            # Rejected (scanned, DRM, corrupt, unsupported, etc.)
            st.markdown(
                error_card(f"Could not index {fname}", report["error"] or "File rejected."),
                unsafe_allow_html=True,
            )
            continue

        # 5. Check if adding these chunks exceeds 3000 chunks limit
        chunks_to_add = report["chunks"]
        if current_chunks + len(chunks_to_add) > settings.max_chunks_per_session:
            st.warning(
                f"Session chunk limit reached ({settings.max_chunks_per_session:,} chunks). "
                f"Could not add {fname} ({len(chunks_to_add)} chunks). Stop indexing."
            )
            break

        # 6. Index into in-memory FAISS and BM25
        ok, msg = add_document(
            doc_name=fname,
            chunks=chunks_to_add,
            status_note=report["status_note"],
            total_pages=report["total_pages"],
        )

        if ok:
            current_chunks += len(chunks_to_add)
            note = report["status_note"]
            if note != "Ready":
                st.info(f"✓ **{fname}** ready ({note})")
            else:
                st.success(f"✓ **{fname}** ready ({len(chunks_to_add)} chunks)")
        else:
            st.error(f"Failed to index {fname}: {msg}")

    progress_bar.progress(100, text="Done.")


def _run_library_page():
    # ── Header ──
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <h2 style="background:linear-gradient(135deg,#7C3AED,#2DD4BF);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         font-weight:800;font-size:1.8rem;margin:0">📚 Study Library</h2>
      <p style="color:#A09CB5;margin-top:0.3rem">
        Upload your text-based study materials. Files are processed in memory.
      </p>
    </div>""", unsafe_allow_html=True)

    st.caption("Your files are private to this session and are removed when you close the app.")
    st.markdown("---")

    docs = get_documents()

    # ── File Uploader ──
    exts = get_supported_extensions()
    clean_exts = [e.lstrip(".") for e in exts]

    uploaded_files = st.file_uploader(
        "Upload text-based study files (PDF, PPTX, DOCX, EPUB, TXT, MD)",
        type=clean_exts,
        accept_multiple_files=True,
        key="uploader_library_files",
        help="Max 5 files per session, max 20 MB each. Scanned or image-only files without selectable text are not supported.",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Files", type="primary", key="btn_process_library_files"):
            _handle_upload_processing(uploaded_files)
            st.rerun()

    st.markdown("---")
    st.markdown("### Uploaded Documents")

    if not docs:
        st.markdown(
            empty_state("📂", "Your Library is Empty", "Upload lecture slides, notes, textbook chapters, or summaries to begin asking questions."),
            unsafe_allow_html=True,
        )
        return

    # ── Document List with Status and Delete Button ──
    for d in docs:
        doc_name = d["doc_name"]
        status_note = d.get("status_note", "Ready")
        file_ext = Path(doc_name).suffix.lstrip(".")

        c_tile, c_btn = st.columns([5, 1])
        with c_tile:
            st.markdown(doc_tile(doc_name, file_ext, status=status_note), unsafe_allow_html=True)
        with c_btn:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Delete", key=f"del_doc_{doc_name}", use_container_width=True):
                delete_document(doc_name)
                st.rerun()


# Wrap page render in safe_page
safe_page(_run_library_page)
