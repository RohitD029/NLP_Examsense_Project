"""
Document Processor
==================
Orchestrates ingestion: detect format → extract → chunk → return report.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EXTRACTORS = {
    ".pdf": "core.ingestion.pdf_extractor.extract_text_from_pdf",
    ".pptx": "core.ingestion.pptx_extractor.extract_text_from_pptx",
    ".ppt": "core.ingestion.pptx_extractor.extract_text_from_pptx",
    ".docx": "core.ingestion.docx_extractor.extract_text_from_docx",
    ".doc": "core.ingestion.docx_extractor.extract_text_from_docx",
    ".epub": "core.ingestion.epub_extractor.extract_text_from_epub",
    ".txt": "_txt",
    ".md": "_txt",
    ".png": "_img",
    ".jpg": "_img",
    ".jpeg": "_img",
}


def get_supported_extensions() -> list[str]:
    return list(EXTRACTORS.keys())


def process_document(
    file_path: str,
    original_filename: str,
    collection: str = "course_material",
    year: Optional[str] = None,
    subject: Optional[str] = None,
) -> dict:
    """
    Process a single document.

    Returns report dict:
      doc_id, doc_name, collection, total_pages, total_chunks,
      chunks_list (Chunk.to_dict()), pages_ocr'd, chars_extracted,
      status, message, extraction_warnings
    """
    from core.ingestion.chunker import chunk_pages

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix not in EXTRACTORS:
        return _error_report(original_filename, collection, f"Unsupported format: {suffix}")

    # Extract
    pages = _extract(str(path), suffix)

    # Handle DRM
    if pages and pages[0].get("text") == "__DRM_PROTECTED__":
        return _error_report(
            original_filename, collection,
            "This EPUB is DRM-protected and cannot be read. "
            "Please use a DRM-free copy."
        )

    if not pages:
        return _empty_report(original_filename, collection)

    total_chars = sum(len(p.get("text", "")) for p in pages)
    pages_ocrd = sum(1 for p in pages if p.get("extraction_method") == "ocr")

    # Check near-empty
    warnings = []
    if total_chars < 100:
        warnings.append(
            "⚠️ **No readable text found** — this document may be scanned or image-only. "
            "Install Tesseract OCR to extract text from scanned documents."
        )

    doc_id = str(uuid.uuid4())

    # Add year/subject metadata to each page for past_papers
    extra_meta = {}
    if year:
        extra_meta["year"] = year
    if subject:
        extra_meta["subject"] = subject

    chunks = chunk_pages(
        pages=pages,
        doc_name=original_filename,
        collection=collection,
        doc_id=doc_id,
    )

    # Add extra metadata to each chunk
    chunk_dicts = []
    for c in chunks:
        d = c.to_dict()
        d["metadata"].update(extra_meta)
        chunk_dicts.append(d)

    return {
        "doc_id": doc_id,
        "doc_name": original_filename,
        "collection": collection,
        "total_pages": len(pages),
        "total_chunks": len(chunk_dicts),
        "chunks_list": chunk_dicts,
        "pages_ocrd": pages_ocrd,
        "chars_extracted": total_chars,
        "status": "processed",
        "message": f"Extracted {total_chars:,} chars from {len(pages)} pages, created {len(chunk_dicts)} chunks.",
        "extraction_warnings": warnings,
        "year": year,
        "subject": subject,
    }


def save_uploaded_file(file_content: bytes, original_filename: str) -> str:
    """Save file bytes to uploads dir with unique prefix."""
    from core.config import settings
    uploads_dir = settings.uploads_dir
    unique = f"{uuid.uuid4().hex[:8]}_{original_filename}"
    save_path = uploads_dir / unique
    save_path.write_bytes(file_content)
    logger.info(f"Saved: {save_path}")
    return str(save_path)


def detect_paper_metadata(filename: str) -> dict:
    """Auto-detect year and subject from filename."""
    import re
    meta = {}
    m = re.search(r"(20\d{2})", filename)
    if m:
        meta["year"] = m.group(1)
    stem = Path(filename).stem
    if "year" in meta:
        stem = stem.replace(meta["year"], "")
    fillers = {
        "question","paper","exam","endterm","midterm","final",
        "mid","end","term","semester","sem","test","quiz",
    }
    parts = [p for p in stem.replace("_", " ").replace("-", " ").split()
             if p.lower() not in fillers and len(p) > 1]
    if parts:
        meta["subject"] = " ".join(parts)
    return meta


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract(file_path: str, suffix: str) -> list[dict]:
    """Dispatch to the correct extractor."""
    extractor_ref = EXTRACTORS[suffix]

    if extractor_ref == "_txt":
        return _extract_txt(file_path)
    if extractor_ref == "_img":
        return _extract_image(file_path)

    module_path, func_name = extractor_ref.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func(file_path)


def _extract_txt(file_path: str) -> list[dict]:
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return [{"page_or_slide": 1, "text": text, "extraction_method": "text", "heading": ""}]
    except Exception as e:
        logger.error(f"TXT read failed: {e}")
        return []


def _extract_image(file_path: str) -> list[dict]:
    from core.ocr_helper import tesseract_available, ocr_image
    if not tesseract_available():
        return [{"page_or_slide": 1, "text": "", "extraction_method": "ocr_unavailable", "heading": ""}]
    try:
        from PIL import Image
        img = Image.open(file_path)
        text = ocr_image(img)
        return [{"page_or_slide": 1, "text": text, "extraction_method": "ocr", "heading": ""}]
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        return []


def _error_report(doc_name: str, collection: str, message: str) -> dict:
    return {
        "doc_id": None, "doc_name": doc_name, "collection": collection,
        "total_pages": 0, "total_chunks": 0, "chunks_list": [],
        "pages_ocrd": 0, "chars_extracted": 0,
        "status": "error", "message": message, "extraction_warnings": [message],
    }


def _empty_report(doc_name: str, collection: str) -> dict:
    return {
        "doc_id": None, "doc_name": doc_name, "collection": collection,
        "total_pages": 0, "total_chunks": 0, "chunks_list": [],
        "pages_ocrd": 0, "chars_extracted": 0,
        "status": "empty",
        "message": "No text could be extracted from this document.",
        "extraction_warnings": [
            "⚠️ **No readable text found** — likely scanned or image-only. "
            "Install Tesseract OCR to extract text from scanned content."
        ],
    }
