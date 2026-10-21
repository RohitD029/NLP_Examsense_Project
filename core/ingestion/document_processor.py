"""
Document Processor
==================
Dispatches file formats, enforces validation rules, rejects scanned/image-based files,
and prepares chunks for in-memory session indexing.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from core.config import settings
from core.ingestion.chunker import chunk_pages
from core.nlp.preprocess import clean_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".epub", ".txt", ".md"}
UNSUPPORTED_TYPE_MESSAGE = (
    "Unsupported file type. Please upload PDF, PPTX, DOCX, EPUB, TXT or MD."
)
SCANNED_FILE_MESSAGE = (
    "This file looks scanned or image-based, so I can't read it. "
    "This app only supports text-based files. "
    "Please upload a version where you can select the text."
)


def get_supported_extensions() -> list[str]:
    return sorted(list(SUPPORTED_EXTENSIONS))


def is_supported_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in SUPPORTED_EXTENSIONS


def process_uploaded_file(file_bytes: bytes, filename: str) -> dict:
    """
    Process an uploaded document from memory bytes.

    Returns dict:
      success: bool
      doc_name: str
      chunks: list of dicts (chunk_id, text, metadata)
      total_pages: int
      skipped_pages: int
      status_note: str  ("Ready" or "N pages/slides had no readable text and were skipped.")
      error: str | None
    """
    path = Path(filename)
    suffix = path.suffix.lower()

    # 1. Extension check
    if suffix not in SUPPORTED_EXTENSIONS:
        return {
            "success": False,
            "doc_name": filename,
            "chunks": [],
            "total_pages": 0,
            "skipped_pages": 0,
            "status_note": "Unsupported",
            "error": UNSUPPORTED_TYPE_MESSAGE,
        }

    # 2. File size check (max 20 MB)
    if len(file_bytes) > settings.max_file_size_bytes:
        return {
            "success": False,
            "doc_name": filename,
            "chunks": [],
            "total_pages": 0,
            "skipped_pages": 0,
            "status_note": "File too large",
            "error": f"File exceeds the 20 MB limit ({len(file_bytes) / (1024*1024):.1f} MB).",
        }

    # 3. Extract text
    tmp_path = None
    try:
        # Write to a temporary file for libraries that require a file path
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        extraction = _extract_file(tmp_path, suffix)
    except Exception as e:
        logger.warning(f"Extraction error for {filename}: {e}")
        return {
            "success": False,
            "doc_name": filename,
            "chunks": [],
            "total_pages": 0,
            "skipped_pages": 0,
            "status_note": "Error",
            "error": f"Could not read file: {str(e)}",
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # 4. Check for extraction-level errors (DRM, corrupt, password, empty)
    if extraction.get("error"):
        return {
            "success": False,
            "doc_name": filename,
            "chunks": [],
            "total_pages": 0,
            "skipped_pages": 0,
            "status_note": "Rejected",
            "error": extraction["error"],
        }

    pages = extraction.get("pages", [])
    total_chars = extraction.get("total_chars", 0)
    image_only_pages = extraction.get("image_only_pages", 0)
    total_pages = len(pages)

    if total_pages == 0:
        return {
            "success": False,
            "doc_name": filename,
            "chunks": [],
            "total_pages": 0,
            "skipped_pages": 0,
            "status_note": "Empty file",
            "error": "This file contains no readable content.",
        }

    # 5. Scanned / image-based detection rules:
    # - If the whole document has fewer than 300 characters of text,
    # - OR more than 70% of its pages/slides are image-only:
    #   REJECT it with SCANNED_FILE_MESSAGE. Do not index it.
    image_only_ratio = image_only_pages / total_pages if total_pages > 0 else 0.0
    if total_chars < settings.min_doc_text_chars or image_only_ratio > settings.max_image_only_ratio:
        return {
            "success": False,
            "doc_name": filename,
            "chunks": [],
            "total_pages": total_pages,
            "skipped_pages": image_only_pages,
            "status_note": "Scanned/Image-based",
            "error": SCANNED_FILE_MESSAGE,
        }

    # 6. Chunking
    chunks_obj = chunk_pages(
        pages=pages,
        doc_name=filename,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    chunks = [c.to_dict() for c in chunks_obj]

    if not chunks:
        return {
            "success": False,
            "doc_name": filename,
            "chunks": [],
            "total_pages": total_pages,
            "skipped_pages": image_only_pages,
            "status_note": "No text",
            "error": SCANNED_FILE_MESSAGE,
        }

    # 7. Note on skipped pages
    if image_only_pages > 0:
        status_note = f"{image_only_pages} pages/slides had no readable text and were skipped."
    else:
        status_note = "Ready"

    return {
        "success": True,
        "doc_name": filename,
        "chunks": chunks,
        "total_pages": total_pages,
        "skipped_pages": image_only_pages,
        "status_note": status_note,
        "error": None,
    }


def _extract_file(file_path: str, suffix: str) -> dict:
    """Dispatch file extraction by extension."""
    if suffix == ".pdf":
        from core.ingestion.pdf_extractor import extract_text_from_pdf
        return extract_text_from_pdf(file_path)

    elif suffix in (".pptx", ".ppt"):
        from core.ingestion.pptx_extractor import extract_text_from_pptx
        return extract_text_from_pptx(file_path)

    elif suffix in (".docx", ".doc"):
        from core.ingestion.docx_extractor import extract_text_from_docx
        return extract_text_from_docx(file_path)

    elif suffix == ".epub":
        from core.ingestion.epub_extractor import extract_text_from_epub
        return extract_text_from_epub(file_path)

    elif suffix in (".txt", ".md"):
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            cleaned = clean_text(content)
            total_chars = len(cleaned.strip())
            return {
                "pages": [{
                    "page_or_slide": 1,
                    "text": cleaned,
                    "is_image_only": False,
                    "heading": "",
                }],
                "total_chars": total_chars,
                "image_only_pages": 0,
                "error": None if total_chars > 0 else "File is empty.",
            }
        except Exception as e:
            return {
                "pages": [],
                "total_chars": 0,
                "image_only_pages": 0,
                "error": f"Failed to read text file: {str(e)}",
            }

    return {
        "pages": [],
        "total_chars": 0,
        "image_only_pages": 0,
        "error": UNSUPPORTED_TYPE_MESSAGE,
    }
