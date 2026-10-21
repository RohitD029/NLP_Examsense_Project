"""
PDF Extractor
=============
Per-page text extraction with PyMuPDF.
Detects password protection, corruption, and image-only/scanned pages.
No OCR, no Pillow, no Tesseract.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extracts text per page from PDF using PyMuPDF.

    Returns dict:
      pages: list of dicts [{"page_or_slide": int, "text": str, "is_image_only": bool, "heading": ""}]
      total_chars: int
      image_only_pages: int
      error: str | None
    """
    import fitz  # PyMuPDF
    from core.config import settings
    from core.nlp.preprocess import clean_text

    path = Path(file_path)
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.warning(f"Could not open PDF {path.name}: {e}")
        return {
            "pages": [],
            "total_chars": 0,
            "image_only_pages": 0,
            "error": "This PDF appears to be corrupted or invalid.",
        }

    try:
        if doc.is_encrypted:
            doc.close()
            return {
                "pages": [],
                "total_chars": 0,
                "image_only_pages": 0,
                "error": "This PDF is password-protected and cannot be read.",
            }

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return {
                "pages": [],
                "total_chars": 0,
                "image_only_pages": 0,
                "error": "This PDF is empty.",
            }

        # First pass — collect raw text to detect repeated headers/footers
        raw_texts: list[str] = []
        for page in doc:
            try:
                raw_texts.append(page.get_text("text") or "")
            except Exception:
                raw_texts.append("")

        header_footer = _detect_repeated_lines(raw_texts)

        pages = []
        image_only_count = 0
        total_chars = 0

        for page_num, raw_text in enumerate(raw_texts, 1):
            stripped_text = _strip_lines(raw_text, header_footer)
            cleaned = clean_text(stripped_text)
            char_count = len(cleaned.strip())

            # A page with fewer than min_pdf_page_chars characters is considered image-only
            is_image_only = char_count < settings.min_pdf_page_chars
            if is_image_only:
                image_only_count += 1
            else:
                total_chars += char_count

            pages.append({
                "page_or_slide": page_num,
                "text": cleaned if not is_image_only else "",
                "is_image_only": is_image_only,
                "heading": "",
            })

        doc.close()

        return {
            "pages": pages,
            "total_chars": total_chars,
            "image_only_pages": image_only_count,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error reading PDF {path.name}: {e}")
        try:
            doc.close()
        except Exception:
            pass
        return {
            "pages": [],
            "total_chars": 0,
            "image_only_pages": 0,
            "error": f"Failed to extract text from PDF: {str(e)}",
        }


def _detect_repeated_lines(raw_texts: list[str], threshold: float = 0.6) -> set[str]:
    """Find lines that appear on > threshold fraction of pages (headers/footers)."""
    if len(raw_texts) < 3:
        return set()

    line_counts: dict[str, int] = collections.Counter()
    for text in raw_texts:
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) < 120:
                line_counts[line] += 1

    n_pages = len(raw_texts)
    return {line for line, count in line_counts.items() if count / n_pages > threshold}


def _strip_lines(text: str, to_remove: set[str]) -> str:
    if not to_remove:
        return text
    lines = []
    for line in text.splitlines():
        if line.strip() not in to_remove:
            lines.append(line)
    return "\n".join(lines)
