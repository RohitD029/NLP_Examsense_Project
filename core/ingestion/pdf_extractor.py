"""
PDF Extractor
=============
Per-page text with PyMuPDF. Falls back to OCR (pytesseract via PyMuPDF pixmap)
for pages with fewer than min_page_chars_for_text characters.
No Poppler, no pdf2image.
Also removes repeated headers/footers.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Returns list of page dicts:
      {page_or_slide, text, extraction_method, heading}
    """
    import fitz  # PyMuPDF

    from core.config import settings
    from core.ocr_helper import tesseract_available, ocr_image

    path = Path(file_path)
    pages = []

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.error(f"Cannot open PDF {path.name}: {e}")
        return []

    # First pass — collect raw text to detect headers/footers
    raw_texts: list[str] = []
    for page in doc:
        raw_texts.append(page.get_text("text") or "")

    header_footer = _detect_repeated_lines(raw_texts)

    for page_num, page in enumerate(doc, 1):
        raw_text = raw_texts[page_num - 1]
        # Remove repeated headers/footers
        clean_text = _strip_lines(raw_text, header_footer)
        method = "text"

        if len(clean_text.strip()) < settings.min_page_chars_for_text:
            # Try OCR
            if tesseract_available():
                try:
                    mat = fitz.Matrix(settings.ocr_dpi / 72, settings.ocr_dpi / 72)
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = ocr_image(img)
                    if ocr_text.strip():
                        clean_text = ocr_text
                        method = "ocr"
                except Exception as e:
                    logger.warning(f"OCR failed on page {page_num}: {e}")

        pages.append({
            "page_or_slide": page_num,
            "text": clean_text.strip(),
            "extraction_method": method,
            "heading": "",
        })

    doc.close()
    return pages


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
