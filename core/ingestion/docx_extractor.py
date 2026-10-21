"""
DOCX Extractor
==============
Paragraphs with heading structure and tables.
Cleaned with clean_text.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_docx(file_path: str) -> dict:
    """
    Extracts structured sections from a DOCX document (paragraphs + tables).

    Returns dict:
      pages: list of section dicts [{"page_or_slide": int, "text": str, "is_image_only": bool, "heading": str}]
      total_chars: int
      image_only_pages: int
      error: str | None
    """
    import docx
    from core.nlp.preprocess import clean_text

    path = Path(file_path)
    try:
        doc = docx.Document(str(path))
    except Exception as e:
        logger.warning(f"Cannot open DOCX {path.name}: {e}")
        return {
            "pages": [],
            "total_chars": 0,
            "image_only_pages": 0,
            "error": "This document appears to be corrupted or invalid.",
        }

    try:
        sections: list[dict] = []
        current_heading = ""
        current_parts: list[str] = []
        section_idx = 1
        total_chars = 0

        def flush():
            nonlocal section_idx, total_chars
            raw_section = "\n".join(current_parts).strip()
            cleaned = clean_text(raw_section)
            if cleaned:
                c_len = len(cleaned)
                total_chars += c_len
                sections.append({
                    "page_or_slide": section_idx,
                    "text": cleaned,
                    "is_image_only": False,
                    "heading": current_heading,
                })
                section_idx += 1

        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                para = docx.text.paragraph.Paragraph(element, doc)
                style = (para.style.name or "").lower()
                text = para.text.strip()

                if "heading" in style and text:
                    flush()
                    current_heading = text
                    current_parts = []
                elif text:
                    current_parts.append(text)

            elif tag == "tbl":
                from docx.table import Table
                try:
                    table = Table(element, doc)
                    for row in table.rows:
                        cell_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if cell_texts:
                            current_parts.append(" | ".join(cell_texts))
                except Exception:
                    pass

        flush()

        if not sections:
            full_text = clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
            if full_text:
                total_chars = len(full_text)
                sections.append({
                    "page_or_slide": 1,
                    "text": full_text,
                    "is_image_only": False,
                    "heading": "",
                })

        return {
            "pages": sections,
            "total_chars": total_chars,
            "image_only_pages": 0,
            "error": None if sections else "This Word document is empty.",
        }
    except Exception as e:
        logger.error(f"Error reading DOCX {path.name}: {e}")
        return {
            "pages": [],
            "total_chars": 0,
            "image_only_pages": 0,
            "error": f"Failed to extract text from Word document: {str(e)}",
        }
