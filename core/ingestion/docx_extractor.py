"""
DOCX Extractor
===============
Paragraphs with heading structure and tables.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_docx(file_path: str) -> list[dict]:
    """
    Returns list of pseudo-page dicts (one entry per heading section,
    or one entry per ~N paragraphs if no headings found).
    """
    import docx

    path = Path(file_path)
    try:
        doc = docx.Document(str(path))
    except Exception as e:
        logger.error(f"Cannot open DOCX {path.name}: {e}")
        return []

    sections: list[dict] = []
    current_heading = ""
    current_parts: list[str] = []
    section_idx = 1

    def flush():
        nonlocal section_idx
        text = "\n".join(current_parts).strip()
        if text:
            sections.append({
                "page_or_slide": section_idx,
                "text": text,
                "extraction_method": "text",
                "heading": current_heading,
            })
            section_idx += 1

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            # Paragraph
            para = docx.text.paragraph.Paragraph(element, doc)
            style = (para.style.name or "").lower()
            text = para.text.strip()

            if "heading" in style and text:
                # New section
                flush()
                current_heading = text
                current_parts = []
            elif text:
                current_parts.append(text)

        elif tag == "tbl":
            from docx.table import Table
            try:
                table = Table(element, doc)
                rows = []
                for row in table.rows:
                    cell_texts = [cell.text.strip() for cell in row.cells]
                    rows.append(" | ".join(cell_texts))
                current_parts.extend(rows)
            except Exception:
                pass

    flush()

    # If no structure found, return as single chunk
    if not sections:
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if full_text.strip():
            sections.append({
                "page_or_slide": 1,
                "text": full_text.strip(),
                "extraction_method": "text",
                "heading": "",
            })

    return sections
