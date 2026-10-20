"""
PPTX Extractor
===============
Per-slide extraction: title, text frames, tables, speaker notes.
Recurses into grouped shapes. OCRs image-only slides when Tesseract available.
Merges consecutive small slides into chunks (target ~200 tokens).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pptx(file_path: str) -> list[dict]:
    """
    Returns one dict per slide (or merged group of small slides):
      {page_or_slide, text, extraction_method, heading}
    """
    from pptx import Presentation
    from core.config import settings
    from core.ocr_helper import tesseract_available, ocr_image

    path = Path(file_path)
    try:
        prs = Presentation(str(path))
    except Exception as e:
        logger.error(f"Cannot open PPTX {path.name}: {e}")
        return []

    slide_dicts = []

    for slide_num, slide in enumerate(prs.slides, 1):
        parts = []
        title = ""
        has_image = False
        method = "text"

        # Get title
        try:
            if slide.shapes.title and slide.shapes.title.text:
                title = slide.shapes.title.text.strip()
        except Exception:
            pass

        # Recurse shapes
        _collect_shapes(slide.shapes, parts, has_image_ref=[False])
        has_image = _contains_image(slide.shapes)

        # Speaker notes
        try:
            notes_slide = slide.notes_slide
            notes_text = notes_slide.notes_text_frame.text.strip()
            if notes_text:
                parts.append(f"[Notes] {notes_text}")
        except Exception:
            pass

        # Deduplicate paragraphs in slide while maintaining order
        seen_parts = set()
        clean_parts = []
        for p in parts:
            if p not in seen_parts:
                seen_parts.add(p)
                clean_parts.append(p)

        text = "\n".join(clean_parts).strip()

        # OCR image-only slide
        if len(text) < settings.min_page_chars_for_text and has_image and tesseract_available():
            try:
                from PIL import Image
                import io
                for shape in slide.shapes:
                    if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
                        img_bytes = shape.image.blob
                        img = Image.open(io.BytesIO(img_bytes))
                        ocr_text = ocr_image(img)
                        if ocr_text.strip():
                            text = (text + "\n" + ocr_text).strip() if text else ocr_text.strip()
                            method = "ocr"
                            break
            except Exception as e:
                logger.warning(f"Slide {slide_num} OCR failed: {e}")

        slide_dicts.append({
            "page_or_slide": slide_num,
            "text": text,
            "extraction_method": method,
            "heading": title,
        })

    # Preserve each slide individually with its real slide number and heading
    return slide_dicts


def _collect_shapes(shapes, parts: list[str], has_image_ref: list[bool]) -> None:
    """Recursively collect text from all shapes."""
    for shape in shapes:
        # Grouped shapes
        if shape.shape_type == 6:  # MSO_SHAPE_TYPE.GROUP
            try:
                _collect_shapes(shape.shapes, parts, has_image_ref)
            except Exception:
                pass
            continue

        # Text frames
        if shape.has_text_frame:
            for para in shape.text_frame.paragraphs:
                t = para.text.strip()
                if t:
                    parts.append(t)

        # Tables
        if shape.has_table:
            for row in shape.table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells
                )
                if row_text.strip(" |"):
                    parts.append(row_text)

        # Image
        if shape.shape_type == 13:
            has_image_ref[0] = True


def _contains_image(shapes) -> bool:
    for shape in shapes:
        if shape.shape_type == 13:
            return True
        if shape.shape_type == 6:
            try:
                if _contains_image(shape.shapes):
                    return True
            except Exception:
                pass
    return False


def _merge_small_slides(slides: list[dict], target_tokens: int = 200) -> list[dict]:
    """Merge consecutive slides that are individually small."""
    if not slides:
        return []

    def _approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    merged = []
    buffer: list[dict] = []
    buffer_tokens = 0

    for slide in slides:
        st = _approx_tokens(slide["text"])
        if buffer and buffer_tokens + st > target_tokens:
            # Flush buffer
            merged.append(_merge_group(buffer))
            buffer = []
            buffer_tokens = 0
        buffer.append(slide)
        buffer_tokens += st

    if buffer:
        merged.append(_merge_group(buffer))

    return merged


def _merge_group(group: list[dict]) -> dict:
    if len(group) == 1:
        return group[0]
    combined_text = "\n\n".join(s["text"] for s in group if s["text"])
    headings = [s["heading"] for s in group if s["heading"]]
    method = "ocr" if any(s["extraction_method"] == "ocr" for s in group) else "text"
    return {
        "page_or_slide": group[0]["page_or_slide"],
        "text": combined_text,
        "extraction_method": method,
        "heading": " / ".join(headings) if headings else "",
    }
