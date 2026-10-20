"""
EPUB Extractor
===============
Uses ebooklib + BeautifulSoup. Chapter-by-chapter extraction.
DRM-protected files show a friendly error.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_epub(file_path: str) -> list[dict]:
    """
    Returns list of chapter dicts:
      {page_or_slide, text, extraction_method, heading}
    """
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError as e:
        logger.error(f"ebooklib/beautifulsoup4 not installed: {e}")
        return []

    path = Path(file_path)
    try:
        book = epub.read_epub(str(path))
    except Exception as e:
        err = str(e).lower()
        if "drm" in err or "encrypted" in err or "rights" in err:
            logger.error(f"DRM-protected EPUB: {path.name}")
            return [{"page_or_slide": 0, "text": "__DRM_PROTECTED__",
                     "extraction_method": "error", "heading": ""}]
        logger.error(f"Cannot open EPUB {path.name}: {e}")
        return []

    chapters = []
    chap_idx = 1

    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        try:
            html = item.get_content().decode("utf-8", errors="replace")
        except Exception:
            continue

        soup = BeautifulSoup(html, "html.parser")

        # Extract chapter title
        title = ""
        for tag in ["h1", "h2", "h3", "title"]:
            el = soup.find(tag)
            if el and el.get_text(strip=True):
                title = el.get_text(strip=True)
                break

        # Remove scripts/styles
        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = _clean_text(text)

        if len(text.strip()) < 30:
            continue

        chapters.append({
            "page_or_slide": chap_idx,
            "text": text.strip(),
            "extraction_method": "text",
            "heading": title,
        })
        chap_idx += 1

    return chapters


def _clean_text(text: str) -> str:
    import re
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
