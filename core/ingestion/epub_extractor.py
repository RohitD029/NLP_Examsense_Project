"""
EPUB Extractor
==============
Uses ebooklib + BeautifulSoup. Chapter-by-chapter extraction, keeping chapter title.
Detects DRM, corruption, and empty files with clear friendly reasons.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_epub(file_path: str) -> dict:
    """
    Extracts text per chapter from an EPUB document.

    Returns dict:
      pages: list of chapter dicts [{"page_or_slide": int, "text": str, "is_image_only": bool, "heading": str}]
      total_chars: int
      image_only_pages: int
      error: str | None
    """
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError as e:
        logger.error(f"ebooklib or beautifulsoup4 not installed: {e}")
        return {
            "pages": [],
            "total_chars": 0,
            "image_only_pages": 0,
            "error": "EPUB reader libraries (ebooklib / beautifulsoup4) not available.",
        }

    from core.nlp.preprocess import clean_text

    path = Path(file_path)
    try:
        book = epub.read_epub(str(path))
    except Exception as e:
        err_str = str(e).lower()
        if any(w in err_str for w in ("drm", "encrypted", "rights", "password")):
            return {
                "pages": [],
                "total_chars": 0,
                "image_only_pages": 0,
                "error": "This EPUB is DRM-protected and cannot be read. Please upload a DRM-free version.",
            }
        logger.warning(f"Cannot open EPUB {path.name}: {e}")
        return {
            "pages": [],
            "total_chars": 0,
            "image_only_pages": 0,
            "error": "This EPUB appears to be corrupted or invalid.",
        }

    try:
        chapters = []
        chap_idx = 1
        total_chars = 0

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

            # Remove scripts and styles
            for tag in soup(["script", "style", "head"]):
                tag.decompose()

            raw_text = soup.get_text(separator="\n", strip=True)
            cleaned = clean_text(raw_text)

            if len(cleaned.strip()) < 30:
                continue

            c_len = len(cleaned)
            total_chars += c_len
            chapters.append({
                "page_or_slide": chap_idx,
                "text": cleaned,
                "is_image_only": False,
                "heading": title,
            })
            chap_idx += 1

        if not chapters:
            return {
                "pages": [],
                "total_chars": 0,
                "image_only_pages": 0,
                "error": "This EPUB contains no readable chapter text.",
            }

        return {
            "pages": chapters,
            "total_chars": total_chars,
            "image_only_pages": 0,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error reading EPUB {path.name}: {e}")
        return {
            "pages": [],
            "total_chars": 0,
            "image_only_pages": 0,
            "error": f"Failed to extract text from EPUB: {str(e)}",
        }
