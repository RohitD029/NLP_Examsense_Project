"""
Image Text Extractor (OCR)
==========================
Extracts text from standalone images (JPG, PNG, etc.) using pytesseract.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_image(file_path: str) -> list[dict]:
    """
    Extract text from an image file using Tesseract OCR.

    Args:
        file_path: Path to the image file.

    Returns:
        List of dicts: [{"page": 1, "text": str}]
        (Images are single-page by nature.)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        import pytesseract
        from PIL import Image

        image = Image.open(str(file_path))

        # Convert to RGB if necessary (e.g., RGBA PNGs)
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")

        text = pytesseract.image_to_string(image).strip()

        if text:
            logger.info(f"OCR extracted {len(text)} chars from image: {file_path.name}")
            return [{"page": 1, "text": text}]
        else:
            logger.warning(f"OCR extracted no text from image: {file_path.name}")
            return []

    except ImportError:
        logger.error(
            "pytesseract is not installed. Install with: pip install pytesseract "
            "and ensure the Tesseract OCR engine is installed on your system."
        )
        return []
    except Exception as e:
        logger.error(f"Failed to OCR image {file_path.name}: {e}")
        return []
