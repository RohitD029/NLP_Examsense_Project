"""
Tesseract detection and OCR helper.
Never crashes — returns empty string with a logged warning if unavailable.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    """Return True if a usable Tesseract binary is found."""
    from core.config import settings  # late import avoids circular

    cmd = settings.tesseract_cmd
    # Prefer env-var / config path
    if cmd and Path(cmd).is_file():
        _set_cmd(cmd)
        return True
    # Fall back to Windows default
    if Path(_WINDOWS_DEFAULT).is_file():
        _set_cmd(_WINDOWS_DEFAULT)
        return True
    # Fall back to PATH
    if shutil.which("tesseract"):
        _set_cmd("tesseract")
        return True

    logger.warning(
        "Tesseract not found. OCR is disabled. "
        "Install from: https://github.com/UB-Mannheim/tesseract/wiki"
    )
    return False


def _set_cmd(path: str) -> None:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = path
        logger.info(f"Tesseract found at: {path}")
    except ImportError:
        pass


def ocr_image(pil_image) -> str:
    """Run OCR on a PIL image. Returns '' if Tesseract unavailable."""
    if not tesseract_available():
        return ""
    try:
        import pytesseract
        return pytesseract.image_to_string(pil_image, lang="eng").strip()
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


def tesseract_install_instructions() -> str:
    return (
        "**Tesseract not installed.** To enable OCR for scanned PDFs and image slides:\n\n"
        "1. Download the Windows installer from "
        "[UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)\n"
        "2. Run the installer (e.g. `tesseract-ocr-w64-setup-5.x.x.exe`)\n"
        "3. Default install path: `C:\\Program Files\\Tesseract-OCR\\`\n"
        "4. Add that folder to your system PATH **or** set env var "
        "`EXAMSENSE_TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe`\n"
        "5. Restart this app."
    )
