"""
scripts/doctor.py
==================
Prints OK / FAIL for every critical dependency and system requirement.
Run: python scripts/doctor.py  (using .venv312)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKS = []


def check(name: str, fn) -> dict:
    try:
        result = fn()
        return {"name": name, "status": "OK", "detail": str(result) if result else ""}
    except Exception as e:
        return {"name": name, "status": "FAIL", "detail": str(e)}


def run_all() -> list[dict]:
    results = []

    # Python version
    results.append(check("Python >= 3.11", lambda: (
        sys.version if sys.version_info >= (3, 11)
        else (_ for _ in ()).throw(RuntimeError(f"Python {sys.version} < 3.11"))
    )))

    # Core imports
    imports = [
        ("streamlit", "import streamlit"),
        ("sentence_transformers", "import sentence_transformers"),
        ("faiss", "import faiss"),
        ("rank_bm25", "from rank_bm25 import BM25Okapi"),
        ("numpy", "import numpy"),
        ("pandas", "import pandas"),
        ("scikit-learn", "from sklearn.cluster import AgglomerativeClustering"),
        ("scipy", "import scipy"),
        ("pymupdf (fitz)", "import fitz"),
        ("python-pptx", "from pptx import Presentation"),
        ("python-docx", "import docx"),
        ("ebooklib", "import ebooklib"),
        ("beautifulsoup4", "from bs4 import BeautifulSoup"),
        ("pillow", "from PIL import Image"),
        ("pytesseract", "import pytesseract"),
        ("plotly", "import plotly"),
        ("openai", "from openai import OpenAI"),
        ("python-dotenv", "from dotenv import load_dotenv"),
        ("pytest", "import pytest"),
    ]

    for pkg_name, stmt in imports:
        def _make_check(s):
            return lambda: exec(s) or "imported"
        results.append(check(f"import {pkg_name}", _make_check(stmt)))

    # Tesseract binary
    def _check_tesseract():
        from core.ocr_helper import tesseract_available
        avail = tesseract_available()
        if avail:
            return "found"
        raise RuntimeError(
            "Not found. Install from https://github.com/UB-Mannheim/tesseract/wiki"
        )
    results.append(check("Tesseract binary", _check_tesseract))

    # Groq API key
    def _check_groq():
        from core.config import settings
        if not settings.groq_api_key:
            raise RuntimeError("EXAMSENSE_GROQ_API_KEY not set in .env or environment")
        return f"set (len={len(settings.groq_api_key)})"
    results.append(check("Groq API key", _check_groq))

    # Embedding model loadable
    def _check_embedding():
        from core.retrieval.embedder import _load_model
        _, name = _load_model()
        return f"loaded: {name}"
    results.append(check("Embedding model", _check_embedding))

    # Index dir writable
    def _check_index_dir():
        from core.config import settings
        d = settings.index_dir
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return str(d)
    results.append(check("Index dir writable", _check_index_dir))

    return results


def print_results(results: list[dict]) -> None:
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = len(results) - ok
    print(f"\n{'='*60}")
    print(f"  ExamSense Doctor -- {ok} OK, {fail} FAIL")
    print(f"{'='*60}")
    for r in results:
        icon = "[OK]  " if r["status"] == "OK" else "[FAIL]"
        detail = f"  ({r['detail']})" if r["detail"] else ""
        print(f"  {icon} {r['name']}{detail}")
    print(f"{'='*60}\n")
    if fail:
        print("[!] Note: Review the FAIL items before running.")
    else:
        print("[+] All critical checks passed -- app is ready to run!")


def get_status_dict() -> list[dict]:
    """For use in the Streamlit sidebar."""
    return run_all()


if __name__ == "__main__":
    results = run_all()
    print_results(results)
    sys.exit(1 if any(r["status"] == "FAIL" for r in results) else 0)
