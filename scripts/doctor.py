"""
scripts/doctor.py
==================
Checks critical dependencies and system requirements for ExamSense.
Run: python scripts/doctor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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
        ("fastembed or sentence_transformers", "import fastembed"),
        ("faiss", "import faiss"),
        ("rank_bm25", "from rank_bm25 import BM25Plus"),
        ("numpy", "import numpy"),
        ("pymupdf (fitz)", "import fitz"),
        ("python-pptx", "from pptx import Presentation"),
        ("python-docx", "import docx"),
        ("ebooklib", "import ebooklib"),
        ("beautifulsoup4", "from bs4 import BeautifulSoup"),
        ("openai", "from openai import OpenAI"),
        ("python-dotenv", "from dotenv import load_dotenv"),
        ("pytest", "import pytest"),
        ("ruff", "import ruff"),
    ]

    for pkg_name, stmt in imports:
        def _make_check(s):
            return lambda: exec(s) or "imported"
        results.append(check(f"import {pkg_name}", _make_check(stmt)))

    # Embedding model loadable
    def _check_embedding():
        from core.retrieval.embedder import get_cached_model
        backend, _ = get_cached_model()
        return f"backend: {backend}"
    results.append(check("Embedding model", _check_embedding))

    # Preprocessing
    def _check_nlp():
        from core.nlp.preprocess import clean_text, bm25_tokenize
        t = clean_text("Hello--\nworld! Page 1")
        toks = bm25_tokenize(t)
        return f"tokens: {toks}"
    results.append(check("NLP preprocess", _check_nlp))

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


if __name__ == "__main__":
    results = run_all()
    print_results(results)
    sys.exit(1 if any(r["status"] == "FAIL" for r in results) else 0)
