"""
ExamSense Configuration
=======================
Plain dataclass + python-dotenv. No pydantic-settings dependency.
All tuneable values live here — one source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


@dataclass
class Settings:
    # ── API Keys ──────────────────────────────────────────────────────────────
    groq_api_key: str = field(default_factory=lambda: os.getenv("EXAMSENSE_GROQ_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("EXAMSENSE_GEMINI_API_KEY", ""))

    # ── LLM ───────────────────────────────────────────────────────────────────
    groq_primary_model: str = os.getenv("EXAMSENSE_GROQ_MODEL", "openai/gpt-oss-120b")
    groq_fallback_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_max_tokens: int = 600
    llm_temperature: float = 0.2
    llm_context_token_limit: int = 2500   # hard cap on context sent to LLM

    # ── Rate Limiter ───────────────────────────────────────────────────────────
    rate_limit_rpm: int = 28              # requests per 60 s (stay under 30)
    rate_limit_tpm: int = 7500           # tokens per 60 s  (stay under 8000)

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_fallback: str = "all-MiniLM-L6-v2"
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_target_tokens: int = 200
    chunk_overlap_tokens: int = 40
    min_page_chars_for_text: int = 50    # below this → try OCR
    ocr_dpi: int = 200

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_final: int = 4
    rrf_k: int = 60
    confidence_threshold: float = 0.30   # below this → abstain
    abstain_require_both_weak: bool = True  # abstain only if BOTH dense+BM25 weak

    # ── Exam Frequency ─────────────────────────────────────────────────────────
    cluster_distance_threshold: float = 0.35

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")

    # ── Tesseract ─────────────────────────────────────────────────────────────
    tesseract_cmd: str = field(default_factory=lambda: (
        os.getenv("TESSERACT_CMD", "")
        or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ))

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def eval_set_path(self) -> Path:
        return self.data_dir / "eval_set.json"

    @property
    def clusters_path(self) -> Path:
        return self.index_dir / "exam_clusters.json"

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_model

    @property
    def max_chunk_chars(self) -> int:
        return self.chunk_target_tokens * 4


# Singleton
settings = Settings()
