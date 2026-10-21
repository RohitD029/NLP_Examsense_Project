"""
ExamSense Configuration
=======================
Central settings for ExamSense.
Single source of truth for models, token limits, and session quotas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    _ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass


@dataclass
class Settings:
    # ── Embedding Model (Single Source of Truth) ──────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_fallback: str = "all-MiniLM-L6-v2"
    bge_query_prefix: str = "query: "

    # ── LLM (Groq via openai client) ──────────────────────────────────────────
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_primary_model: str = "openai/gpt-oss-20b"
    groq_fallback_model: str = "openai/gpt-oss-120b"
    llm_max_output_tokens: int = 350
    llm_temperature: float = 0.2
    llm_context_token_limit: int = 1200     # Top 3 chunks <= ~1200 tokens context
    max_chunk_display_tokens: int = 350

    # ── Chunking & Text Rules ─────────────────────────────────────────────────
    chunk_target_tokens: int = 200
    chunk_overlap_tokens: int = 40
    min_pdf_page_chars: int = 30            # < 30 chars = image-only page
    min_doc_text_chars: int = 300           # < 300 chars = rejected scanned doc
    max_image_only_ratio: float = 0.70      # > 70% image-only pages = rejected

    # ── Session Limits ────────────────────────────────────────────────────────
    max_files_per_session: int = 5
    max_file_size_bytes: int = 20 * 1024 * 1024  # 20 MB
    max_chunks_per_session: int = 3000
    shared_key_max_questions: int = 30

    # ── Retrieval Parameters ──────────────────────────────────────────────────
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_final: int = 3
    rrf_k: int = 60
    confidence_threshold: float = 0.30

    @property
    def embedding_model_name(self) -> str:
        return self.embedding_model


# Global singleton instance
settings = Settings()
