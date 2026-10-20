"""
Embedder
=========
Loads sentence-transformer model once (cached). Supports BGE query prefix.
Falls back to all-MiniLM-L6-v2 if primary model fails to load.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Load embedding model once, with fallback."""
    from core.config import settings
    from sentence_transformers import SentenceTransformer

    for model_name in [settings.embedding_model, settings.embedding_fallback]:
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded: {model_name}")
            return model, model_name
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")

    raise RuntimeError("No embedding model could be loaded.")


def get_model_name() -> str:
    _, name = _load_model()
    return name


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts (document passages).
    Returns L2-normalised float32 array of shape (N, D).
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    model, _ = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2 normalise → cosine via dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string with the BGE query prefix if applicable.
    Returns L2-normalised float32 array of shape (D,).
    """
    from core.config import settings

    model, model_name = _load_model()

    # BGE models benefit from a query prefix
    if "bge" in model_name.lower():
        query = settings.bge_query_prefix + query

    embedding = model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.astype(np.float32)
