"""
Embedder
========
Loads embedding model once (cached via @st.cache_resource).
Prefers fastembed (ONNX, memory-safe, no torch), falls back to sentence-transformers.
Applies BGE query prefix for queries only.
Returns float32 L2-normalised embeddings.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Tuple

import numpy as np

from core.config import settings

logger = logging.getLogger(__name__)


def _load_model_uncached() -> Tuple[str, Any]:
    """Load the configured embedding model with fastembed, falling back to sentence-transformers."""
    model_name = settings.embedding_model

    # 1. Try fastembed
    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(model_name=model_name)
        logger.info(f"Loaded fastembed model: {model_name}")
        return ("fastembed", model)
    except Exception as e:
        logger.warning(f"fastembed not available or failed for {model_name} ({e}). Falling back.")

    # 2. Fall back to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info(f"Loaded sentence-transformers model: {model_name}")
        return ("sentence_transformers", model)
    except Exception as e:
        logger.warning(f"sentence-transformers failed for {model_name}: {e}")

    # 3. Fallback model name
    try:
        from sentence_transformers import SentenceTransformer
        fallback = settings.embedding_fallback
        model = SentenceTransformer(fallback)
        logger.info(f"Loaded fallback model: {fallback}")
        return ("sentence_transformers", model)
    except Exception as e:
        raise RuntimeError(f"Could not load any embedding model: {e}")


# Streamlit-compatible cache decorator
def get_cached_model() -> Tuple[str, Any]:
    """Load embedding model once, shared across sessions."""
    try:
        import streamlit as st
        # Check if st.cache_resource is available and functioning
        if hasattr(st, "cache_resource"):
            @st.cache_resource(show_spinner=False)
            def _cached_loader():
                return _load_model_uncached()
            return _cached_loader()
    except Exception:
        pass

    # Standalone / non-streamlit execution
    return _load_model_lru()


@lru_cache(maxsize=1)
def _load_model_lru() -> Tuple[str, Any]:
    return _load_model_uncached()


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of text passages.
    Returns L2-normalised float32 array of shape (N, D).
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    backend, model = get_cached_model()

    if backend == "fastembed":
        embeddings = list(model.embed(texts))
        arr = np.array(embeddings, dtype=np.float32)
    else:
        # sentence-transformers
        arr = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    # Ensure L2 normalization
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string with BGE query prefix.
    Returns L2-normalised float32 array of shape (D,).
    """
    backend, model = get_cached_model()

    if backend == "fastembed":
        # fastembed query_embed handles proper query prefix for retrieval models
        q_emb = list(model.query_embed([query]))[0]
        arr = np.array(q_emb, dtype=np.float32)
    else:
        # sentence-transformers: prepend prefix
        q_str = settings.bge_query_prefix + query
        arr = model.encode(
            q_str,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.astype(np.float32)
