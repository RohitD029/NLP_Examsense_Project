"""
LLM Client
===========
Uses openai package with Groq's OpenAI-compatible endpoint.
Primary model → fallback model → friendly error card.
Rate limiter + exponential backoff on 429.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are ExamSense, an academic study assistant. Answer ONLY from the numbered \
source passages below. Rules:
1. Cite sources as [1], [2] etc. matching the passage numbers given to you.
2. Never mention document names, file names, or page numbers in your prose — \
   the UI shows citations separately.
3. If the passages do not contain enough information to answer, respond with \
   exactly: ABSTAIN
4. Keep your answer concise (under 300 words). Use bullet points for lists.
5. End with **Confidence: X.X** (0.0–1.0).
"""


def get_active_groq_key() -> str:
    """Single source of truth for API key: st.session_state falling back to settings/.env."""
    try:
        import streamlit as st
        val = st.session_state.get("sidebar_groq_key", "")
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass
    from core.config import settings
    return (settings.groq_api_key or "").strip()


def generate_answer(
    query: str,
    context_chunks: list[dict],
    max_tokens: int | None = None,
    system_prompt: str | None = None,
) -> dict:
    """
    Generate a grounded answer.

    Returns dict:
      answer, model_used, provider, raw_confidence, abstained
    """
    from core.config import settings

    if not context_chunks:
        return _no_key_response("No context chunks provided.")

    active_key = get_active_groq_key()
    if not active_key:
        return _no_key_response("No Groq API key configured. Enter your key in Settings.")

    context_text, used_tokens = _format_context(context_chunks, settings.llm_context_token_limit)
    user_message = f"PASSAGES:\n{context_text}\n\nQUESTION: {query}"
    tokens = max_tokens or settings.llm_max_tokens

    for model in [settings.groq_primary_model, settings.groq_fallback_model]:
        result = _call_with_retry(
            user_message=user_message,
            model=model,
            max_tokens=tokens,
            settings=settings,
            api_key=active_key,
            system_prompt=system_prompt or SYSTEM_PROMPT,
        )
        if result:
            return result

    return _no_key_response("All LLM providers failed after retries.")


def _call_with_retry(
    user_message: str,
    model: str,
    max_tokens: int,
    settings,
    api_key: str,
    system_prompt: str,
    max_retries: int = 3,
) -> Optional[dict]:
    from core.generation.rate_limiter import get_limiter
    from openai import OpenAI, RateLimitError, APIConnectionError

    limiter = get_limiter()
    est_tokens = len(user_message.split()) + max_tokens

    for attempt in range(max_retries):
        try:
            limiter.wait_if_needed(est_tokens)

            client = OpenAI(
                api_key=api_key,
                base_url=settings.groq_base_url,
                timeout=25.0,
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=settings.llm_temperature,
            )

            text = response.choices[0].message.content or ""
            tokens_used = getattr(response.usage, "total_tokens", est_tokens)
            limiter.record(tokens_used)

            abstained = text.strip().upper().startswith("ABSTAIN")
            confidence = _parse_confidence(text)

            return {
                "answer": text,
                "model_used": model,
                "provider": "groq",
                "raw_confidence": confidence,
                "abstained": abstained,
            }

        except RateLimitError as e:
            retry_after = _parse_retry_after(str(e))
            wait = retry_after or (2 ** attempt * 5)
            logger.warning(f"Rate limited on {model}, waiting {wait}s (attempt {attempt+1})")
            time.sleep(wait)

        except APIConnectionError as e:
            logger.warning(f"Connection error on {model}: {e} (attempt {attempt+1})")
            time.sleep(2 ** attempt)

        except Exception as e:
            logger.error(f"LLM call failed on {model}: {e}")
            return None

    return None


def _format_context(chunks: list[dict], token_limit: int) -> tuple[str, int]:
    """Format chunks as numbered passages, respecting token budget."""
    parts = []
    total_tokens = 0
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        doc = meta.get("doc_name", "?")
        page = meta.get("page_or_slide", "?")
        text = chunk["text"]
        entry = f"[{i}] {doc} (p.{page})\n{text}"
        entry_tokens = len(entry.split())
        if total_tokens + entry_tokens > token_limit and parts:
            break
        parts.append(entry)
        total_tokens += entry_tokens
    return "\n\n---\n\n".join(parts), total_tokens


def _strip_invalid_citations(text: str, n_chunks: int) -> str:
    """Remove any [N] references the model invented beyond n_chunks."""
    def _replace(m):
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_chunks else ""
    return re.sub(r"\[(\d+)\]", _replace, text)


def _parse_confidence(text: str) -> float:
    m = re.search(r"\*?\*?[Cc]onfidence:?\s*([\d.]+)\*?\*?", text)
    if m:
        try:
            v = float(m.group(1))
            if v > 1.0:
                v = v / 10.0 if v <= 10 else v / 100.0
            return round(min(max(v, 0.0), 1.0), 2)
        except ValueError:
            pass
    return 0.5


def _parse_retry_after(error_str: str) -> float | None:
    m = re.search(r"retry.after[=: ]+([0-9.]+)", error_str, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _no_key_response(reason: str) -> dict:
    return {
        "answer": "",
        "model_used": "none",
        "provider": "none",
        "raw_confidence": 0.0,
        "abstained": True,
        "error": reason,
    }


def test_groq_connection(api_key: str | None = None) -> dict:
    """Test connection to Groq API and verify active model."""
    from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
    from core.config import settings

    key = (api_key or get_active_groq_key()).strip()
    if not key:
        return {
            "ready": False,
            "status_text": "✗ Missing API Key",
            "reason": "Missing key. Enter your Groq API key in Settings below.",
            "provider": "Groq",
            "model": "None",
        }

    try:
        client = OpenAI(api_key=key, base_url=settings.groq_base_url, timeout=7.0)
        models = client.models.list()
        avail = [m.id for m in models.data]

        # Ensure active model is one that actually exists on this key
        target_model = settings.groq_primary_model
        if target_model not in avail:
            chat_models = [m for m in avail if any(x in m for x in ("gpt-oss", "llama", "qwen", "allam"))]
            if chat_models:
                target_model = chat_models[0]
                settings.groq_primary_model = target_model

        model_short = target_model.split("/")[-1]
        return {
            "ready": True,
            "status_text": f"✓ Ready: {model_short}",
            "reason": "Connected successfully to Groq.",
            "provider": "Groq",
            "model": target_model,
        }
    except AuthenticationError:
        return {
            "ready": False,
            "status_text": "✗ Invalid API Key",
            "reason": "Key rejected by Groq (401 Unauthorized).",
            "provider": "Groq",
            "model": "None",
        }
    except RateLimitError:
        return {
            "ready": False,
            "status_text": "✗ Rate Limited",
            "reason": "Groq rate limit reached (429). Wait a minute.",
            "provider": "Groq",
            "model": "None",
        }
    except APIConnectionError:
        return {
            "ready": False,
            "status_text": "✗ Connection Error",
            "reason": "Cannot reach Groq API. Check your internet connection.",
            "provider": "Groq",
            "model": "None",
        }
    except Exception as e:
        return {
            "ready": False,
            "status_text": "✗ Connection Failed",
            "reason": str(e),
            "provider": "Groq",
            "model": "None",
        }


def check_llm_status() -> dict:
    """Return status for the sidebar pill. Caches results in session_state when in Streamlit."""
    key = get_active_groq_key()
    if not key:
        return {
            "ready": False,
            "status_text": "✗ No API Key",
            "reason": "Please enter your Groq API key in Settings.",
            "provider": "Groq",
            "model": "None",
        }

    # Check cached status in session_state
    try:
        import streamlit as st
        cached = st.session_state.get("groq_conn_status")
        if cached and cached.get("key") == key:
            return cached["status"]
    except Exception:
        pass

    # Perform connection check
    status = test_groq_connection(key)
    try:
        import streamlit as st
        st.session_state["groq_conn_status"] = {"key": key, "status": status}
    except Exception:
        pass
    return status

