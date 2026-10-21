"""
LLM Client for Groq
===================
Uses openai package with Groq's OpenAI-compatible endpoint.
Key priority: User's sidebar key > st.secrets["GROQ_API_KEY"] > environment/.env.
Handles retries (primary -> fallback model), backoff on HTTP 429, session caps,
and friendly failure messages without tracebacks.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional, Tuple

from core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Answer ONLY from the numbered excerpts. "
    "Be concise, at most 120 words unless asked for more. "
    "Cite like [1]. "
    "If the answer is not in the excerpts, reply exactly: "
    "I couldn't find this in your documents."
)


def get_active_groq_key() -> str:
    """
    Key priority order:
    1. User's own key typed in sidebar (st.session_state["sidebar_user_groq_key"])
    2. st.secrets["GROQ_API_KEY"]
    3. environment variable / .env (GROQ_API_KEY or EXAMSENSE_GROQ_API_KEY)
    """
    # 1. User typed key in session_state
    try:
        import streamlit as st
        user_key = st.session_state.get("sidebar_user_groq_key", "")
        if isinstance(user_key, str) and user_key.strip():
            return user_key.strip()
    except Exception:
        pass

    # 2. st.secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GROQ_API_KEY" in st.secrets:
                val = st.secrets["GROQ_API_KEY"]
                if isinstance(val, str) and val.strip():
                    return val.strip()
            # Check nested table if present
            if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
                val = st.secrets["groq"]["api_key"]
                if isinstance(val, str) and val.strip():
                    return val.strip()
    except Exception:
        pass

    # 3. Environment variables
    env_key = os.getenv("GROQ_API_KEY") or os.getenv("EXAMSENSE_GROQ_API_KEY") or ""
    return env_key.strip()


def is_user_custom_key() -> bool:
    """Check if the currently active key was provided directly by the user in this session."""
    try:
        import streamlit as st
        user_key = st.session_state.get("sidebar_user_groq_key", "")
        return bool(isinstance(user_key, str) and user_key.strip())
    except Exception:
        return False


def check_session_question_cap() -> Tuple[bool, str]:
    """
    Check if the user has reached the 30 questions session cap when using the shared server key.
    No cap when the user supplies their own key.
    """
    if is_user_custom_key():
        return True, ""

    try:
        import streamlit as st
        asked = st.session_state.get("session_questions_count", 0)
        if asked >= settings.shared_key_max_questions:
            return False, (
                f"You have reached the free session limit of {settings.shared_key_max_questions} questions "
                "on the shared key. Please enter your own free Groq API key in the sidebar to continue."
            )
    except Exception:
        pass

    return True, ""


def increment_session_question_count() -> None:
    """Increment questions asked count if on shared server key."""
    if is_user_custom_key():
        return
    try:
        import streamlit as st
        st.session_state["session_questions_count"] = st.session_state.get("session_questions_count", 0) + 1
    except Exception:
        pass


def test_groq_connection(api_key: Optional[str] = None) -> dict:
    """
    Makes one tiny API call to verify the key.
    Returns dict: ready (bool), status_text (str), message (str).
    """
    key = (api_key or get_active_groq_key()).strip()
    if not key:
        return {
            "ready": False,
            "status_text": "Add key",
            "message": "No Groq API key configured. Enter your key in the sidebar.",
        }

    from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

    try:
        client = OpenAI(
            api_key=key,
            base_url=settings.groq_base_url,
            timeout=8.0,
        )
        # Small call to verify key
        client.models.list()
        return {
            "ready": True,
            "status_text": "Ready",
            "message": "Connected successfully to Groq API.",
        }
    except AuthenticationError:
        return {
            "ready": False,
            "status_text": "Invalid key",
            "message": "Groq API key was rejected (401 Unauthorized).",
        }
    except RateLimitError:
        return {
            "ready": False,
            "status_text": "Rate limited",
            "message": "Groq rate limit reached. Please wait a minute.",
        }
    except APIConnectionError:
        return {
            "ready": False,
            "status_text": "Connection error",
            "message": "Could not connect to Groq. Please check your internet connection.",
        }
    except Exception as e:
        return {
            "ready": False,
            "status_text": "Error",
            "message": f"Connection check failed: {str(e)}",
        }


def generate_answer(
    query: str,
    context_chunks: list[dict],
    system_prompt: str | None = None,
) -> dict:
    """
    Generate short answer using Groq via openai package.
    Sends top 3 chunks (trimmed to ~350 tokens each).
    Retries once on fallback model.
    """
    # 1. Key check
    active_key = get_active_groq_key()
    if not active_key:
        return {
            "answer": "",
            "error": "No Groq API key configured. Please enter your free key in the sidebar.",
            "abstained": True,
        }

    # 2. Session quota check
    can_ask, quota_msg = check_session_question_cap()
    if not can_ask:
        return {
            "answer": "",
            "error": quota_msg,
            "abstained": True,
        }

    if not context_chunks:
        return {
            "answer": "I couldn't find this in your documents. Try rephrasing or upload more material.",
            "error": "",
            "abstained": True,
        }

    # Format excerpts: top 3 chunks, trimmed to ~350 tokens each (~1400 chars)
    formatted_passages = []
    for i, c in enumerate(context_chunks[:settings.top_k_final], 1):
        meta = c.get("metadata", {})
        doc = meta.get("doc_name", "Document")
        page = meta.get("page_or_slide", "?")
        trimmed_text = c["text"][:1400]
        formatted_passages.append(f"[{i}] {doc} (p.{page})\n{trimmed_text}")

    context_str = "\n\n---\n\n".join(formatted_passages)
    user_prompt = f"EXCERPTS:\n{context_str}\n\nQUESTION: {query}"
    sys_prompt = system_prompt or SYSTEM_PROMPT

    from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError

    client = OpenAI(
        api_key=active_key,
        base_url=settings.groq_base_url,
        timeout=25.0,
    )

    models_to_try = [settings.groq_primary_model, settings.groq_fallback_model]
    last_error = ""

    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=settings.llm_max_output_tokens,
                temperature=settings.llm_temperature,
            )

            text = response.choices[0].message.content or ""
            increment_session_question_count()

            abstained = "couldn't find this in your documents" in text.lower()
            return {
                "answer": text.strip(),
                "error": "",
                "model_used": model_name,
                "abstained": abstained,
            }

        except RateLimitError as e:
            # Back off and honor retry-after if provided
            wait_time = _extract_retry_after(str(e)) or 3.0
            logger.warning(f"Rate limited on {model_name}, waiting {wait_time}s...")
            time.sleep(wait_time)
            # Try again once for the same model if 429
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=settings.llm_max_output_tokens,
                    temperature=settings.llm_temperature,
                )
                text = response.choices[0].message.content or ""
                increment_session_question_count()
                return {
                    "answer": text.strip(),
                    "error": "",
                    "model_used": model_name,
                    "abstained": "couldn't find this in your documents" in text.lower(),
                }
            except Exception as retry_err:
                last_error = "Rate limit reached on Groq. Please wait a moment or use your own key in the sidebar."
                logger.warning(f"Retry after 429 failed: {retry_err}")
                continue

        except AuthenticationError:
            return {
                "answer": "",
                "error": "Your Groq API key is invalid or expired. Please check your key in the sidebar.",
                "abstained": True,
            }

        except APIConnectionError:
            last_error = "Could not connect to Groq. Please check your internet connection."

        except Exception as e:
            last_error = f"Groq request failed ({str(e)})."
            logger.warning(f"Model {model_name} failed: {e}")

    return {
        "answer": "",
        "error": last_error or "Unable to generate an answer at this time. Please try again.",
        "abstained": True,
    }


def _extract_retry_after(err_msg: str) -> Optional[float]:
    """Parse retry-after seconds from error message if available."""
    m = re.search(r"retry[-_ ]after[:\s=]+([0-9.]+)", err_msg, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None
