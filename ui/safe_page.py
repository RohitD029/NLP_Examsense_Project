"""
Safe Page Wrapper
=================
Catches unexpected exceptions during page render, logs full tracebacks to
logs/app.log (or stderr if filesystem is read-only), and renders a clean friendly error card.
Never swallows Streamlit control-flow exceptions (st.rerun, st.stop, st.switch_page).
"""
from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import streamlit as st

logger = logging.getLogger("examsense.safe_page")

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
APP_LOG = LOGS_DIR / "app.log"


def is_streamlit_control_flow(e: BaseException) -> bool:
    """Detect Streamlit's internal control-flow exceptions that must never be caught."""
    name = type(e).__name__
    mod = getattr(type(e), "__module__", "")
    return (
        name in ("StopException", "RerunException", "SwitchPageException")
        or "streamlit.runtime.scriptrunner" in mod
    )


def safe_page(render_fn: Callable[[], None]) -> None:
    """
    Execute a page function safely.
    Catches unhandled errors, logs traceback to logs/app.log or stderr, and shows
    a friendly error card without raw tracebacks.
    """
    try:
        render_fn()
    except BaseException as e:
        if is_streamlit_control_flow(e):
            raise e

        # Format traceback
        tb = traceback.format_exc()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n{'='*70}\n[{timestamp}] UNHANDLED ERROR: {e}\n{tb}\n{'='*70}\n"

        # Ensure logs directory exists and write to logs/app.log
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(APP_LOG, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception:
            sys.stderr.write(log_entry)
            sys.stderr.flush()

        logger.error(f"Page render error: {e}", exc_info=True)

        from ui.components import error_card
        st.markdown(
            error_card(
                "Unable to display page",
                "An unexpected issue occurred while loading this view. Please try again or re-upload your document.",
            ),
            unsafe_allow_html=True,
        )
