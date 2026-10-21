"""
UI Components
==============
Reusable HTML components for the ExamSense Streamlit app.
All components return HTML strings to be injected via st.markdown(unsafe_allow_html=True).
"""
from __future__ import annotations


def hero_header(title: str = "ExamSense", tagline: str = "AI-Powered Study Assistant") -> str:
    """Gradient hero header."""
    return f"""
    <div class="hero-header">
        <h1>{title}</h1>
        <p>{tagline}</p>
    </div>
    """


def doc_tile(filename: str, file_type: str, status: str = "Ready") -> str:
    """A clean document tile for the library view."""
    type_icons = {
        "pdf": "📕", "pptx": "📙", "ppt": "📙",
        "docx": "📘", "doc": "📘",
        "epub": "📗", "txt": "📄", "md": "📝",
    }
    icon = type_icons.get(file_type.lower().lstrip("."), "📄")
    status_class = "status-ready" if status.lower() == "ready" else "status-error"

    return f"""
    <div class="doc-tile">
        <div class="doc-icon">{icon}</div>
        <div class="doc-info">
            <div class="doc-name">{filename}</div>
            <div class="doc-meta">.{file_type.lstrip('.')}</div>
        </div>
        <div class="doc-status {status_class}">{status}</div>
    </div>
    """


def empty_state(icon: str, title: str, description: str) -> str:
    """Empty state placeholder with icon."""
    return f"""
    <div class="empty-state">
        <div class="empty-icon">{icon}</div>
        <div class="empty-title">{title}</div>
        <div class="empty-desc">{description}</div>
    </div>
    """


def status_pill(text: str, is_ready: bool = True) -> str:
    """Status pill — green for ready, amber/red for missing/error."""
    css_class = "status-ready" if is_ready else "status-error"
    dot_class = "green" if is_ready else "red"
    return f"""<span class="status-pill {css_class}">
        <span class="pulse-dot {dot_class}"></span>
        {text}
    </span>"""


def error_card(title: str, message: str) -> str:
    """Styled error card that replaces raw tracebacks."""
    return f"""
    <div class="glass-card" style="border-color:rgba(244,63,94,0.3)">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span style="font-size:1.3rem">⚠️</span>
            <span style="color:#F43F5E;font-weight:600;font-size:1rem">{title}</span>
        </div>
        <p style="color:#A09CB5;font-size:0.9rem;margin:0">{message}</p>
    </div>
    """


def not_found_card(message: str = "I couldn't find this in your documents. Try rephrasing or upload more material.") -> str:
    """Polished 'Not found' card for abstention."""
    return f"""
    <div class="glass-card" style="border-color:rgba(245,158,11,0.25);text-align:center;padding:1.5rem">
        <div style="font-size:2rem;margin-bottom:0.6rem">🔍</div>
        <p style="color:#E2E0F0;font-size:0.95rem;margin:0;max-width:450px;margin:0 auto">
            {message}
        </p>
    </div>
    """
