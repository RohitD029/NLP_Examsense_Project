"""
UI Components
==============
Reusable HTML components for the ExamSense Streamlit app.
All components return HTML strings to be injected via st.markdown(unsafe_allow_html=True).
"""

import math


def hero_header(title: str = "ExamSense", tagline: str = "AI-powered study assistant with exam frequency intelligence") -> str:
    """Gradient hero header for the Ask page."""
    return f"""
    <div class="hero-header">
        <h1>{title}</h1>
        <p>{tagline}</p>
    </div>
    """


def metric_card(value: str, label: str, icon: str = "") -> str:
    """A single glassmorphic metric card."""
    icon_html = f'<div style="font-size:1.4rem;margin-bottom:0.3rem">{icon}</div>' if icon else ""
    return f"""
    <div class="metric-card">
        {icon_html}
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def citation_chip(doc_name: str, page: int | str) -> str:
    """A citation chip showing document name and page."""
    return f"""<span class="citation-chip">
        <span class="chip-icon">📄</span>
        {doc_name} · p.{page}
    </span>"""


def citation_chips(citations: list[dict]) -> str:
    """Render multiple citation chips from a list of citation dicts."""
    if not citations:
        return ""
    chips = []
    for c in citations:
        doc = c.get("source_document", "Unknown")
        page = c.get("page_number", "?")
        chips.append(citation_chip(doc, page))
    return f'<div style="margin:8px 0;line-height:2.2">{"".join(chips)}</div>'


def exam_frequency_badge(frequency: int, years: list[str]) -> str:
    """Exam frequency badge with fire icon."""
    if frequency == 0:
        return ""
    years_str = ", ".join(years) if years else "—"
    heat = "🔥" if frequency >= 3 else "📊"
    return f"""<span class="exam-badge">
        <span class="badge-icon">{heat}</span>
        Asked {frequency}× &nbsp;·&nbsp; {years_str}
    </span>"""


def confidence_gauge(confidence: float, size: int = 56) -> str:
    """SVG ring confidence gauge."""
    conf = max(0.0, min(1.0, confidence))
    pct = conf * 100
    r = (size - 8) / 2
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    dash = circumference * conf
    gap = circumference - dash

    # Color based on confidence
    if conf >= 0.7:
        color = "#10B981"  # emerald
    elif conf >= 0.4:
        color = "#F59E0B"  # amber
    else:
        color = "#F43F5E"  # rose

    return f"""
    <div class="confidence-gauge">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <circle cx="{cx}" cy="{cy}" r="{r}"
                fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="5"/>
            <circle cx="{cx}" cy="{cy}" r="{r}"
                fill="none" stroke="{color}" stroke-width="5"
                stroke-dasharray="{dash:.1f} {gap:.1f}"
                stroke-linecap="round"
                transform="rotate(-90 {cx} {cy})"
                style="transition: stroke-dasharray 0.8s ease;"/>
            <text x="{cx}" y="{cy + 1}" text-anchor="middle" dominant-baseline="central"
                fill="{color}" font-size="{size * 0.25:.0f}" font-weight="700"
                font-family="Inter, sans-serif">
                {pct:.0f}%
            </text>
        </svg>
        <span style="color:{color};font-weight:600;font-size:0.85rem">
            {"High" if conf >= 0.7 else "Medium" if conf >= 0.4 else "Low"} Confidence
        </span>
    </div>
    """


def doc_tile(filename: str, file_type: str, chunks: int, status: str = "Indexed", doc_id: str = "") -> str:
    """A document tile for the library view."""
    type_icons = {
        "pdf": "📕", "pptx": "📙", "ppt": "📙",
        "docx": "📘", "doc": "📘",
        "png": "🖼️", "jpg": "🖼️", "jpeg": "🖼️",
        "tiff": "🖼️", "tif": "🖼️", "bmp": "🖼️", "webp": "🖼️",
    }
    icon = type_icons.get(file_type.lower().lstrip("."), "📄")

    return f"""
    <div class="doc-tile">
        <div class="doc-icon">{icon}</div>
        <div class="doc-info">
            <div class="doc-name">{filename}</div>
            <div class="doc-meta">{chunks} chunks · .{file_type.lstrip('.')}</div>
        </div>
        <div class="doc-status status-indexed">{status}</div>
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
    """Status pill — green for ready, red for error."""
    css_class = "status-ready" if is_ready else "status-error"
    dot_class = "green" if is_ready else "red"
    return f"""<span class="status-pill {css_class}">
        <span class="pulse-dot {dot_class}"></span>
        {text}
    </span>"""


def sidebar_stat(label: str, value: str | int) -> str:
    """Small stat line for the sidebar."""
    return f"""<div style="display:flex;justify-content:space-between;align-items:center;
        padding:4px 0;font-size:0.85rem">
        <span style="color:#A09CB5">{label}</span>
        <span style="color:#E2E0F0;font-weight:600">{value}</span>
    </div>"""


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


def not_found_card() -> str:
    """Polished 'Not found in your material' card for abstention."""
    return """
    <div class="glass-card" style="border-color:rgba(245,158,11,0.25);text-align:center;padding:2rem">
        <div style="font-size:2.5rem;margin-bottom:0.8rem">🔍</div>
        <div style="color:#F59E0B;font-weight:700;font-size:1.1rem;margin-bottom:0.5rem">
            Not Found in Your Material
        </div>
        <p style="color:#A09CB5;font-size:0.9rem;margin:0;max-width:400px;margin:0 auto">
            The question doesn't closely match any content in your uploaded documents.
            Try rephrasing, or upload additional material covering this topic.
        </p>
    </div>
    """
