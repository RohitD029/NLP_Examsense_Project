"""
Chunker
========
Splits extracted pages/slides into overlapping chunks.
Target: ~200 tokens, ~40 overlap.
Splits on headings -> paragraphs -> sentences.
Prefixed with section/slide title.
Metadata on every chunk: doc_name, page_or_slide (or chapter), chunk_index, heading.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_name: str
    page_or_slide: int
    chunk_index: int
    heading: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        meta = {
            "doc_name": self.doc_name,
            "page_or_slide": self.page_or_slide,
            "chunk_index": self.chunk_index,
            "heading": self.heading or "",
            **self.metadata,
        }
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": meta,
        }


def _approx_tokens(text: str) -> int:
    """Approximate token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _split_text(text: str, target: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks (~target tokens, ~overlap tokens).
    Splits on double newlines (paragraphs) first, then sentences.
    """
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        p_tokens = _approx_tokens(para)

        if p_tokens > target * 2:
            # Paragraph is too long — split on sentences
            sentences = re.split(r"(?<=[.?!])\s+", para)
            for sent in sentences:
                st = _approx_tokens(sent)
                if current_tokens + st > target and current_parts:
                    chunks.append("\n\n".join(current_parts))
                    # Retain overlap from end of current_parts
                    overlap_parts: list[str] = []
                    overlap_tokens = 0
                    for part in reversed(current_parts):
                        pt = _approx_tokens(part)
                        if overlap_tokens + pt <= overlap:
                            overlap_parts.insert(0, part)
                            overlap_tokens += pt
                        else:
                            break
                    current_parts = overlap_parts
                    current_tokens = overlap_tokens
                current_parts.append(sent)
                current_tokens += st
        else:
            if current_tokens + p_tokens > target and current_parts:
                chunks.append("\n\n".join(current_parts))
                # Retain overlap
                overlap_parts = []
                overlap_tokens = 0
                for part in reversed(current_parts):
                    pt = _approx_tokens(part)
                    if overlap_tokens + pt <= overlap:
                        overlap_parts.insert(0, part)
                        overlap_tokens += pt
                    else:
                        break
                current_parts = overlap_parts
                current_tokens = overlap_tokens
            current_parts.append(para)
            current_tokens += p_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks or [text[:800]]


def chunk_pages(
    pages: list[dict],
    doc_name: str,
    target_tokens: int = 200,
    overlap_tokens: int = 40,
) -> list[Chunk]:
    """
    Chunk extracted pages/slides/chapters.

    Each item in pages:
      page_or_slide: int
      text: str
      heading: str (optional)
      is_image_only: bool (optional)
    """
    chunks: list[Chunk] = []
    chunk_idx = 0

    for page in pages:
        if page.get("is_image_only", False):
            continue

        text = page.get("text", "").strip()
        if not text:
            continue

        page_num = page.get("page_or_slide", 1)
        heading = (page.get("heading") or "").strip()

        # Prefix chunk with section/slide title for grounding context
        prefix = f"[{heading}] " if heading else ""

        splits = _split_text(text, target=target_tokens, overlap=overlap_tokens)
        for raw_split in splits:
            chunk_text = (prefix + raw_split).strip()
            if not chunk_text:
                continue

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                doc_name=doc_name,
                page_or_slide=page_num,
                chunk_index=chunk_idx,
                heading=heading or None,
            ))
            chunk_idx += 1

    return chunks
