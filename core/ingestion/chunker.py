"""
Chunker
========
Splits extracted pages/slides into overlapping chunks.
Target: ~200 tokens, ~40 overlap. Splits on headings → paragraphs → sentences.
Every chunk gets full metadata. Title prefix added for context.
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
    doc_id: str
    doc_name: str
    page_or_slide: int
    chunk_index: int
    collection: str
    extraction_method: str = "text"    # "text" or "ocr"
    heading: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "collection": self.collection,
            "metadata": {
                "doc_id": self.doc_id,
                "doc_name": self.doc_name,
                "page_or_slide": self.page_or_slide,
                "chunk_index": self.chunk_index,
                "extraction_method": self.extraction_method,
                "heading": self.heading or "",
                **self.metadata,
            },
        }


def _approx_tokens(text: str) -> int:
    """Approximate token count: ~4 chars per token (BPE avg)."""
    return max(1, len(text) // 4)


def _split_text(text: str, target: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.
    Splits on double-newlines (paragraphs) first, then sentences.
    """
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        p_tokens = _approx_tokens(para)

        if p_tokens > target * 2:
            # Para too long — split on sentences
            sentences = re.split(r"(?<=[.?!])\s+", para)
            for sent in sentences:
                st = _approx_tokens(sent)
                if current_tokens + st > target and current_parts:
                    chunks.append("\n\n".join(current_parts))
                    # Keep overlap
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
                # Keep overlap
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

    return chunks or [text[:800]]  # fallback


def chunk_pages(
    pages: list[dict],
    doc_name: str,
    collection: str,
    doc_id: str | None = None,
) -> list[Chunk]:
    """
    Chunk a list of page/slide dicts.

    Each page dict must have:
      page_or_slide: int
      text: str
      extraction_method: str  ("text" or "ocr")
      heading: str (optional)
    """
    from core.config import settings

    doc_id = doc_id or str(uuid.uuid4())
    target = settings.chunk_target_tokens
    overlap = settings.chunk_overlap_tokens
    chunks: list[Chunk] = []
    chunk_idx = 0

    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue

        page_num = page.get("page_or_slide", 0)
        method = page.get("extraction_method", "text")
        heading = page.get("heading", "")

        # Prefix with heading/slide title for context
        prefix = f"[{heading}] " if heading else ""

        splits = _split_text(text, target, overlap)
        for raw_chunk in splits:
            chunk_text = (prefix + raw_chunk).strip()
            if not chunk_text:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                doc_id=doc_id,
                doc_name=doc_name,
                page_or_slide=page_num,
                chunk_index=chunk_idx,
                collection=collection,
                extraction_method=method,
                heading=heading or None,
            ))
            chunk_idx += 1

    return chunks
