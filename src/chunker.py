"""Token-bounded chunker. Targets 200-350 tokens per chunk; merges tiny
sections, splits long ones; preserves headings; SHA256-hashes each chunk.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import tiktoken

from . import config
from .cleaner import split_sections
from .io_utils import write_json

_ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> list[int]:
    return _ENC.encode(text)


def token_count(text: str) -> int:
    return len(_tokens(text))


def _split_long(text: str, heading: str, target: int = config.CHUNK_TOKEN_TARGET, hard_max: int = config.CHUNK_TOKEN_MAX) -> list[str]:
    """Split a long block of text into chunks of ~target tokens, breaking on
    sentence boundaries where possible. Never produces a piece > hard_max."""
    if token_count(text) <= hard_max:
        return [text]

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    heading_tokens = token_count(heading) if heading else 0
    target_budget = max(1, target - heading_tokens)
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s_tokens = token_count(s)
        if s_tokens > hard_max:
            # The sentence itself is too big — hard split by tokens.
            t = _tokens(s)
            for i in range(0, len(t), target_budget):
                piece = _ENC.decode(t[i : i + target_budget])
                pieces.append(piece)
            continue
        if current_tokens + s_tokens > target_budget and current:
            pieces.append(" ".join(current))
            current = [s]
            current_tokens = s_tokens
        else:
            current.append(s)
            current_tokens += s_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def _content_hash(source_url: str, section_title: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(source_url.encode("utf-8"))
    h.update(b"\x1f")
    h.update(section_title.encode("utf-8"))
    h.update(b"\x1f")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:48] or "section"


def chunk_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the corpus dict from cleaned page records.

    Returns a dict with keys: corpus_version, chunks, sources.
    Only success pages contribute chunks. Failed pages still appear under
    sources for traceability.
    """
    chunks: list[dict[str, Any]] = []
    sources_meta: list[dict[str, Any]] = []
    for page in pages:
        sources_meta.append(
            {
                "source_url": page["source_url"],
                "title": page.get("title", ""),
                "status": page["status"],
                "scraped_at": page.get("scraped_at"),
                "error": page.get("error"),
                "source_type": page.get("source_type", "scraped"),
            }
        )
        if page["status"] != "success":
            continue
        text = page.get("clean_text", "")
        if not text.strip():
            continue
        sections = split_sections(text) or [(page.get("title", ""), text)]

        # Merge tiny consecutive sections together until they reach the min target.
        # When merging, keep all headings: the joined heading list serves as
        # section_title and the body inlines '## {heading}' between sections so
        # retrieval still matches against the inner topic words.
        merged: list[tuple[str, str]] = []
        buf_headings: list[str] = []
        buf_body = ""

        def _flush() -> None:
            nonlocal buf_headings, buf_body
            if not buf_body:
                return
            joined = " | ".join(h for h in buf_headings if h) or page.get("title", "")
            merged.append((joined, buf_body))
            buf_headings = []
            buf_body = ""

        for heading, body in sections:
            heading_str = heading.strip() if heading else ""
            if not body.strip() and not heading_str:
                continue
            section_block = body
            if heading_str:
                section_block = f"## {heading_str}\n{body}".strip()
            section_tokens = token_count(section_block)

            # If this section is already large enough to stand alone, flush
            # whatever tiny material was buffered first so it doesn't dilute
            # this section's topic.
            if section_tokens >= config.CHUNK_TOKEN_MIN:
                _flush()
                merged.append((heading_str or page.get("title", ""), section_block))
                continue

            combined = (buf_body + "\n\n" + section_block).strip() if buf_body else section_block
            new_heading_list = buf_headings + ([heading_str] if heading_str else [])
            if token_count(combined) < config.CHUNK_TOKEN_MIN:
                buf_headings = new_heading_list
                buf_body = combined
                continue
            buf_headings = new_heading_list
            buf_body = combined
            _flush()
        _flush()

        # If after merging we still have nothing, skip
        if not merged:
            continue

        chunk_index_counter = 0
        for heading, body in merged:
            pieces = _split_long(body, heading)
            for piece in pieces:
                tk = token_count(piece)
                if tk == 0:
                    continue
                chunk_id = (
                    f"{_slug(page.get('title') or page['source_url'])}_"
                    f"{_slug(heading) if heading else 'body'}_{chunk_index_counter:04d}"
                )
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source_url": page["source_url"],
                        "section_title": heading or page.get("title", ""),
                        "chunk_index": chunk_index_counter,
                        "token_count": tk,
                        "content_hash": _content_hash(page["source_url"], heading or "", piece),
                        "text": piece,
                        "source_type": page.get("source_type", "scraped"),
                    }
                )
                chunk_index_counter += 1

    corpus = {
        "corpus_version": datetime.now(timezone.utc).isoformat(),
        "chunks": chunks,
        "sources": sources_meta,
    }
    return corpus


def chunk_and_save(pages: list[dict[str, Any]]) -> dict[str, Any]:
    corpus = chunk_pages(pages)
    write_json(config.CORPUS_PATH, corpus)
    print(f"[chunker] {len(corpus['chunks'])} chunks -> {config.CORPUS_PATH}")
    return corpus
