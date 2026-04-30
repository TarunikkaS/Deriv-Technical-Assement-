"""HTML cleaning helpers used by the scraper."""

from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup, Tag

_DROP_TAGS = ("script", "style", "noscript", "iframe", "nav", "footer", "header", "form", "aside")
_DROP_ROLES = ("navigation", "banner", "contentinfo", "search")
_DROP_CLASSES = (
    "cookie", "consent", "navbar", "menu", "footer", "header", "breadcrumb",
    "subscribe", "newsletter", "social", "share", "related-articles",
)


def clean_html(html: str) -> tuple[str, str]:
    """Returns (title, cleaned_text). Cleaned text preserves headings on
    their own lines and normalizes whitespace.
    """
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()

    for tag_name in _DROP_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()
    for el in soup.find_all(attrs={"role": True}):
        if str(el.get("role", "")).lower() in _DROP_ROLES:
            el.decompose()
    for el in soup.find_all(class_=True):
        try:
            classes = el.get("class") or []
            if isinstance(classes, str):
                classes = [classes]
            cls = " ".join(classes).lower()
        except (AttributeError, TypeError):
            continue
        if any(token in cls for token in _DROP_CLASSES):
            el.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    if not isinstance(main, Tag):
        return title, ""

    pieces: list[str] = []
    for el in main.descendants:
        if not isinstance(el, Tag):
            continue
        name = el.name.lower()
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            txt = el.get_text(" ", strip=True)
            if txt:
                pieces.append(f"\n## {txt}\n")
        elif name in {"p", "li"}:
            txt = el.get_text(" ", strip=True)
            if txt:
                pieces.append(txt)
    cleaned = "\n".join(pieces)
    cleaned = _normalize(cleaned)
    return title, cleaned


def normalize_text(text: str) -> str:
    return _normalize(text)


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split cleaned text into (heading, body) pairs based on '## ' markers.
    Falls back to a single ('', text) pair if no headings are present.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body or current_heading:
        sections.append((current_heading, current_body))
    return [(h, _normalize("\n".join(b))) for h, b in sections if _normalize("\n".join(b)) or h]
