"""Scrapes URLs from sources.json into clean text. Logs failures rather than
crashing. Falls back to bundled synthetic fixtures if scraped content is too
thin to demonstrate retrieval (Cloudflare/JS-heavy pages often return little).
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import trafilatura

from . import config
from .cleaner import clean_html, normalize_text
from .io_utils import write_json

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_MIN_CONTENT_CHARS = 200  # below this we treat the extraction as failed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:120]


def _fetch(url: str) -> tuple[int, str | None, str | None]:
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        return r.status_code, r.text, None
    except requests.RequestException as e:
        return 0, None, str(e)


def _extract(html: str) -> tuple[str, str]:
    """Try trafilatura first, fall back to BeautifulSoup."""
    title = ""
    text = ""
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
    except Exception:  # noqa: BLE001 - trafilatura raises a variety of types
        extracted = None
    if extracted and len(extracted) >= _MIN_CONTENT_CHARS:
        text = normalize_text(extracted)
    bs_title, bs_text = clean_html(html)
    title = bs_title or ""
    if not text or len(text) < len(bs_text):
        text = bs_text
    return title, text


def scrape_all(urls: list[str]) -> list[dict[str, Any]]:
    """Scrape each URL and return a list of cleaned page records.

    Always returns one record per URL, with status='success' or 'failed'.
    Saves raw HTML for successful pages under artifacts/raw_pages/.
    """
    config.ensure_dirs()
    records: list[dict[str, Any]] = []
    for url in urls:
        status, html, err = _fetch(url)
        ts = _now_iso()
        if html is None or status >= 400:
            records.append(
                {
                    "source_url": url,
                    "status": "failed",
                    "title": "",
                    "clean_text": "",
                    "scraped_at": ts,
                    "error": err or f"http_status={status}",
                }
            )
            continue
        # Save raw HTML
        raw_path = config.RAW_PAGES_DIR / f"{_slug(url)}.html"
        try:
            raw_path.write_text(html, encoding="utf-8")
        except OSError:
            pass  # raw HTML is best-effort

        title, text = _extract(html)
        if not text or len(text) < _MIN_CONTENT_CHARS:
            records.append(
                {
                    "source_url": url,
                    "status": "failed",
                    "title": title,
                    "clean_text": "",
                    "scraped_at": ts,
                    "error": f"thin_content len={len(text)}",
                }
            )
            continue
        records.append(
            {
                "source_url": url,
                "status": "success",
                "title": title,
                "clean_text": text,
                "scraped_at": ts,
                "error": None,
            }
        )
        time.sleep(0.4)  # be polite
    return records


def _approx_token_count(text: str) -> int:
    # Cheap heuristic before tiktoken is loaded; chars/4 is the usual rule.
    return max(1, len(text) // 4)


def supplement_with_fixtures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """If the total successful content is below SYNTHETIC_TOKEN_FLOOR, append
    fixture records (status='success', source_url=file://...) so the corpus
    has enough material to demonstrate retrieval and grounding.

    The original scrape failures stay in the record list so validation can
    still verify per-source ingestion outcomes.
    """
    success_chars = sum(len(r["clean_text"]) for r in records if r["status"] == "success")
    success_tokens = _approx_token_count(" " * success_chars) * 1  # ~chars/4
    if success_tokens >= config.SYNTHETIC_TOKEN_FLOOR:
        return records

    fixture_dir = config.SYNTHETIC_FIXTURES_DIR
    if not fixture_dir.exists():
        return records

    augmented = list(records)
    for md_path in sorted(fixture_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else md_path.stem
        # Strip the leading '# Title' line from body
        body = re.sub(r"^#\s+.+\n+", "", text, count=1, flags=re.MULTILINE)
        augmented.append(
            {
                "source_url": f"fixture://{md_path.name}",
                "status": "success",
                "title": title,
                "clean_text": normalize_text(body),
                "scraped_at": _now_iso(),
                "error": None,
                "source_type": "synthetic",
            }
        )
    return augmented


def scrape_and_save(urls: list[str]) -> list[dict[str, Any]]:
    records = scrape_all(urls)
    records = supplement_with_fixtures(records)
    write_json(config.CLEANED_PAGES_PATH, records)
    success = sum(1 for r in records if r["status"] == "success")
    failed = sum(1 for r in records if r["status"] == "failed")
    print(f"[scraper] {success} success, {failed} failed -> {config.CLEANED_PAGES_PATH}")
    return records
