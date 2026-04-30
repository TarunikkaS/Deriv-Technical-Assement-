"""Input file loading + sample creation. The evaluator may swap files at any time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Sample content used only when the file is missing on disk. We write it via
# `ensure_sample_*`, never inline it in the pipeline modules.
_SAMPLE_SOURCES = {
    "sources": [
        "https://deriv.com/help-centre/trading/",
        "https://deriv.com/help-centre/deposits-and-withdrawals/",
        "https://deriv.com/help-centre/accounts/",
        "https://deriv.com/help-centre/security/",
    ]
}

_SAMPLE_TEST_QUERIES = [
    {"id": "Q1", "query": "How do I reset my two-factor authentication if I've lost access to my authenticator app?"},
    {"id": "Q2", "query": "What is the minimum withdrawal amount for bank transfer?"},
    {"id": "Q3", "query": "Can I have more than one real account on Deriv?"},
    {"id": "Q4", "query": "How long does an e-wallet withdrawal typically take?"},
    {"id": "Q5", "query": "What happens to my open positions if I close my account?"},
    {"id": "Q6", "query": "Is there a fee for depositing via cryptocurrency?"},
    {"id": "Q7", "query": "How do I set trading limits as part of responsible gambling tools?"},
    {"id": "Q8", "query": "What documents are required for account verification in Malaysia?"},
]


class InputValidationError(ValueError):
    """Raised when sources.json or test_queries.json is malformed."""


def ensure_sample_sources(path: Path) -> None:
    if path.exists():
        return
    path.write_text(json.dumps(_SAMPLE_SOURCES, indent=2) + "\n", encoding="utf-8")
    print(f"[io] Created sample sources.json at {path}")


def ensure_sample_test_queries(path: Path) -> None:
    if path.exists():
        return
    path.write_text(json.dumps(_SAMPLE_TEST_QUERIES, indent=2) + "\n", encoding="utf-8")
    print(f"[io] Created sample test_queries.json at {path}")


def load_sources(path: Path) -> list[str]:
    ensure_sample_sources(path)
    raw = _read_json(path)
    if not isinstance(raw, dict) or "sources" not in raw:
        raise InputValidationError(f"{path.name} must be a JSON object with a 'sources' key")
    sources = raw["sources"]
    if not isinstance(sources, list) or not sources:
        raise InputValidationError(f"{path.name} 'sources' must be a non-empty list")
    for i, s in enumerate(sources):
        if not isinstance(s, str) or not s.strip():
            raise InputValidationError(f"{path.name} sources[{i}] must be a non-empty string URL")
    return [s.strip() for s in sources]


def load_test_queries(path: Path) -> list[dict[str, str]]:
    ensure_sample_test_queries(path)
    raw = _read_json(path)
    if not isinstance(raw, list) or not raw:
        raise InputValidationError(f"{path.name} must be a non-empty JSON list")
    out: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputValidationError(f"{path.name}[{i}] must be a JSON object")
        if "id" not in item or "query" not in item:
            raise InputValidationError(f"{path.name}[{i}] must contain 'id' and 'query'")
        qid = str(item["id"]).strip()
        query = str(item["query"]).strip()
        if not qid or not query:
            raise InputValidationError(f"{path.name}[{i}] 'id' and 'query' must be non-empty")
        if qid in seen_ids:
            raise InputValidationError(f"{path.name}[{i}] duplicate id '{qid}'")
        seen_ids.add(qid)
        out.append({"id": qid, "query": query})
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise InputValidationError(f"{path.name} is not valid JSON: {e}") from e


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
