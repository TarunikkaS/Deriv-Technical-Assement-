"""Per-query audit assembly. Every test query produces one record that is
collected into artifacts/answer_audit.json at the end of the run."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from . import config
from .io_utils import write_json


@dataclass
class QueryAudit:
    query_id: str
    query: str
    conversation_context_used: str = ""
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    confidence: dict[str, Any] = field(default_factory=dict)
    fallback_triggered: bool = False
    generated_answer: str | None = None
    grounding_verification: list[dict[str, Any]] | None = None
    regeneration: dict[str, Any] | None = None
    quality_scores: dict[str, Any] | None = None
    final_response: str = ""
    stages: list[str] = field(default_factory=list)


def write_audit(audits: list[QueryAudit]) -> None:
    write_json(config.ANSWER_AUDIT_PATH, [asdict(a) for a in audits])
