"""Optional: clusters the lowest-confidence queries (including fallbacks)
into Help Centre content gaps."""

from __future__ import annotations

from typing import Any

from . import config, prompts
from .audit import QueryAudit
from .io_utils import write_json
from .llm_client import LLMClient


def detect_gaps(
    *,
    client: LLMClient,
    audits: list[QueryAudit],
    bottom_k: int = 5,
) -> list[dict[str, Any]]:
    """Pick fallback queries plus the lowest-confidence non-fallback ones,
    then ask the LLM to cluster them into improvement topics."""
    weak: list[dict[str, Any]] = []
    for a in audits:
        weak.append(
            {
                "query_id": a.query_id,
                "query": a.query,
                "top_score": float(a.confidence.get("top_score", 0.0) or 0.0),
                "fallback": bool(a.fallback_triggered),
            }
        )
    # Always include all fallbacks, plus the lowest-score non-fallbacks up to bottom_k
    fallbacks = [w for w in weak if w["fallback"]]
    non_fallback_sorted = sorted(
        [w for w in weak if not w["fallback"]], key=lambda w: w["top_score"]
    )
    selection = fallbacks + non_fallback_sorted[: max(0, bottom_k - len(fallbacks))]
    if not selection:
        write_json(config.KNOWLEDGE_GAP_REPORT_PATH, [])
        return []

    user = prompts.gap_user_prompt(selection)
    parsed = client.complete_json(
        [{"role": "user", "content": user}],
        system=prompts.GAP_SYSTEM,
        stage="gap_detection",
        query_id=None,
        input_artifacts=[str(config.ANSWER_AUDIT_PATH)],
        output_artifact=str(config.KNOWLEDGE_GAP_REPORT_PATH),
    )

    if isinstance(parsed, dict):
        for key in ("topics", "clusters", "gaps", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            if "topic" in parsed:
                parsed = [parsed]
    if not isinstance(parsed, list):
        parsed = []

    cleaned: list[dict[str, Any]] = []
    valid_ids = {w["query_id"] for w in selection}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip() or "Uncategorized"
        ids = item.get("query_ids", []) or []
        if not isinstance(ids, list):
            ids = []
        ids = [str(i) for i in ids if str(i) in valid_ids]
        cleaned.append(
            {
                "topic": topic,
                "query_ids": ids,
                "evidence": str(item.get("evidence", "")).strip(),
                "recommended_content_improvement": str(
                    item.get("recommended_content_improvement", "")
                ).strip(),
            }
        )

    write_json(config.KNOWLEDGE_GAP_REPORT_PATH, cleaned)
    return cleaned
