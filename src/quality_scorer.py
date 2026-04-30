"""Optional Stage-3: scores the final answer on completeness, specificity,
and tone (0-10 each). Flags operator_review_required if any score < 6."""

from __future__ import annotations

import json
from typing import Any

from . import config, prompts
from .io_utils import write_json
from .llm_client import LLMClient


def score_quality(
    *,
    client: LLMClient,
    query: str,
    answer: str,
    query_id: str,
) -> dict[str, Any]:
    user = prompts.quality_user_prompt(query, answer)
    parsed = client.complete_json(
        [{"role": "user", "content": user}],
        system=prompts.QUALITY_SYSTEM,
        stage="quality_scoring",
        query_id=query_id,
        input_artifacts=[str(config.GENERATED_ANSWERS_PATH)],
        output_artifact=str(config.ANSWER_QUALITY_SCORES_PATH),
    )
    if not isinstance(parsed, dict):
        parsed = {}
    completeness = _coerce_score(parsed.get("completeness"))
    specificity = _coerce_score(parsed.get("specificity"))
    tone = _coerce_score(parsed.get("tone_appropriateness"))
    operator_required = any(s < 6 for s in (completeness, specificity, tone))
    record = {
        "query_id": query_id,
        "completeness": completeness,
        "specificity": specificity,
        "tone_appropriateness": tone,
        "operator_review_required": operator_required,
    }
    _append(record)
    return record


def _coerce_score(value: Any) -> int:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(10, n))


def _append(record: dict[str, Any]) -> None:
    if config.ANSWER_QUALITY_SCORES_PATH.exists():
        try:
            data = json.loads(config.ANSWER_QUALITY_SCORES_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    data.append(record)
    write_json(config.ANSWER_QUALITY_SCORES_PATH, data)
