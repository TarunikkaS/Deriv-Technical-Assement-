"""Stage-2 grounding verification. SEPARATE LLM call from generation.

Returns a list of {claim, grounded, supporting_chunk_ids, explanation}.
"""

from __future__ import annotations

import json
from typing import Any

from . import config, prompts
from .io_utils import write_json
from .llm_client import LLMClient


def verify_answer(
    *,
    client: LLMClient,
    answer: str,
    chunks: list[dict[str, Any]],
    query_id: str,
    second_pass: bool = False,
) -> list[dict[str, Any]]:
    """Run claim-level grounding verification. Always a separate LLM call."""
    user = prompts.verification_user_prompt(answer, chunks)
    stage = "second_grounding_verification" if second_pass else "grounding_verification"
    parsed = client.complete_json(
        [{"role": "user", "content": user}],
        system=prompts.VERIFICATION_SYSTEM,
        stage=stage,
        query_id=query_id,
        input_artifacts=[str(config.CORPUS_PATH), str(config.GENERATED_ANSWERS_PATH)],
        output_artifact=str(config.GROUNDING_VERIFICATION_PATH),
    )
    return _normalize_verification(parsed, chunks)


def _normalize_verification(parsed: Any, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce model output to the spec shape."""
    valid_ids = {c["chunk_id"] for c in chunks}
    # Accept several shapes: a list of claims, a wrapper dict {"claims": [...]},
    # a wrapper dict {"results": [...]}, or a single claim object.
    if isinstance(parsed, dict):
        for key in ("claims", "results", "verifications"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            if "claim" in parsed:
                parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        grounded = bool(item.get("grounded", False))
        supports_raw = item.get("supporting_chunk_ids", []) or []
        if not isinstance(supports_raw, list):
            supports_raw = []
        supports = [str(s) for s in supports_raw if str(s) in valid_ids]
        explanation = str(item.get("explanation", "")).strip()
        if not grounded:
            supports = []
            explanation = "ungrounded"
        out.append(
            {
                "claim": claim,
                "grounded": grounded,
                "supporting_chunk_ids": supports,
                "explanation": explanation,
            }
        )
    return out


def append_verification_record(query_id: str, claims: list[dict[str, Any]], second_pass: bool = False) -> None:
    """Persist to grounding_verification.json (list of per-query records).
    A query can appear twice if regeneration was triggered."""
    if config.GROUNDING_VERIFICATION_PATH.exists():
        try:
            data = json.loads(config.GROUNDING_VERIFICATION_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    data.append({
        "query_id": query_id,
        "second_pass": second_pass,
        "claims": claims,
    })
    write_json(config.GROUNDING_VERIFICATION_PATH, data)


def has_ungrounded(claims: list[dict[str, Any]]) -> bool:
    return any(not c["grounded"] for c in claims)


def list_ungrounded_claim_text(claims: list[dict[str, Any]]) -> list[str]:
    return [c["claim"] for c in claims if not c["grounded"]]
