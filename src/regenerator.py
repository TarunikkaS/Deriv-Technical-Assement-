"""Targeted regeneration. Only fires when the verifier finds at least one
ungrounded claim. The regeneration prompt explicitly enumerates the
unsupported claims so the model is steered away from repeating them.

The full regeneration audit (original answer, ungrounded claims, prompt
hash, regenerated answer, second verification) is returned so callers can
fold it into answer_audit.json.
"""

from __future__ import annotations

import hashlib
from typing import Any

from . import config, prompts
from .llm_client import LLMClient


def regenerate(
    *,
    client: LLMClient,
    query: str,
    query_id: str,
    chunks: list[dict[str, Any]],
    previous_answer: str,
    ungrounded_claims: list[str],
) -> dict[str, Any]:
    """Make a Stage-2 regeneration call (separate from the original
    generation). Returns the regenerated answer and the prompt hash."""
    user = prompts.regeneration_user_prompt(
        query=query,
        chunks=chunks,
        ungrounded_claims=ungrounded_claims,
        previous_answer=previous_answer,
    )
    # Hash the rendered regeneration prompt (system + user) so the audit can
    # prove it differs from the original generation prompt.
    prompt_hash = hashlib.sha256(
        (prompts.REGENERATION_SYSTEM + "\n" + user).encode("utf-8")
    ).hexdigest()

    text = client.complete(
        [{"role": "user", "content": user}],
        system=prompts.REGENERATION_SYSTEM,
        stage="regeneration",
        query_id=query_id,
        input_artifacts=[str(config.GENERATED_ANSWERS_PATH), str(config.GROUNDING_VERIFICATION_PATH)],
        output_artifact=str(config.ANSWER_AUDIT_PATH),
    )
    return {
        "regenerated_answer": text.strip(),
        "regeneration_prompt_hash": prompt_hash,
    }
