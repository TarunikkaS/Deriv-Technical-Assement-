"""Pipeline orchestrator. One entry point for both batch (run_pipeline.py)
and CLI (src.cli) callers.

run_for_query() executes the full per-query stage sequence:
  QUERY_RECEIVED -> CHUNKS_RETRIEVED -> CONFIDENCE_CHECKED ->
  (ANSWER_GENERATED -> GROUNDING_VERIFIED -> [ANSWER_REGENERATED_IF_NEEDED ->
   GROUNDING_VERIFIED]) | (fallback) ->
  QUALITY_SCORED (optional) -> ANSWER_RETURNED_OR_FALLBACK

Returns a QueryAudit record so callers can assemble answer_audit.json.
"""

from __future__ import annotations

from typing import Any

from . import config
from .answer_generator import append_generated_answer, generate_answer
from .audit import QueryAudit
from .confidence import evaluate_retrieval
from .llm_client import LLMClient
from .quality_scorer import score_quality
from .regenerator import regenerate
from .retriever import retrieve
from .stages import PipelineState, Stage
from .verifier import (
    append_verification_record,
    has_ungrounded,
    list_ungrounded_claim_text,
    verify_answer,
)


def run_for_query(
    *,
    client: LLMClient,
    query: str,
    query_id: str,
    conversation_context: str | None = None,
    score_quality_after: bool = True,
) -> QueryAudit:
    state = PipelineState()
    state.transition(Stage.QUERY_RECEIVED)

    audit = QueryAudit(
        query_id=query_id,
        query=query,
        conversation_context_used=conversation_context or "",
    )

    retrieved = retrieve(
        query,
        query_id=query_id,
        conversation_context=conversation_context,
    )
    audit.retrieved_chunks = retrieved
    state.transition(Stage.CHUNKS_RETRIEVED)

    decision = evaluate_retrieval(retrieved)
    audit.confidence = decision
    state.transition(Stage.CONFIDENCE_CHECKED)

    if decision["fallback"]:
        # Deterministic fallback. NO answer-generation LLM call.
        audit.fallback_triggered = True
        audit.final_response = decision["message"]
        state.transition(Stage.ANSWER_RETURNED_OR_FALLBACK)
        audit.stages = [s.name for s in state.history]
        return audit

    # Stage 1: grounded generation
    answer = generate_answer(
        client=client, query=query, query_id=query_id, chunks=retrieved
    )
    append_generated_answer(query_id, query, answer)
    audit.generated_answer = answer
    state.transition(Stage.ANSWER_GENERATED)

    # Stage 2: grounding verification (separate LLM call)
    claims = verify_answer(
        client=client, answer=answer, chunks=retrieved, query_id=query_id
    )
    append_verification_record(query_id, claims, second_pass=False)
    audit.grounding_verification = claims
    state.transition(Stage.GROUNDING_VERIFIED)

    final_answer = answer

    if has_ungrounded(claims):
        ungrounded = list_ungrounded_claim_text(claims)
        regen = regenerate(
            client=client,
            query=query,
            query_id=query_id,
            chunks=retrieved,
            previous_answer=answer,
            ungrounded_claims=ungrounded,
        )
        state.transition(Stage.ANSWER_REGENERATED_IF_NEEDED)

        # Verify the regenerated answer (separate LLM call)
        second_claims = verify_answer(
            client=client,
            answer=regen["regenerated_answer"],
            chunks=retrieved,
            query_id=query_id,
            second_pass=True,
        )
        append_verification_record(query_id, second_claims, second_pass=True)
        state.transition(Stage.GROUNDING_VERIFIED)

        audit.regeneration = {
            "original_answer": answer,
            "ungrounded_claims": ungrounded,
            "regeneration_prompt_hash": regen["regeneration_prompt_hash"],
            "regenerated_answer": regen["regenerated_answer"],
            "second_verification_result": second_claims,
        }
        final_answer = regen["regenerated_answer"]

    if score_quality_after:
        try:
            scores = score_quality(
                client=client, query=query, answer=final_answer, query_id=query_id
            )
            audit.quality_scores = scores
            state.transition(Stage.QUALITY_SCORED)
        except Exception as e:  # noqa: BLE001 - quality scoring is optional
            audit.quality_scores = {"error": str(e)}

    audit.final_response = final_answer
    state.transition(Stage.ANSWER_RETURNED_OR_FALLBACK)
    audit.stages = [s.name for s in state.history]
    return audit
