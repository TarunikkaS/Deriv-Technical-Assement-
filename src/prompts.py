"""All LLM prompts. Kept as functions so chunks/claims interpolate cleanly."""

from __future__ import annotations

from typing import Any

GENERATION_SYSTEM = (
    "You are a Deriv Help Centre assistant. You answer ONLY using the provided context "
    "chunks. Do not use outside knowledge. Cite the chunk_id of every fact you state, "
    "in square brackets like [chunk_id: <id>]. If the context does not contain the "
    "answer, reply exactly with: 'The provided context does not contain the answer.' "
    "Be concise and factual."
)

VERIFICATION_SYSTEM = (
    "You are a strict grounding verifier. Given an answer and the context chunks it "
    "supposedly came from, extract every factual claim from the answer and check "
    "whether each claim is directly supported by the chunks. Return JSON ONLY — no "
    "prose, no markdown fences. The JSON must be a list of objects, each with: "
    "'claim' (string), 'grounded' (true|false), 'supporting_chunk_ids' (array of "
    "chunk_id strings — empty if not grounded), and 'explanation' (string; for "
    "ungrounded claims this must be exactly 'ungrounded')."
)

REGENERATION_SYSTEM = (
    "You are regenerating an answer. The PREVIOUS answer contained UNSUPPORTED claims "
    "that were not found in the provided context chunks. You must NOT repeat those "
    "claims unless they are directly supported by the chunks. Answer ONLY using the "
    "chunks. Cite chunk_id like [chunk_id: <id>]. If the chunks cannot answer the "
    "question, reply exactly with: 'The provided context does not contain the answer.'"
)

QUALITY_SYSTEM = (
    "You are an answer-quality scorer. Score the assistant's answer on three axes "
    "from 0 to 10: completeness (does it fully address the question?), specificity "
    "(does it use concrete facts vs vague generalities?), and tone_appropriateness "
    "(is it professional and helpful?). Return JSON ONLY (no markdown) with keys "
    "completeness, specificity, tone_appropriateness, each an integer 0-10."
)

GAP_SYSTEM = (
    "You analyze a list of low-confidence customer queries and cluster them into "
    "topics that the Help Centre poorly covers. For each topic, give the topic name, "
    "the query_ids it contains, a one-sentence evidence note, and a recommended "
    "content improvement. Return JSON ONLY (no markdown) — a list of objects with "
    "keys: topic, query_ids, evidence, recommended_content_improvement."
)


def render_chunks_block(chunks: list[dict[str, Any]]) -> str:
    """Render chunks for inclusion in a prompt: numbered, with id/source/title."""
    parts: list[str] = []
    for c in chunks:
        parts.append(
            f"--- chunk_id: {c['chunk_id']} | source: {c.get('source_url', '?')} | "
            f"section: {c.get('section_title', '?')} ---\n{c['text']}"
        )
    return "\n\n".join(parts)


def generation_user_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    return (
        f"User question:\n{query}\n\n"
        f"Context chunks (cite by chunk_id):\n{render_chunks_block(chunks)}\n\n"
        "Answer using ONLY the chunks above. Cite chunk_ids in your answer."
    )


def verification_user_prompt(answer: str, chunks: list[dict[str, Any]]) -> str:
    return (
        f"Assistant answer to verify:\n{answer}\n\n"
        f"Context chunks (the only evidence allowed):\n{render_chunks_block(chunks)}\n\n"
        "Extract every factual claim from the answer. Decompose multi-sentence "
        "answers into one claim per fact. For each claim, set grounded=true "
        "ONLY if a chunk directly supports it; otherwise grounded=false with "
        "supporting_chunk_ids=[] and explanation='ungrounded'.\n\n"
        "Return a JSON ARRAY (a list of objects). Even if there is only one "
        "claim, the response must be a list, e.g. "
        '[{"claim": "...", "grounded": true, "supporting_chunk_ids": ["..."], "explanation": ""}].'
    )


def regeneration_user_prompt(
    query: str,
    chunks: list[dict[str, Any]],
    ungrounded_claims: list[str],
    previous_answer: str,
) -> str:
    bullets = "\n".join(f"- {c}" for c in ungrounded_claims) if ungrounded_claims else "- (none listed)"
    return (
        f"User question:\n{query}\n\n"
        f"Previous answer (contained unsupported claims):\n{previous_answer}\n\n"
        f"Unsupported claims to AVOID (do not repeat unless directly supported by chunks):\n"
        f"{bullets}\n\n"
        f"Context chunks:\n{render_chunks_block(chunks)}\n\n"
        "Write a new answer using ONLY the chunks above. Cite chunk_ids. Do not "
        "repeat any of the unsupported claims unless a chunk directly supports them. "
        "If the chunks do not contain the answer, say so."
    )


def quality_user_prompt(query: str, answer: str) -> str:
    return (
        f"Question: {query}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score the answer. Return JSON only with keys completeness, specificity, "
        "tone_appropriateness."
    )


def gap_user_prompt(weak_queries: list[dict[str, Any]]) -> str:
    rendered = "\n".join(
        f"- {q['query_id']} (top_score={q['top_score']:.2f}, fallback={q['fallback']}): {q['query']}"
        for q in weak_queries
    )
    return (
        "Low-confidence queries:\n"
        f"{rendered}\n\n"
        "Cluster them into topics. Return a JSON ARRAY (a list of objects), "
        "even if there is only one topic. Each object must have keys: "
        "topic, query_ids (list), evidence, recommended_content_improvement. "
        'Example: [{"topic": "...", "query_ids": ["Q1","Q2"], "evidence": "...", '
        '"recommended_content_improvement": "..."}].'
    )
