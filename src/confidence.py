"""Deterministic retrieval-confidence gate.

The LLM is NEVER consulted for the fallback decision. If the top-1 cosine
similarity is strictly below CONFIDENCE_THRESHOLD, we suppress generation
and return a fixed-format fallback message.
"""

from __future__ import annotations

from typing import Any

from . import config


def should_fallback(top_score: float, threshold: float = config.CONFIDENCE_THRESHOLD) -> bool:
    return top_score < threshold


def build_fallback_message(
    *,
    top_score: float,
    threshold: float = config.CONFIDENCE_THRESHOLD,
    top_source_url: str | None = None,
) -> str:
    src = top_source_url or "no source available"
    return (
        "I could not confidently find this answer in the indexed Deriv Help Centre "
        f"content. The closest source was: {src}. Retrieval confidence was "
        f"{top_score:.2f}, below the required threshold of {threshold:.2f}."
    )


def evaluate_retrieval(retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    """Inspect retrieval results and return a fallback decision.

    Returns: {top_score, top_source_url, fallback (bool), threshold, message?}
    """
    if not retrieved:
        return {
            "top_score": 0.0,
            "top_source_url": None,
            "fallback": True,
            "threshold": config.CONFIDENCE_THRESHOLD,
            "message": build_fallback_message(top_score=0.0, top_source_url=None),
        }
    top = retrieved[0]
    fallback = should_fallback(top["score"])
    out = {
        "top_score": top["score"],
        "top_source_url": top["source_url"],
        "fallback": fallback,
        "threshold": config.CONFIDENCE_THRESHOLD,
    }
    if fallback:
        out["message"] = build_fallback_message(
            top_score=top["score"], top_source_url=top["source_url"]
        )
    return out
