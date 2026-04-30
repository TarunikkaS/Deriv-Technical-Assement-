"""Pipeline stage tracking — enforces ordering for auditability."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Stage(IntEnum):
    INIT = 0
    SOURCES_LOADED = 1
    CONTENT_SCRAPED = 2
    CORPUS_CHUNKED = 3
    EMBEDDINGS_CACHED = 4
    QUERY_RECEIVED = 5
    CHUNKS_RETRIEVED = 6
    CONFIDENCE_CHECKED = 7
    ANSWER_GENERATED = 8
    GROUNDING_VERIFIED = 9
    ANSWER_REGENERATED_IF_NEEDED = 10
    QUALITY_SCORED = 11
    ANSWER_RETURNED_OR_FALLBACK = 12
    AUDIT_EXPORTED = 13


@dataclass
class PipelineState:
    """Tracks current stage. Per-query stages can re-enter from QUERY_RECEIVED."""

    stage: Stage = Stage.INIT
    history: list[Stage] = field(default_factory=list)

    def transition(self, target: Stage) -> None:
        # Allow re-entry to per-query stages or the regen back-edge.
        per_query_restart = target == Stage.QUERY_RECEIVED
        regen_back_edge = (
            self.stage == Stage.ANSWER_REGENERATED_IF_NEEDED
            and target == Stage.GROUNDING_VERIFIED
        )
        if not (per_query_restart or regen_back_edge or target >= self.stage):
            raise RuntimeError(
                f"Illegal stage transition: {self.stage.name} -> {target.name}"
            )
        self.stage = target
        self.history.append(target)
