"""Stage-1 grounded answer generation. Only invoked after the confidence
gate has passed."""

from __future__ import annotations

import json
from typing import Any

from . import config, prompts
from .io_utils import write_json
from .llm_client import LLMClient


def generate_answer(
    *,
    client: LLMClient,
    query: str,
    query_id: str,
    chunks: list[dict[str, Any]],
) -> str:
    """Stage 1: ask the LLM to answer using only the provided chunks."""
    user = prompts.generation_user_prompt(query, chunks)
    text = client.complete(
        [{"role": "user", "content": user}],
        system=prompts.GENERATION_SYSTEM,
        stage="answer_generation",
        query_id=query_id,
        input_artifacts=[str(config.CORPUS_PATH), str(config.RETRIEVAL_LOGS_PATH)],
        output_artifact=str(config.GENERATED_ANSWERS_PATH),
    )
    return text.strip()


def append_generated_answer(query_id: str, query: str, answer: str) -> None:
    """Persist generated answers to artifacts/generated_answers.json
    (a list, replaced on each run, but appended within one run)."""
    if config.GENERATED_ANSWERS_PATH.exists():
        try:
            data = json.loads(config.GENERATED_ANSWERS_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    data.append({"query_id": query_id, "query": query, "answer": answer})
    write_json(config.GENERATED_ANSWERS_PATH, data)
