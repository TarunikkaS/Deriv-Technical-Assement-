"""Top-k retrieval against the persisted vector store. Uses cosine
similarity (dot product on normalized vectors). Logs each retrieval event."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config
from .embeddings import embed_query, load_vector_store
from .io_utils import append_jsonl


def retrieve(
    query: str,
    *,
    query_id: str | None = None,
    conversation_context: str | None = None,
    top_k: int = config.TOP_K,
) -> list[dict[str, Any]]:
    """Retrieve top_k chunks. Returns a list of {rank, chunk_id, score,
    source_url, section_title, text} sorted by score descending."""
    arr, metadata = load_vector_store()
    if arr.shape[0] == 0:
        _log(query, query_id, conversation_context, [])
        return []
    # Embedding query: combine conversation context if any
    embed_text = query if not conversation_context else f"{conversation_context}\n{query}"
    q = embed_query(embed_text)  # already normalized
    sims = arr @ q  # cosine since both sides are unit-normalized
    k = min(top_k, sims.shape[0])
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_idx, start=1):
        meta = metadata[int(idx)]
        results.append(
            {
                "rank": rank,
                "chunk_id": meta["chunk_id"],
                "score": float(sims[int(idx)]),
                "source_url": meta["source_url"],
                "section_title": meta["section_title"],
                "text": meta["text"],
            }
        )
    _log(query, query_id, conversation_context, results)
    return results


def _log(
    query: str,
    query_id: str | None,
    conversation_context: str | None,
    results: list[dict[str, Any]],
) -> None:
    record = {
        "query_id": query_id,
        "query": query,
        "conversation_context_used": conversation_context or "",
        "retrieved_chunks": results,
    }
    append_jsonl(config.RETRIEVAL_LOGS_PATH, record)
