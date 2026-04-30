"""Embeddings + file-based vector store.

Uses sentence-transformers all-MiniLM-L6-v2 (or whatever EMBEDDING_MODEL is
set to). Vectors are L2-normalized at write-time so retrieval reduces to a
single matmul. A persistent cache keyed by content_hash means unchanged
chunks are not re-embedded across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .io_utils import write_json

_MODEL = None  # lazy-loaded


def _load_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
    return _MODEL


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return v / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts and return an L2-normalized 2D array."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _load_model()
    vecs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    vecs = vecs.astype(np.float32)
    return _normalize(vecs)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def _load_cache() -> dict[str, list[float]]:
    if not config.EMBEDDING_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(config.EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    config.EMBEDDING_CACHE_PATH.write_text(
        json.dumps(cache), encoding="utf-8"
    )


def _load_previous_manifest() -> dict[str, Any]:
    if not config.EMBEDDING_MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(config.EMBEDDING_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_vector_store(corpus: dict[str, Any]) -> dict[str, Any]:
    """Embed all chunks (reusing cache by content_hash) and persist the store.

    Writes:
      vector_store/embeddings.npy
      vector_store/metadata.json   (list of {chunk_id, source_url, ...})
      vector_store/embedding_manifest.json
      corpus_version_report.json   (added/updated/unchanged/removed counts)
      vector_store/cache.json      (content_hash -> list[float])
    """
    config.ensure_dirs()
    chunks = corpus.get("chunks", [])
    if not chunks:
        # Persist an empty store so downstream code has stable inputs.
        np.save(config.EMBEDDINGS_PATH, np.zeros((0, 384), dtype=np.float32))
        write_json(config.VECTOR_METADATA_PATH, [])
        write_json(
            config.EMBEDDING_MANIFEST_PATH,
            {
                "corpus_version": corpus.get("corpus_version"),
                "embedding_model": config.EMBEDDING_MODEL,
                "chunk_id_to_index": {},
                "chunk_id_to_hash": {},
            },
        )
        write_json(
            config.CORPUS_VERSION_REPORT_PATH,
            {"chunks_unchanged": 0, "chunks_updated": 0, "chunks_added": 0, "chunks_removed": 0},
        )
        return {"embeddings": np.zeros((0, 384), dtype=np.float32), "metadata": []}

    cache = _load_cache()
    prev_manifest = _load_previous_manifest()
    prev_chunk_to_hash: dict[str, str] = prev_manifest.get("chunk_id_to_hash", {})

    # Identify cache hits/misses. We embed the section title alongside the
    # chunk text so the heading contributes to similarity — a common practice
    # that helps short queries match topical sections.
    misses_idx: list[int] = []
    misses_text: list[str] = []
    for i, c in enumerate(chunks):
        if c["content_hash"] not in cache:
            misses_idx.append(i)
            heading = c.get("section_title") or ""
            embed_input = f"{heading}\n{c['text']}" if heading else c["text"]
            misses_text.append(embed_input)

    # Compute missing embeddings
    if misses_text:
        new_vecs = embed_texts(misses_text)
        for idx, vec in zip(misses_idx, new_vecs):
            cache[chunks[idx]["content_hash"]] = vec.tolist()

    # Assemble final embeddings array in chunk order
    dim = len(next(iter(cache.values()))) if cache else 384
    arr = np.zeros((len(chunks), dim), dtype=np.float32)
    metadata: list[dict[str, Any]] = []
    chunk_id_to_index: dict[str, int] = {}
    chunk_id_to_hash: dict[str, str] = {}
    for i, c in enumerate(chunks):
        vec = cache[c["content_hash"]]
        arr[i] = np.asarray(vec, dtype=np.float32)
        metadata.append(
            {
                "chunk_id": c["chunk_id"],
                "source_url": c["source_url"],
                "section_title": c["section_title"],
                "token_count": c["token_count"],
                "content_hash": c["content_hash"],
                "source_type": c.get("source_type", "scraped"),
                "text": c["text"],
            }
        )
        chunk_id_to_index[c["chunk_id"]] = i
        chunk_id_to_hash[c["chunk_id"]] = c["content_hash"]

    # Write everything
    np.save(config.EMBEDDINGS_PATH, arr)
    write_json(config.VECTOR_METADATA_PATH, metadata)
    manifest = {
        "corpus_version": corpus.get("corpus_version"),
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dim": int(arr.shape[1]),
        "chunk_id_to_index": chunk_id_to_index,
        "chunk_id_to_hash": chunk_id_to_hash,
    }
    write_json(config.EMBEDDING_MANIFEST_PATH, manifest)

    # Diff stats
    prev_ids = set(prev_chunk_to_hash.keys())
    cur_ids = set(chunk_id_to_hash.keys())
    added = cur_ids - prev_ids
    removed = prev_ids - cur_ids
    updated = {
        cid for cid in cur_ids & prev_ids
        if prev_chunk_to_hash[cid] != chunk_id_to_hash[cid]
    }
    unchanged = (cur_ids & prev_ids) - updated
    write_json(
        config.CORPUS_VERSION_REPORT_PATH,
        {
            "chunks_unchanged": len(unchanged),
            "chunks_updated": len(updated),
            "chunks_added": len(added),
            "chunks_removed": len(removed),
            "embedding_model": config.EMBEDDING_MODEL,
            "corpus_version": corpus.get("corpus_version"),
        },
    )
    # Prune cache: drop hashes that aren't referenced by any current chunk
    referenced = {c["content_hash"] for c in chunks}
    pruned_cache = {h: v for h, v in cache.items() if h in referenced}
    _save_cache(pruned_cache)

    print(
        f"[embeddings] {arr.shape[0]} chunks; cache misses={len(misses_idx)}; "
        f"unchanged={len(unchanged)} updated={len(updated)} added={len(added)} removed={len(removed)}"
    )
    return {"embeddings": arr, "metadata": metadata}


def load_vector_store() -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load the persisted vector store from disk."""
    if not config.EMBEDDINGS_PATH.exists() or not config.VECTOR_METADATA_PATH.exists():
        raise FileNotFoundError("Vector store not found — run the pipeline first.")
    arr = np.load(config.EMBEDDINGS_PATH)
    metadata = json.loads(config.VECTOR_METADATA_PATH.read_text(encoding="utf-8"))
    return arr, metadata
