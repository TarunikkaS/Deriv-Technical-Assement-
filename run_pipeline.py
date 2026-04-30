"""Batch pipeline runner.

  1. Loads sources.json and test_queries.json (creates samples if missing).
  2. Scrapes pages -> cleaned_pages.json (supplements with synthetic
     fixtures if scraping yields too little).
  3. Chunks -> corpus.json.
  4. Embeds (with content_hash cache) -> vector_store/.
  5. For each test query: retrieve -> confidence-check -> (generate ->
     verify -> regenerate-if-needed) | fallback -> quality-score.
  6. Emits knowledge_gap_report.json over the weakest queries.
  7. Writes answer_audit.json with the full per-query trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src import config
from src.audit import QueryAudit, write_audit
from src.chunker import chunk_and_save
from src.embeddings import build_vector_store
from src.gap_detector import detect_gaps
from src.io_utils import load_sources, load_test_queries
from src.llm_client import LLMConfigurationError, make_client
from src.pipeline import run_for_query
from src.scraper import scrape_and_save
from src.stages import PipelineState, Stage


def _reset_per_run_artifacts() -> None:
    """Clear append-style artifacts from prior runs so a fresh batch starts
    cleanly without inheriting old records."""
    for p in (
        config.LLM_CALLS_PATH,
        config.RETRIEVAL_LOGS_PATH,
        config.GENERATED_ANSWERS_PATH,
        config.GROUNDING_VERIFICATION_PATH,
        config.ANSWER_AUDIT_PATH,
        config.ANSWER_QUALITY_SCORES_PATH,
        config.KNOWLEDGE_GAP_REPORT_PATH,
    ):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> int:
    config.ensure_dirs()
    _reset_per_run_artifacts()

    state = PipelineState()
    state.transition(Stage.INIT)

    sources = load_sources(config.SOURCES_PATH)
    queries = load_test_queries(config.TEST_QUERIES_PATH)
    state.transition(Stage.SOURCES_LOADED)
    print(f"[run] {len(sources)} sources, {len(queries)} test queries")

    # 1. Scrape (or fall back to fixtures)
    pages = scrape_and_save(sources)
    state.transition(Stage.CONTENT_SCRAPED)

    # 2. Chunk
    corpus = chunk_and_save(pages)
    state.transition(Stage.CORPUS_CHUNKED)

    # 3. Embed
    build_vector_store(corpus)
    state.transition(Stage.EMBEDDINGS_CACHED)

    # 4. Build LLM client (fail fast if provider misconfigured)
    try:
        client = make_client()
    except LLMConfigurationError as e:
        print(f"[run] LLM provider misconfigured: {e}", file=sys.stderr)
        return 2

    print(f"[run] LLM provider: {client.provider} ({client.model})")

    audits: list[QueryAudit] = []
    for q in queries:
        print(f"[run] === {q['id']}: {q['query'][:80]}")
        try:
            audit = run_for_query(
                client=client,
                query=q["query"],
                query_id=q["id"],
                conversation_context=None,
            )
        except Exception as e:  # noqa: BLE001 - keep batch robust
            print(f"[run] {q['id']} failed: {e}")
            audit = QueryAudit(
                query_id=q["id"],
                query=q["query"],
                final_response=f"ERROR: {e}",
                stages=["INIT_ERROR"],
            )
        audits.append(audit)

    write_audit(audits)
    state.transition(Stage.AUDIT_EXPORTED)

    # 5. Gap detection over weakest queries
    try:
        detect_gaps(client=client, audits=audits)
    except Exception as e:  # noqa: BLE001 - optional stage
        print(f"[run] gap detection skipped: {e}")

    print(f"[run] Done. Audit -> {config.ANSWER_AUDIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
