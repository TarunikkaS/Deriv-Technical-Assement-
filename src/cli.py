"""Multi-turn interactive CLI.

Maintains a conversation history of (query, answer) pairs. On follow-up
turns the previous turn is folded into the retrieval embedding-query so
references like "the device above" still hit the right chunks.

Each turn flows through the same staged pipeline (confidence gate ->
generation -> verification -> regeneration -> quality) as the batch
runner. After exit, the per-turn audit is appended to answer_audit.json
under conversation_id 'cli'.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import config
from .audit import QueryAudit, write_audit
from .chunker import chunk_and_save
from .embeddings import build_vector_store
from .io_utils import load_sources, load_test_queries
from .llm_client import LLMConfigurationError, make_client
from .pipeline import run_for_query
from .scraper import scrape_and_save


_HISTORY_TURNS = 1  # number of previous turns to fold into retrieval


def _ensure_index_built(force_build: bool, console: Console) -> None:
    """If the vector store is missing (or force_build), run the
    ingestion pipeline so retrieval has something to query."""
    if not force_build and config.EMBEDDINGS_PATH.exists() and config.VECTOR_METADATA_PATH.exists():
        return
    console.print("[dim]Building corpus + vector store (first run can take a minute)...[/dim]")
    sources = load_sources(config.SOURCES_PATH)
    pages = scrape_and_save(sources)
    corpus = chunk_and_save(pages)
    build_vector_store(corpus)


def _format_context(history: list[tuple[str, str]]) -> str:
    """Concatenate the most recent turns into a retrieval-context string."""
    if not history:
        return ""
    recent = history[-_HISTORY_TURNS:]
    pieces: list[str] = []
    for q, a in recent:
        pieces.append(f"Previous Q: {q}")
        pieces.append(f"Previous A: {a}")
    return "\n".join(pieces)


def main() -> int:
    console = Console()
    console.print(Panel.fit(
        "[bold]Deriv Help Centre RAG — interactive CLI[/bold]\n"
        "Type a question and press Enter. Type [cyan]/quit[/cyan] to exit, "
        "[cyan]/reset[/cyan] to clear conversation history, "
        "[cyan]/rebuild[/cyan] to re-scrape and re-embed.",
        border_style="cyan",
    ))

    config.ensure_dirs()
    _ensure_index_built(force_build=False, console=console)

    try:
        client = make_client()
    except LLMConfigurationError as e:
        console.print(f"[red]LLM provider misconfigured:[/red] {e}")
        return 2
    console.print(f"[dim]LLM: {client.provider} ({client.model})[/dim]")

    history: list[tuple[str, str]] = []
    cli_audits: list[QueryAudit] = []
    turn = 0

    while True:
        try:
            user_input = console.input("\n[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if not user_input:
            continue
        if user_input in {"/quit", "/exit"}:
            break
        if user_input == "/reset":
            history.clear()
            console.print("[dim]history cleared[/dim]")
            continue
        if user_input == "/rebuild":
            _ensure_index_built(force_build=True, console=console)
            continue

        turn += 1
        ctx = _format_context(history)
        try:
            audit = run_for_query(
                client=client,
                query=user_input,
                query_id=f"CLI_{turn:03d}",
                conversation_context=ctx or None,
                score_quality_after=False,  # keep CLI snappy; batch run scores
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]error:[/red] {e}")
            continue

        cli_audits.append(audit)
        history.append((user_input, audit.final_response))

        # Render the answer
        if audit.fallback_triggered:
            console.print(Panel(audit.final_response, border_style="yellow", title="fallback"))
        else:
            console.print(Panel(Markdown(audit.final_response), border_style="green", title="answer"))
            top = audit.retrieved_chunks[0] if audit.retrieved_chunks else None
            if top:
                console.print(
                    f"[dim]top retrieval: {top['chunk_id']} "
                    f"(score={top['score']:.2f}) {top['source_url']}[/dim]"
                )

    # Persist CLI audits — note: this overwrites the batch audit. Keep them
    # separate by writing under a sibling path.
    if cli_audits:
        from .io_utils import write_json
        cli_audit_path = config.ARTIFACTS / "cli_session_audit.json"
        write_json(cli_audit_path, [_audit_to_dict(a) for a in cli_audits])
        console.print(f"[dim]session audit -> {cli_audit_path}[/dim]")
    return 0


def _audit_to_dict(a: QueryAudit) -> dict:
    from dataclasses import asdict
    return asdict(a)


if __name__ == "__main__":
    raise SystemExit(main())
