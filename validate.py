"""Validation script for the Deriv RAG pipeline.

Pure on-disk checks — no network, no LLM. Validates that the pipeline has
run correctly and all artifacts satisfy the spec. Exits 0 on full pass, 1
on any failure. Writes a structured artifacts/validation_report.json.

Run: python validate.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total": len(self.checks),
            "passed_count": sum(1 for c in self.checks if c.passed),
            "failed_count": sum(1 for c in self.checks if not c.passed),
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _safe_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _safe_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out
    except (OSError, json.JSONDecodeError):
        return None


def run() -> Report:
    r = Report()

    # --- Input files ---
    sources_raw = _safe_json(config.SOURCES_PATH)
    r.add("sources.json exists", config.SOURCES_PATH.exists())
    r.add("sources.json valid JSON", sources_raw is not None)
    r.add(
        "sources.json has top-level 'sources' key",
        isinstance(sources_raw, dict) and "sources" in (sources_raw or {}),
    )
    sources_list: list[str] = []
    if isinstance(sources_raw, dict) and isinstance(sources_raw.get("sources"), list):
        sources_list = sources_raw["sources"]
    r.add(
        "sources.json 'sources' is non-empty list of strings",
        bool(sources_list) and all(isinstance(s, str) and s.strip() for s in sources_list),
    )

    queries_raw = _safe_json(config.TEST_QUERIES_PATH)
    r.add("test_queries.json exists", config.TEST_QUERIES_PATH.exists())
    r.add("test_queries.json valid JSON", queries_raw is not None)
    r.add(
        "test_queries.json is non-empty list",
        isinstance(queries_raw, list) and len(queries_raw or []) > 0,
    )
    if isinstance(queries_raw, list):
        all_have = all(isinstance(q, dict) and "id" in q and "query" in q for q in queries_raw)
        r.add("every test query has 'id' and 'query'", all_have)

    # --- Required artifacts exist + valid JSON ---
    required_json = [
        ("corpus.json", config.CORPUS_PATH),
        ("vector_store/metadata.json", config.VECTOR_METADATA_PATH),
        ("vector_store/embedding_manifest.json", config.EMBEDDING_MANIFEST_PATH),
        ("generated_answers.json", config.GENERATED_ANSWERS_PATH),
        ("grounding_verification.json", config.GROUNDING_VERIFICATION_PATH),
        ("answer_audit.json", config.ANSWER_AUDIT_PATH),
    ]
    for name, path in required_json:
        r.add(f"artifact exists: {name}", path.exists())
        if path.exists():
            r.add(f"artifact valid JSON: {name}", _safe_json(path) is not None)

    r.add("vector_store/embeddings.npy exists", config.EMBEDDINGS_PATH.exists())
    r.add("llm_calls.jsonl exists", config.LLM_CALLS_PATH.exists())
    r.add("retrieval_logs.jsonl exists", config.RETRIEVAL_LOGS_PATH.exists())

    # --- Cleaned pages: every source ingested or failure logged ---
    cleaned = _safe_json(config.CLEANED_PAGES_PATH) or []
    if sources_list:
        ingested_urls = {p["source_url"] for p in cleaned if isinstance(p, dict)}
        missing_sources = [s for s in sources_list if s not in ingested_urls]
        r.add(
            "every source URL has an ingestion record",
            not missing_sources,
            f"missing: {missing_sources}" if missing_sources else "",
        )
        statuses = {p["source_url"]: p.get("status") for p in cleaned if isinstance(p, dict)}
        bad = [u for u in sources_list if u in statuses and statuses[u] not in {"success", "failed"}]
        r.add("every source has status success|failed", not bad, f"bad: {bad}" if bad else "")

    # --- Corpus: chunk schema + token bounds ---
    corpus = _safe_json(config.CORPUS_PATH) or {}
    chunks = corpus.get("chunks", []) if isinstance(corpus, dict) else []
    required_chunk_keys = {"chunk_id", "source_url", "section_title", "chunk_index", "token_count", "content_hash", "text"}
    schema_ok = all(required_chunk_keys.issubset(c.keys()) for c in chunks if isinstance(c, dict))
    r.add("every chunk has required metadata keys", bool(chunks) and schema_ok)
    if chunks:
        in_range = sum(1 for c in chunks if 200 <= c.get("token_count", 0) <= 350)
        ratio = in_range / max(1, len(chunks))
        r.add(
            "majority of chunk token counts in 200-350",
            ratio >= 0.5,
            f"{in_range}/{len(chunks)} in range",
        )

    # --- Vector store sanity ---
    if config.EMBEDDINGS_PATH.exists() and config.VECTOR_METADATA_PATH.exists():
        try:
            import numpy as np
            arr = np.load(config.EMBEDDINGS_PATH)
            metadata = _safe_json(config.VECTOR_METADATA_PATH) or []
            r.add("embeddings shape matches metadata length", arr.shape[0] == len(metadata),
                  f"{arr.shape[0]} vs {len(metadata)}")
            r.add("embeddings is 2D float array", arr.ndim == 2 and arr.shape[1] > 0)
            r.add("embeddings count matches corpus chunks", arr.shape[0] == len(chunks))
        except Exception as e:  # noqa: BLE001
            r.add("vector store loadable", False, str(e))

    # --- Audit completeness: every test query produced an audit record ---
    audit = _safe_json(config.ANSWER_AUDIT_PATH) or []
    if isinstance(queries_raw, list) and isinstance(audit, list):
        audited_ids = {a.get("query_id") for a in audit if isinstance(a, dict)}
        expected_ids = {q.get("id") for q in queries_raw if isinstance(q, dict)}
        r.add(
            "answer_audit covers every test query",
            audited_ids >= expected_ids,
            f"missing: {expected_ids - audited_ids}",
        )
    if isinstance(audit, list) and audit:
        keys_ok = all(
            {"query_id", "query", "retrieved_chunks", "confidence", "fallback_triggered", "final_response"}.issubset(a.keys())
            for a in audit if isinstance(a, dict)
        )
        r.add("answer_audit records have full trace shape", keys_ok)

        # Every query produced a verified answer OR a documented fallback
        valid_outcomes = 0
        for a in audit:
            if not isinstance(a, dict):
                continue
            if a.get("fallback_triggered") and "Retrieval confidence was" in a.get("final_response", ""):
                valid_outcomes += 1
            elif not a.get("fallback_triggered") and a.get("grounding_verification") is not None:
                valid_outcomes += 1
        r.add(
            "every audited query is verified-answer or documented fallback",
            valid_outcomes == len(audit),
            f"{valid_outcomes}/{len(audit)} valid",
        )

        # Generated answers cite chunk IDs (when not fallback)
        cite_ok = True
        cite_detail = []
        for a in audit:
            if not isinstance(a, dict) or a.get("fallback_triggered"):
                continue
            ans = (a.get("regeneration") or {}).get("regenerated_answer") or a.get("generated_answer", "") or ""
            if "[chunk_id:" not in ans:
                cite_ok = False
                cite_detail.append(a.get("query_id"))
        r.add(
            "generated answers cite chunk IDs",
            cite_ok,
            f"missing citations in: {cite_detail}" if not cite_ok else "",
        )

    # --- LLM call log: separate stage records ---
    llm_calls = _safe_jsonl(config.LLM_CALLS_PATH) or []
    stages_seen = {c.get("stage") for c in llm_calls if isinstance(c, dict)}
    r.add(
        "llm_calls.jsonl present and parses",
        config.LLM_CALLS_PATH.exists() and llm_calls,
    )
    # Generation + verification must always happen for at least one query
    r.add("llm_calls includes 'answer_generation'", "answer_generation" in stages_seen)
    r.add("llm_calls includes 'grounding_verification'", "grounding_verification" in stages_seen)
    # Each LLM call record has the spec-required keys
    if llm_calls:
        record_keys = {"stage", "query_id", "timestamp", "provider", "model", "prompt_hash"}
        all_ok = all(record_keys.issubset(c.keys()) for c in llm_calls if isinstance(c, dict))
        r.add("llm_calls records contain stage/timestamp/provider/model/prompt_hash", all_ok)

    # --- Regeneration: prompt hash differs from generation hash + lists claims ---
    if isinstance(audit, list):
        regen_audits = [a for a in audit if isinstance(a, dict) and a.get("regeneration")]
        # Find generation hashes from llm_calls per query_id
        gen_hashes = {
            c.get("query_id"): c.get("prompt_hash")
            for c in llm_calls
            if isinstance(c, dict) and c.get("stage") == "answer_generation"
        }
        for a in regen_audits:
            qid = a.get("query_id")
            regen_hash = a["regeneration"].get("regeneration_prompt_hash")
            r.add(
                f"regeneration prompt hash differs from generation hash ({qid})",
                regen_hash and gen_hashes.get(qid) and regen_hash != gen_hashes.get(qid),
            )
            ungrounded = a["regeneration"].get("ungrounded_claims") or []
            r.add(
                f"regeneration audit lists ungrounded claims ({qid})",
                isinstance(ungrounded, list) and len(ungrounded) > 0,
            )
            # If regeneration occurred, llm_calls must show second_grounding_verification
            r.add(
                f"second grounding verification logged ({qid})",
                "second_grounding_verification" in stages_seen,
            )

    # --- Confidence gate: every fallback trace has top_score < threshold ---
    if isinstance(audit, list):
        fb_audits = [a for a in audit if isinstance(a, dict) and a.get("fallback_triggered")]
        ok = True
        bad_qids: list[str] = []
        for a in fb_audits:
            score = float((a.get("confidence") or {}).get("top_score", 1.0))
            threshold = float((a.get("confidence") or {}).get("threshold", config.CONFIDENCE_THRESHOLD))
            if score >= threshold:
                ok = False
                bad_qids.append(a.get("query_id", "?"))
        r.add(
            "every fallback was triggered by score below threshold",
            ok,
            f"violations: {bad_qids}" if not ok else "",
        )
        # And no fallback query produced an answer_generation LLM call
        gen_qids = {c.get("query_id") for c in llm_calls if isinstance(c, dict) and c.get("stage") == "answer_generation"}
        leaked = [a.get("query_id") for a in fb_audits if a.get("query_id") in gen_qids]
        r.add(
            "no answer_generation call for fallback queries",
            not leaked,
            f"leaked: {leaked}" if leaked else "",
        )

    # --- CLI multi-turn implementation exists ---
    cli_path = Path("src") / "cli.py"
    cli_src = cli_path.read_text(encoding="utf-8") if cli_path.exists() else ""
    r.add("CLI module exists", cli_path.exists())
    r.add(
        "CLI preserves conversation history",
        "history" in cli_src and "Previous Q:" in cli_src,
    )

    # --- corpus_version_report.json (optional but should attempt) ---
    if config.CORPUS_VERSION_REPORT_PATH.exists():
        cv = _safe_json(config.CORPUS_VERSION_REPORT_PATH)
        if isinstance(cv, dict):
            keys_ok = {"chunks_unchanged", "chunks_updated", "chunks_added", "chunks_removed"}.issubset(cv.keys())
            r.add("corpus_version_report has add/update/unchanged/remove counts", keys_ok)

    return r


def main() -> int:
    config.ensure_dirs()
    report = run()
    config.VALIDATION_REPORT_PATH.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    # Pretty summary
    pad = max((len(c.name) for c in report.checks), default=0)
    for c in report.checks:
        marker = "PASS" if c.passed else "FAIL"
        suffix = f"  -- {c.detail}" if c.detail and not c.passed else ""
        print(f"[{marker}] {c.name.ljust(pad)}{suffix}")
    print()
    if report.passed:
        print(f"All {len(report.checks)} checks passed.")
    else:
        failed = [c for c in report.checks if not c.passed]
        print(f"{len(failed)} of {len(report.checks)} checks FAILED.")
    print(f"Report -> {config.VALIDATION_REPORT_PATH}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
