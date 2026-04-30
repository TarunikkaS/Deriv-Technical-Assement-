# Deriv Help Centre — Replayable RAG Pipeline

A staged, auditable Retrieval-Augmented Generation system over the Deriv Help Centre.
Every retrieval → confidence-check → generation → grounding-verification step
is recorded so the full reasoning trace can be replayed and audited.

The pipeline is designed to be **replayable** from a clean checkout: it
reads `sources.json` and `test_queries.json` dynamically, regenerates every
artifact from the configured inputs, and supports an evaluator swapping
either input file and re-running.

## Branches to review

This repository currently has two useful branches:

- `main` contains the complete replayable RAG pipeline plus the generated
  batch artifacts from the main assessment run.
- `artifact-updates` contains the latest follow-up updates, including the
  refreshed artifacts and `artifacts/cli_session_audit.json`.

Please check both branches when reviewing the submission: start with `main`
for the baseline implementation, then compare `artifact-updates` for the
newest artifact and prompt/verification updates.

---

## What the system does

1. **Loads inputs.** `sources.json` (URLs to index) and `test_queries.json`
   (questions to evaluate) are read at runtime — never inlined in code.
2. **Scrapes** each source URL with `requests` + `trafilatura` (BeautifulSoup
   fallback). Failures are *recorded*, not raised. Raw HTML is saved under
   `artifacts/raw_pages/`.
3. **Cleans + chunks** the corpus into 200–350-token chunks, preserving
   headings, with each chunk identified by a `content_hash`.
4. **Embeds** chunks locally using `sentence-transformers/all-MiniLM-L6-v2`,
   L2-normalized at write-time. Embeddings are cached by content hash so
   unchanged chunks are never re-embedded.
5. **Retrieves** the top 5 chunks by cosine similarity for each query.
6. **Gates** answer generation deterministically: if the top similarity
   score is below `CONFIDENCE_THRESHOLD` (default 0.72), the pipeline emits
   a fixed-format fallback message and **does not call the LLM**.
7. **Generates** a grounded answer (Stage 1) using only the retrieved
   chunks, with `[chunk_id: …]` citations.
8. **Verifies** the answer claim-by-claim (Stage 2) — a *separate* LLM call
   that classifies each factual claim as `grounded` or `ungrounded`.
9. **Regenerates** (Stage 2.5) — only when the verifier finds at least one
   ungrounded claim. The regeneration prompt **explicitly enumerates the
   unsupported claims** the model must avoid. The regenerated answer is
   then re-verified.
10. **Scores** the final answer (Stage 3, optional) on completeness,
    specificity, and tone (0-10 each); flags `operator_review_required`
    when any score drops below 6.
11. **Audits** every query: retrieval scores, confidence decision, generated
    answer, verification, regeneration details, quality scores, final
    response. Plus a `knowledge_gap_report.json` clustering low-confidence
    queries into Help Centre improvement topics.

## Pipeline stages

```
INIT
 -> SOURCES_LOADED -> CONTENT_SCRAPED -> CORPUS_CHUNKED -> EMBEDDINGS_CACHED
 -> QUERY_RECEIVED -> CHUNKS_RETRIEVED -> CONFIDENCE_CHECKED
   -> ANSWER_GENERATED -> GROUNDING_VERIFIED
      -> ANSWER_REGENERATED_IF_NEEDED -> GROUNDING_VERIFIED  (only when ungrounded)
   -> QUALITY_SCORED   (optional)
 -> ANSWER_RETURNED_OR_FALLBACK -> AUDIT_EXPORTED
```

The order is enforced in `src/stages.py` via a `Stage(IntEnum)` and a
`PipelineState.transition()` method. Confidence-fallback queries skip
generation, verification, regeneration, and quality scoring entirely.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit keys / model names as needed
```

By default the pipeline uses **Ollama** for local, free LLM inference. Pull
a model first:

```bash
# install Ollama from https://ollama.com if you don't have it
ollama pull llama3.1:8b
```

The first pipeline run will also download the
`sentence-transformers/all-MiniLM-L6-v2` embedding model (~90 MB) into
`~/.cache/huggingface/`.

### Environment variables

All variables are read from `.env`. See `.env.example` for the full list.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `ollama` (default) \| `anthropic` \| `openai` \| `gemini` \| `groq` |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Local Ollama daemon URL + model |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Optional cloud provider |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Optional cloud provider |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Optional cloud provider |
| `GROQ_API_KEY` / `GROQ_MODEL` | Optional cloud provider |
| `CONFIDENCE_THRESHOLD` | Fallback trigger (default 0.72) |
| `EMBEDDING_MODEL` | Sentence-transformers model name |

If a cloud provider is selected without its API key, the pipeline fails
with a clear error at runtime. Imports never fail because of missing keys.

---

## Running the pipeline

```bash
make run        # python run_pipeline.py
make validate   # python validate.py     -> all on-disk checks, exits 0/1
make cli        # python -m src.cli      -> interactive multi-turn shell
```

`make run` performs the full batch: scrape → chunk → embed → answer all
test queries → write the audit. `make validate` is read-only and
fast — it inspects the artifacts produced by the most recent run.

The CLI maintains a conversation history and on each follow-up turn folds
the previous `(query, answer)` pair into the embedding query so questions
like "What if I lost the device above?" still retrieve the right chunks.

### Re-running

A second `make run` will re-use cached embeddings — `corpus_version_report.json`
will show `chunks_unchanged > 0` and `cache.json` will not regrow. Edit a
chunk's source page (or `sources.json`) and re-run; the report will reflect
`chunks_added`, `chunks_updated`, or `chunks_removed`.

---

## Generated artifacts

| Path | Contents |
|---|---|
| `artifacts/raw_pages/*.html` | Raw scraped HTML (best-effort) |
| `artifacts/cleaned_pages.json` | Per-source ingestion records (success or failed) |
| `artifacts/corpus.json` | Chunked corpus with `corpus_version` timestamp |
| `artifacts/vector_store/embeddings.npy` | Normalized embedding matrix |
| `artifacts/vector_store/metadata.json` | Per-row chunk metadata |
| `artifacts/vector_store/embedding_manifest.json` | `chunk_id → {index, hash}` |
| `artifacts/vector_store/cache.json` | `content_hash → vector` cache (gitignored) |
| `artifacts/corpus_version_report.json` | Diff stats vs the previous run |
| `artifacts/retrieval_logs.jsonl` | One record per retrieval event |
| `artifacts/generated_answers.json` | Raw Stage-1 answers with citations |
| `artifacts/grounding_verification.json` | Claim-by-claim verification |
| `artifacts/answer_audit.json` | Full per-query trace |
| `artifacts/llm_calls.jsonl` | One record per LLM call (stage, hash, …) |
| `artifacts/answer_quality_scores.json` | Stage-3 quality scoring (optional) |
| `artifacts/knowledge_gap_report.json` | Topic clustering of weak queries |
| `artifacts/validation_report.json` | Output of `validate.py` |

---

## How the deterministic confidence fallback works

The fallback decision is made in pure Python in `src/confidence.py`. After
retrieval we read `top_score = retrieved[0]["score"]`. If
`top_score < CONFIDENCE_THRESHOLD`, we set `fallback=True`, build the
fallback message:

```
I could not confidently find this answer in the indexed Deriv Help Centre
content. The closest source was: <url>. Retrieval confidence was 0.64,
below the required threshold of 0.72.
```

…and **return immediately** — no `answer_generation` LLM call is made for
that query. The audit records this with `fallback_triggered: true`,
preserving the threshold and score so the decision is fully reproducible.

The LLM is *never* asked whether to fall back — that would defeat the
purpose. `validate.py` enforces this by verifying that no
`answer_generation` log entry exists for any fallback query and that
`top_score < threshold` for every fallback case.

## How grounding verification works

After generation, `src/verifier.py` makes a **separate** LLM call (logged
as `grounding_verification` in `llm_calls.jsonl`) that asks the model to:

1. enumerate every factual claim in the generated answer,
2. mark each claim `grounded: true|false`,
3. list `supporting_chunk_ids` for grounded claims,
4. set `explanation: "ungrounded"` for any claim it cannot trace.

The verifier returns JSON only; the client strips Markdown fences and
retries once on parse failure (logged as a separate stage). The output is
appended to `artifacts/grounding_verification.json`.

## How regeneration works

If the verifier reports any `grounded=false` claim, `src/regenerator.py`
makes a **third** LLM call (logged as `regeneration`). The regeneration
prompt is structurally distinct from the original generation prompt — it
includes:

* the previous answer,
* a bullet list of the unsupported claims to avoid,
* a fresh instruction to answer only from the chunks.

The regeneration system + user prompt is SHA256-hashed and stored as
`regeneration_prompt_hash` in the audit, so an evaluator can verify it
differs from the original. The regenerated answer is then re-verified
(`second_grounding_verification`) and its result is recorded.

`validate.py` enforces that whenever a regeneration occurred the prompt
hash differs from the original `answer_generation` prompt hash and that
the audit lists the ungrounded claims that triggered it.

---

## Synthetic-fixture supplementation

Real `deriv.com/help-centre` pages are partially JavaScript-rendered behind
Cloudflare. A pure-`requests` scrape can return very thin content. To
guarantee the pipeline always has a meaningful corpus to retrieve against,
the scraper supplements scraped pages with hand-written markdown fixtures
in `fixtures/synthetic_help_pages/` *only when the total scraped tokens
fall below ~5 000*. Each synthetic chunk is tagged `source_type: synthetic`
in the corpus and metadata so a reviewer can distinguish them. Failed
scrapes still appear in `cleaned_pages.json` as `status: failed` so the
ingestion record is complete.

---

## Limitations and notes

* The synthetic fixtures cover the topics asked by the sample test
  queries (2FA reset, withdrawals, account types, etc.). A new
  `test_queries.json` may ask about topics not in either the scrape or
  the fixtures, in which case the deterministic fallback correctly fires.
* The 0.72 confidence threshold is intentionally tight. With this corpus
  + MiniLM embeddings, several test queries deliberately exercise the
  fallback path. This is by design — it demonstrates the gate works.
* The Ollama model (`llama3.1:8b` default) sometimes emits Markdown fences
  around JSON. The client strips fences and retries once.
* Bonus providers `gemini` and `groq` are included beyond the spec's
  required Ollama / Anthropic / OpenAI because they're useful drop-in
  alternatives. Configure them by setting `LLM_PROVIDER` and the
  corresponding API key in `.env`.

---

## Repository layout

```
sources.json                                # input — list of URLs to index
test_queries.json                           # input — list of {id, query}
requirements.txt
.env.example
Makefile
run_pipeline.py                             # batch driver
validate.py                                 # on-disk validator
fixtures/synthetic_help_pages/              # corpus supplementation
src/
  config.py            stages.py            io_utils.py
  scraper.py           cleaner.py           chunker.py
  embeddings.py        retriever.py         confidence.py
  llm_client.py        prompts.py
  answer_generator.py  verifier.py          regenerator.py
  quality_scorer.py    gap_detector.py
  audit.py             pipeline.py          cli.py
artifacts/                                  # all generated outputs
```
