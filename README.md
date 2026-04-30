# Deriv Help Centre — Replayable RAG Pipeline

A staged, auditable Retrieval-Augmented Generation system over the Deriv Help Centre.

Every retrieval, confidence check, generation step, and grounding verification step is recorded so the full reasoning trace can be replayed and audited.

The pipeline is designed to be **replayable** from a clean checkout. It reads `sources.json` and `test_queries.json` dynamically, regenerates runtime artifacts from the configured inputs, and supports an evaluator replacing either input file before re-running the system.

---

## Branch note: `artifact-updates`

This repository includes an additional branch named `artifact-updates`.

The `main` branch contains the baseline submitted implementation. The `artifact-updates` branch is used to make the artifact trail, generated-output expectations, and validation notes clearer for review.

This branch does not replace the replayable nature of the system. The project is still intended to regenerate all runtime artifacts from `sources.json` and `test_queries.json` whenever the pipeline is executed.

Reviewers may inspect the `artifact-updates` branch to see the latest artifact-related documentation and clarification of expected outputs.

The required generated files remain under the `artifacts/` directory and are produced by running:

```bash
make run
make validate
```

or equivalently:

```bash
python run_pipeline.py
python validate.py
```

The generated artifacts are not meant to be treated as static final answers. They represent the audit trail produced by a run of the pipeline, including scraped content, corpus chunks, cached embeddings, retrieval logs, generated answers, claim-level grounding verification, regeneration details, LLM call logs, and validation output.

---

## What the system does

1. **Loads inputs.**  
   `sources.json` contains the URLs to index, and `test_queries.json` contains the questions to evaluate. Both files are read at runtime and are not intended to be hardcoded into the pipeline.

2. **Scrapes source pages.**  
   Each configured source URL is fetched using `requests` and cleaned using `trafilatura`, with a BeautifulSoup fallback. Scrape failures are recorded instead of being silently ignored.

3. **Preserves raw and cleaned content.**  
   Raw HTML is saved under `artifacts/raw_pages/` where possible, and cleaned ingestion records are saved to `artifacts/cleaned_pages.json`.

4. **Chunks the corpus.**  
   Cleaned content is split into 200–350-token chunks where possible. Each chunk includes metadata such as `chunk_id`, `source_url`, `section_title`, `chunk_index`, `token_count`, `content_hash`, and `text`.

5. **Embeds chunks locally.**  
   Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are stored in a local file-based vector store.

6. **Caches embeddings.**  
   Embeddings are cached by `content_hash`, so unchanged chunks are not re-embedded on later runs.

7. **Retrieves evidence.**  
   For each query, the retriever embeds the query and retrieves the top 5 chunks using cosine similarity.

8. **Checks confidence deterministically.**  
   If the highest similarity score is below `CONFIDENCE_THRESHOLD`, default `0.72`, the system returns a fallback response and does not call the answer-generation LLM.

9. **Generates grounded answers.**  
   If confidence is sufficient, the Stage 1 LLM call generates an answer using only the retrieved chunks and cites chunk IDs in the answer.

10. **Verifies grounding.**  
    A separate Stage 2 LLM call checks the generated answer claim by claim and marks each factual claim as grounded or ungrounded.

11. **Regenerates only when needed.**  
    If any claim is ungrounded, the system performs targeted regeneration. The regeneration prompt explicitly lists the unsupported claims that must be avoided.

12. **Scores answer quality.**  
    An optional Stage 3 LLM call scores the final answer on completeness, specificity, and tone appropriateness.

13. **Exports audit artifacts.**  
    Retrieval evidence, confidence decisions, generated answers, verification results, regeneration details, quality scores, LLM call logs, and final responses are exported under `artifacts/`.

---

## Pipeline stages

```text
INIT
 -> SOURCES_LOADED
 -> CONTENT_SCRAPED
 -> CORPUS_CHUNKED
 -> EMBEDDINGS_CACHED
 -> QUERY_RECEIVED
 -> CHUNKS_RETRIEVED
 -> CONFIDENCE_CHECKED
   -> ANSWER_GENERATED
   -> GROUNDING_VERIFIED
      -> ANSWER_REGENERATED_IF_NEEDED
      -> GROUNDING_VERIFIED  (only when regeneration occurs)
   -> QUALITY_SCORED         (optional)
 -> ANSWER_RETURNED_OR_FALLBACK
 -> AUDIT_EXPORTED
```

The order is enforced in `src/stages.py` through a `Stage` definition and pipeline-state transition logic.

Confidence-fallback queries skip generation, verification, regeneration, and quality scoring because the system has already determined that the indexed corpus does not provide enough evidence for a reliable answer.

---

## Required input files

The project expects two root-level input files.

### `sources.json`

This file controls which Help Centre pages are indexed.

```json
{
  "sources": [
    "https://deriv.com/help-centre/trading/",
    "https://deriv.com/help-centre/deposits-and-withdrawals/",
    "https://deriv.com/help-centre/accounts/",
    "https://deriv.com/help-centre/security/"
  ]
}
```

### `test_queries.json`

This file controls which questions are processed during batch evaluation.

```json
[
  {
    "id": "Q1",
    "query": "How do I reset my two-factor authentication if I've lost access to my authenticator app?"
  },
  {
    "id": "Q2",
    "query": "What is the minimum withdrawal amount for bank transfer?"
  },
  {
    "id": "Q3",
    "query": "Can I have more than one real account on Deriv?"
  },
  {
    "id": "Q4",
    "query": "How long does an e-wallet withdrawal typically take?"
  },
  {
    "id": "Q5",
    "query": "What happens to my open positions if I close my account?"
  },
  {
    "id": "Q6",
    "query": "Is there a fee for depositing via cryptocurrency?"
  },
  {
    "id": "Q7",
    "query": "How do I set trading limits as part of responsible gambling tools?"
  },
  {
    "id": "Q8",
    "query": "What documents are required for account verification in Malaysia?"
  }
]
```

The evaluator may replace either file. The pipeline should therefore work with any valid list of source URLs and any valid list of query objects.

---

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Then edit `.env` as needed.

By default, the pipeline is configured to use **Ollama** for local LLM inference. If using Ollama, pull a model first:

```bash
ollama pull llama3.1:8b
```

The first run may also download the embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## Environment variables

All runtime configuration is read from `.env`.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | LLM provider, for example `ollama`, `anthropic`, `openai`, `gemini`, or `groq` |
| `OLLAMA_BASE_URL` | Local Ollama server URL |
| `OLLAMA_MODEL` | Ollama model name |
| `ANTHROPIC_API_KEY` | Anthropic API key, if using Anthropic |
| `ANTHROPIC_MODEL` | Anthropic model name |
| `OPENAI_API_KEY` | OpenAI API key, if using OpenAI |
| `OPENAI_MODEL` | OpenAI model name |
| `GEMINI_API_KEY` | Gemini API key, if using Gemini |
| `GEMINI_MODEL` | Gemini model name |
| `GROQ_API_KEY` | Groq API key, if using Groq |
| `GROQ_MODEL` | Groq model name |
| `CONFIDENCE_THRESHOLD` | Retrieval confidence threshold, default `0.72` |
| `EMBEDDING_MODEL` | Sentence-transformers embedding model |

If a cloud provider is selected without its required API key, the pipeline should fail with a clear runtime error. Missing cloud keys should not break imports when that provider is not selected.

---

## Running the pipeline

Run the full batch pipeline:

```bash
make run
```

Equivalent command:

```bash
python run_pipeline.py
```

Run validation:

```bash
make validate
```

Equivalent command:

```bash
python validate.py
```

Run the interactive CLI:

```bash
make cli
```

Equivalent command:

```bash
python -m src.cli
```

`make run` performs the full batch workflow:

```text
load sources
scrape pages
clean content
chunk corpus
embed chunks
retrieve evidence
apply confidence fallback
generate grounded answers
verify grounding
regenerate unsupported answers if needed
score answer quality where enabled
export audit artifacts
```

`make validate` inspects the artifacts produced by the most recent run.

---

## Re-running the pipeline

A second `make run` should reuse cached embeddings where possible.

The file `artifacts/corpus_version_report.json` reports how the corpus changed between runs.

Expected fields include:

```json
{
  "chunks_unchanged": 0,
  "chunks_updated": 0,
  "chunks_added": 0,
  "chunks_removed": 0
}
```

---

## Multi-turn CLI behavior

The CLI maintains a conversation history.

For follow-up questions, the previous query and previous answer are folded into the retrieval query so vague follow-ups can still retrieve the right evidence.

Example:

```text
User: How do I reset 2FA?
Assistant: Contact Deriv support to request a reset... [chunk_id: security_0003]

User: What if I lost the device mentioned above?
```

The second query is expanded internally using the previous turn, helping the retriever understand that “device mentioned above” refers to the authenticator app or 2FA device.

---

## Generated artifacts

The pipeline writes runtime outputs under `artifacts/`.

These files are the audit trail of the system. They show what was scraped, how the corpus was chunked, which vectors were cached, what chunks were retrieved for each query, whether confidence fallback was triggered, what the LLM generated, how claims were verified, whether regeneration occurred, and what final response was returned.

The `artifact-updates` branch highlights these generated-output expectations more explicitly. 

| Path | Contents |
|---|---|
| `artifacts/raw_pages/*.html` | Raw scraped HTML from configured sources, saved best-effort for traceability |
| `artifacts/cleaned_pages.json` | Per-source ingestion records, including success or failure status |
| `artifacts/corpus.json` | Chunked corpus with `corpus_version`, source metadata, chunk IDs, token counts, hashes, and text |
| `artifacts/vector_store/embeddings.npy` | Local file-based embedding matrix used for retrieval |
| `artifacts/vector_store/metadata.json` | Metadata linking each embedding row to its chunk |
| `artifacts/vector_store/embedding_manifest.json` | Manifest mapping chunk IDs and content hashes to embedding indexes |
| `artifacts/vector_store/cache.json` | Content-hash-based embedding cache, usually gitignored |
| `artifacts/corpus_version_report.json` | Corpus diff report showing unchanged, updated, added, and removed chunks |
| `artifacts/retrieval_logs.jsonl` | One JSON record per retrieval event, including top-5 chunks and similarity scores |
| `artifacts/generated_answers.json` | Stage-1 grounded answers generated only for queries that pass confidence |
| `artifacts/grounding_verification.json` | Stage-2 claim-level verification records |
| `artifacts/answer_audit.json` | Full per-query audit trail and final returned response |
| `artifacts/llm_calls.jsonl` | One JSON record per LLM call, including stage, model, prompt hash, and artifact paths |
| `artifacts/answer_quality_scores.json` | Optional Stage-3 answer quality scoring |
| `artifacts/knowledge_gap_report.json` | Optional clustering of weak or low-confidence query topics |
| `artifacts/validation_report.json` | Output produced by `validate.py` |

---

## Main audit file

`artifacts/answer_audit.json` is the primary evidence file.

For each query, it is intended to include:

```text
query ID
query text
conversation context used, if any
retrieved chunks
similarity scores
fallback status
fallback reason, if fallback occurred
generated answer, if generation occurred
grounding verification result
regeneration details, if regeneration occurred
quality scores, if attempted
final returned response
```

This file is meant to prove that every final response was either verified against retrieved evidence or replaced by a documented fallback.

---

## LLM call logging

Every LLM call is logged separately in:

```text
artifacts/llm_calls.jsonl
```

Each record includes:

```json
{
  "stage": "ANSWER_GENERATED",
  "query_id": "Q1",
  "timestamp": "ISO-8601 timestamp",
  "provider": "ollama/openai/anthropic/gemini/groq",
  "model": "model name",
  "prompt_hash": "sha256 hash",
  "input_artifacts": ["path"],
  "output_artifact": "path"
}
```

Separate records are expected for:

```text
answer generation
grounding verification
targeted regeneration, if triggered
second grounding verification, if regeneration occurred
answer quality scoring, if attempted
gap detection, if attempted
```

This makes it clear that answer generation and grounding verification are separate model calls.

---

## How deterministic confidence fallback works

The fallback decision is made in code, not by the LLM.

After retrieval, the pipeline reads:

```python
top_score = retrieved_chunks[0]["score"]
```

If:

```python
top_score < CONFIDENCE_THRESHOLD
```

then the system sets `fallback=True`, builds a fallback message, and returns without calling the answer-generation LLM.

Example fallback message:

```text
I could not confidently find this answer in the indexed Deriv Help Centre content.
The closest source was: <url>.
Retrieval confidence was 0.64, below the required threshold of 0.72.
```

The audit records the score and threshold so the decision is reproducible.

The LLM is never asked whether to fall back.

---

## How grounded answer generation works

When retrieval confidence is sufficient, the Stage 1 answer generator receives:

```text
user query
top 5 retrieved chunks
chunk IDs
source URLs
section titles
chunk text
```

The generation prompt instructs the model to:

```text
answer only from the provided chunks
avoid outside knowledge
cite chunk IDs
say when the context does not contain the answer
```

Example citation format:

```text
[chunk_id: security_0003]
```

Generated answers are stored in:

```text
artifacts/generated_answers.json
```

Only queries that pass the confidence threshold should have generated answers.

---

## How grounding verification works

After generation, the system performs a separate Stage 2 verification call.

The verifier receives:

```text
generated answer
retrieved chunks
source metadata
```

The verifier must identify factual claims and return records like:

```json
[
  {
    "claim": "Users should contact support if they lose access to their authenticator app.",
    "grounded": true,
    "supporting_chunk_ids": ["security_0003"],
    "explanation": "The retrieved chunk supports this claim."
  },
  {
    "claim": "The reset takes 24 hours.",
    "grounded": false,
    "supporting_chunk_ids": [],
    "explanation": "ungrounded"
  }
]
```

The verification result is stored in:

```text
artifacts/grounding_verification.json
```

No answer should be treated as final until grounding verification has completed.

---

## How targeted regeneration works

If any verification record contains:

```json
"grounded": false
```

the system regenerates the answer.

The regeneration prompt is different from the original generation prompt and explicitly includes the unsupported claims.

Example regeneration instruction:

```text
The previous answer contained these unsupported claims:
- The reset takes 24 hours.

Regenerate the answer using only the retrieved chunks.
Do not repeat unsupported claims unless they are directly supported.
Cite chunk IDs.
```

The regenerated answer is then verified again.

The audit records:

```text
original answer
ungrounded claims
regeneration prompt hash
regenerated answer
second verification result
final response
```

This avoids blind retrying and makes the repair step auditable.

---

## Answer quality scoring

The optional Stage 3 scorer evaluates the final answer on:

```text
completeness: 0–10
specificity: 0–10
tone appropriateness: 0–10
```

If any score is below 6, the answer is flagged with:

```json
"operator_review_required": true
```

Scores are saved to:

```text
artifacts/answer_quality_scores.json
```

---

## Knowledge gap report

After all test queries are processed, the system can identify weak retrieval areas by inspecting low top-similarity scores and fallback cases.

The knowledge gap report groups weak queries into topics and recommends content improvements.

Output path:

```text
artifacts/knowledge_gap_report.json
```

Expected structure:

```json
[
  {
    "topic": "Account verification requirements",
    "query_ids": ["Q8"],
    "evidence": "The query had weak retrieval confidence.",
    "recommended_content_improvement": "Add clearer Help Centre content about verification documents by country."
  }
]
```

---

## Corpus versioning and embedding reuse

Each chunk has a `content_hash`.

On re-runs, the pipeline compares current chunk hashes with the previous embedding manifest.

Expected behavior:

```text
unchanged chunks reuse cached embeddings
updated chunks receive new embeddings
new chunks are embedded
removed chunks are reported
```

The version report is stored in:

```text
artifacts/corpus_version_report.json
```

Expected fields:

```json
{
  "chunks_unchanged": 0,
  "chunks_updated": 0,
  "chunks_added": 0,
  "chunks_removed": 0
}
```

---

## Synthetic-fixture supplementation

Some Help Centre pages may return limited content through a pure `requests` scrape because modern websites can rely on JavaScript rendering, anti-bot layers, or dynamic page hydration.

The project includes `fixtures/synthetic_help_pages/` as a local fallback mechanism for development and demonstration scenarios where scraping returns insufficient text. When such supplementation is used, synthetic content should be tagged in the corpus and metadata so reviewers can distinguish it from scraped content.

Scrape failures should still be recorded in `artifacts/cleaned_pages.json`.

The intended evaluation behavior remains:

```text
read sources.json
attempt to scrape configured URLs
record successes and failures
build the corpus from available indexed content
return fallback when evidence is weak
```

The system should not rely on hardcoded answers or precomputed final responses.

---

## Limitations and notes

- The system depends on the quality of scraped or indexed Help Centre content. If the configured pages do not contain enough relevant evidence, deterministic fallback should be returned.
- The `0.72` confidence threshold is intentionally strict to reduce hallucinated or weakly grounded answers.
- Local Ollama models may sometimes return JSON inside Markdown fences. The LLM client may need to strip fences or retry parsing.
- The pipeline is designed to produce artifacts dynamically. Generated files under `artifacts/` should be treated as run outputs, not as hand-written final answers.
- Cloud providers require valid API keys when selected through `.env`.
- The evaluator may replace `sources.json` and `test_queries.json`, so the code should not depend on the sample URLs or sample queries.

---

## Validation

`validate.py` is intended to check:

```text
sources.json exists and is valid
test_queries.json exists and is valid
required artifacts exist
JSON artifacts are valid
configured sources were ingested or failures were logged
corpus chunks contain required metadata
embeddings exist in a file-based vector store
each query has either a verified answer or documented fallback
fallback decisions are score-based
generated answers cite chunk IDs
grounding verification is logged as a separate LLM call
ungrounded claims trigger targeted regeneration
regeneration prompts differ from original prompts
answer_audit.json contains retrieval, fallback, verification, and final response data
llm_calls.jsonl contains separate records by stage
```

Run:

```bash
python validate.py
```

or:

```bash
make validate
```

---

## Repository layout

```text
sources.json                                # input: list of URLs to index
test_queries.json                           # input: list of {id, query}
requirements.txt
.env.example
Makefile
run_pipeline.py                             # batch driver
validate.py                                 # artifact validator
fixtures/synthetic_help_pages/              # optional corpus supplementation
src/
  config.py
  stages.py
  io_utils.py
  scraper.py
  cleaner.py
  chunker.py
  embeddings.py
  retriever.py
  confidence.py
  llm_client.py
  prompts.py
  answer_generator.py
  verifier.py
  regenerator.py
  quality_scorer.py
  gap_detector.py
  audit.py
  pipeline.py
  cli.py
artifacts/                                  # generated runtime outputs
```

---

## Design summary

This project implements a guarded RAG workflow:

```text
Retrieve evidence first.
Check confidence deterministically.
Generate only when confidence is sufficient.
Verify every factual claim.
Regenerate when claims are unsupported.
Return fallback when evidence is weak.
Export all intermediate artifacts.
```

The central design principle is that the LLM is not trusted blindly. Retrieval, confidence gating, grounding verification, targeted regeneration, and audit export are separate stages so the final answer can be inspected and reproduced.
