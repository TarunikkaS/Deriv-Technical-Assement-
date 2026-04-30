"""Centralized configuration: paths, env vars, defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# Input files (read at runtime, never inlined)
SOURCES_PATH = ROOT / "sources.json"
TEST_QUERIES_PATH = ROOT / "test_queries.json"

# Artifact paths
ARTIFACTS = ROOT / "artifacts"
RAW_PAGES_DIR = ARTIFACTS / "raw_pages"
CLEANED_PAGES_PATH = ARTIFACTS / "cleaned_pages.json"
CORPUS_PATH = ARTIFACTS / "corpus.json"
VECTOR_STORE_DIR = ARTIFACTS / "vector_store"
EMBEDDINGS_PATH = VECTOR_STORE_DIR / "embeddings.npy"
VECTOR_METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
EMBEDDING_MANIFEST_PATH = VECTOR_STORE_DIR / "embedding_manifest.json"
EMBEDDING_CACHE_PATH = VECTOR_STORE_DIR / "cache.json"
CORPUS_VERSION_REPORT_PATH = ARTIFACTS / "corpus_version_report.json"

RETRIEVAL_LOGS_PATH = ARTIFACTS / "retrieval_logs.jsonl"
GENERATED_ANSWERS_PATH = ARTIFACTS / "generated_answers.json"
GROUNDING_VERIFICATION_PATH = ARTIFACTS / "grounding_verification.json"
ANSWER_AUDIT_PATH = ARTIFACTS / "answer_audit.json"
LLM_CALLS_PATH = ARTIFACTS / "llm_calls.jsonl"
ANSWER_QUALITY_SCORES_PATH = ARTIFACTS / "answer_quality_scores.json"
KNOWLEDGE_GAP_REPORT_PATH = ARTIFACTS / "knowledge_gap_report.json"
VALIDATION_REPORT_PATH = ARTIFACTS / "validation_report.json"

# Synthetic content fallback (used only when scraped corpus is too thin)
SYNTHETIC_FIXTURES_DIR = ROOT / "fixtures" / "synthetic_help_pages"
SYNTHETIC_TOKEN_FLOOR = 5_000  # supplement scrape if total tokens fall below this

# Retrieval / generation knobs
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.72"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = 5
CHUNK_TOKEN_TARGET = 275
CHUNK_TOKEN_MIN = 200
CHUNK_TOKEN_MAX = 350

# LLM provider config
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def ensure_dirs() -> None:
    """Create writable directories the pipeline needs."""
    for d in (ARTIFACTS, RAW_PAGES_DIR, VECTOR_STORE_DIR):
        d.mkdir(parents=True, exist_ok=True)
