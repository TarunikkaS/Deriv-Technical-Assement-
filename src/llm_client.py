"""Provider-agnostic LLM client.

Every call appends a record to artifacts/llm_calls.jsonl. The factory
``make_client()`` selects a provider via the LLM_PROVIDER env var. Provider
SDKs are imported lazily so missing API keys never break import.

Stages call ``client.complete(...)`` for free-form text and
``client.complete_json(...)`` for JSON-only responses. The JSON helper does
fence-stripping and a single retry on parse failure.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from . import config
from .io_utils import append_jsonl


class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    """Raised when a provider is selected but its API key/config is missing."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prompt_hash(messages: list[dict[str, str]], system: str | None) -> str:
    blob = json.dumps({"system": system, "messages": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*|\s*```$", re.MULTILINE)


def extract_json(raw: str) -> Any:
    """Strip markdown fences and parse JSON. Best-effort: also tries the first
    well-balanced { or [ block when surrounding prose is present.
    """
    text = raw.strip()
    text = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first balanced object/array
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"Could not parse JSON from model output: {raw[:300]}...")


class LLMClient(ABC):
    provider: str = "abstract"

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def _raw_complete(self, messages: list[dict[str, str]], system: str | None, json_mode: bool) -> str:
        ...

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        json_mode: bool = False,
        stage: str,
        query_id: str | None = None,
        input_artifacts: list[str] | None = None,
        output_artifact: str | None = None,
    ) -> str:
        """Call the model and log the call. Returns the raw text."""
        t0 = time.time()
        try:
            text = self._raw_complete(messages, system, json_mode)
        except Exception as e:
            self._log_call(
                stage=stage,
                query_id=query_id,
                messages=messages,
                system=system,
                output_artifact=output_artifact,
                input_artifacts=input_artifacts,
                latency_ms=int((time.time() - t0) * 1000),
                error=str(e),
            )
            raise
        self._log_call(
            stage=stage,
            query_id=query_id,
            messages=messages,
            system=system,
            output_artifact=output_artifact,
            input_artifacts=input_artifacts,
            latency_ms=int((time.time() - t0) * 1000),
            error=None,
        )
        return text

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        stage: str,
        query_id: str | None = None,
        input_artifacts: list[str] | None = None,
        output_artifact: str | None = None,
    ) -> Any:
        """Call the model expecting JSON. Strips fences; retries once on parse failure."""
        text = self.complete(
            messages,
            system=system,
            json_mode=True,
            stage=stage,
            query_id=query_id,
            input_artifacts=input_artifacts,
            output_artifact=output_artifact,
        )
        try:
            return extract_json(text)
        except LLMError:
            # One retry with stricter reminder. This is logged as a separate call.
            retry_messages = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. Return ONLY the "
                        "JSON value — no prose, no markdown, no code fences."
                    ),
                }
            ]
            retried = self.complete(
                retry_messages,
                system=system,
                json_mode=True,
                stage=f"{stage}_retry",
                query_id=query_id,
                input_artifacts=input_artifacts,
                output_artifact=output_artifact,
            )
            return extract_json(retried)

    def _log_call(
        self,
        *,
        stage: str,
        query_id: str | None,
        messages: list[dict[str, str]],
        system: str | None,
        input_artifacts: list[str] | None,
        output_artifact: str | None,
        latency_ms: int,
        error: str | None,
    ) -> None:
        record = {
            "stage": stage,
            "query_id": query_id,
            "timestamp": _now_iso(),
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": _prompt_hash(messages, system),
            "input_artifacts": input_artifacts or [],
            "output_artifact": output_artifact,
            "latency_ms": latency_ms,
            "error": error,
        }
        append_jsonl(config.LLM_CALLS_PATH, record)


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, model: str = config.OLLAMA_MODEL, base_url: str = config.OLLAMA_BASE_URL) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")

    def _raw_complete(self, messages: list[dict[str, str]], system: str | None, json_mode: bool) -> str:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed (is the daemon running at {self.base_url}?): {e}") from e
        if r.status_code != 200:
            raise LLMError(f"Ollama returned {r.status_code}: {r.text[:300]}")
        data = r.json()
        msg = data.get("message", {}) or {}
        content = msg.get("content", "")
        if not isinstance(content, str):
            raise LLMError(f"Unexpected Ollama response shape: {data}")
        return content


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str = config.ANTHROPIC_MODEL, api_key: str = config.ANTHROPIC_API_KEY) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise LLMConfigurationError("anthropic SDK not installed. Run: pip install anthropic") from e
        self._client = anthropic.Anthropic(api_key=api_key)

    def _raw_complete(self, messages: list[dict[str, str]], system: str | None, json_mode: bool) -> str:
        # Anthropic separates system from messages.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        # Concatenate text blocks
        return "".join(getattr(b, "text", "") for b in resp.content)


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str = config.OPENAI_MODEL, api_key: str = config.OPENAI_API_KEY) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise LLMConfigurationError("openai SDK not installed. Run: pip install openai") from e
        self._client = OpenAI(api_key=api_key)

    def _raw_complete(self, messages: list[dict[str, str]], system: str | None, json_mode: bool) -> str:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": 0.0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class GeminiClient(LLMClient):
    """Google Gemini via REST. No SDK dependency — uses the v1beta REST API."""

    provider = "gemini"

    def __init__(self, model: str = config.GEMINI_MODEL, api_key: str = config.GEMINI_API_KEY) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is not set.")
        self.api_key = api_key

    def _raw_complete(self, messages: list[dict[str, str]], system: str | None, json_mode: bool) -> str:
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": 0.0},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        try:
            r = requests.post(url, json=body, timeout=300)
        except requests.RequestException as e:
            raise LLMError(f"Gemini request failed: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"Gemini returned {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Gemini response shape: {data}") from e


class GroqClient(LLMClient):
    """Groq via OpenAI-compatible REST endpoint."""

    provider = "groq"

    def __init__(self, model: str = config.GROQ_MODEL, api_key: str = config.GROQ_API_KEY) -> None:
        super().__init__(model)
        if not api_key:
            raise LLMConfigurationError("GROQ_API_KEY is not set.")
        self.api_key = api_key

    def _raw_complete(self, messages: list[dict[str, str]], system: str | None, json_mode: bool) -> str:
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": 0.0,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=300,
            )
        except requests.RequestException as e:
            raise LLMError(f"Groq request failed: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"Groq returned {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Groq response shape: {data}") from e


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[LLMClient]] = {
    "ollama": OllamaClient,
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
}


def make_client(provider: str | None = None) -> LLMClient:
    name = (provider or config.LLM_PROVIDER).lower().strip()
    if name not in _PROVIDERS:
        raise LLMConfigurationError(
            f"Unknown LLM_PROVIDER '{name}'. Choose one of: {sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[name]()
