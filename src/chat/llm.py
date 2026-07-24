"""
Provider-agnostic LLM client, swappable behind one interface (per assignment
tech requirements). Ships with an Ollama implementation to match the local
setup already used elsewhere. Credentials/host read from env vars only --
nothing hardcoded, nothing committed (see .env.example).

Isolating non-determinism: this is the ONLY module in the whole system that
calls an LLM. The delta engine (src/delta/) is 100% deterministic. If the
chat layer's answers vary run-to-run, that variance lives here and nowhere
else -- easy to point at in review.
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str


class LLMClient(ABC):
    @abstractmethod
    def chat(self, system: str, user: str) -> LLMResponse:
        ...


class OllamaClient(LLMClient):
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def chat(self, system: str, user: str) -> LLMResponse:
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            raise RuntimeError(
                f"Ollama call failed ({exc}). Is `ollama serve` running and is "
                f"`{self.model}` pulled? Try: ollama pull {self.model}"
            ) from exc
        latency_ms = (time.time() - t0) * 1000

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            model=self.model,
        )


class MockLLMClient(LLMClient):
    """Deterministic offline stand-in used by tests and CI where no local
    Ollama server is available. Returns a templated, still-grounded answer
    built directly from whatever context was retrieved, so the eval harness
    can run end-to-end without a live model."""

    def chat(self, system: str, user: str) -> LLMResponse:
        # Pull citation-looking tokens straight out of the prompt so the
        # mock's output still exercises the citation-parsing code path.
        import re
        cites = re.findall(r"\[(pid_[ab]:[^\]]+|delta:[^\]]+)\]", user)
        cite_str = " ".join(f"[{c}]" for c in cites[:3]) if cites else ""
        text = (
            "Based on the retrieved context, here is a grounded (mock) answer. "
            f"{cite_str}"
        )
        return LLMResponse(text=text, prompt_tokens=len(user.split()),
                            completion_tokens=len(text.split()), latency_ms=1.0, model="mock")


def get_llm_client() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Supported: ollama, mock")
