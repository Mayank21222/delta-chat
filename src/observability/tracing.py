"""
Per-request trace: the end-to-end record of one request (ingest -> delta ->
retrieve -> LLM -> answer) with stage timings, token/cost, inputs, outputs.

Deliberately file-based JSON, not a hosted tracing backend (Jaeger/Honeycomb/
etc.) -- right call for a take-home eval'd by reading a repo, wrong call for
production (see README "what I'd do differently"). Every trace is written to
traces/{correlation_id}.json and is inspectable with `cat | jq`.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.end is None:
            return None
        return round((self.end - self.start) * 1000, 2)

    def to_dict(self) -> dict:
        return {"name": self.name, "duration_ms": self.duration_ms, "metadata": self.metadata}


class Trace:
    def __init__(self, correlation_id: str, request_type: str, trace_dir: str = "traces"):
        self.correlation_id = correlation_id
        self.request_type = request_type
        self.trace_dir = trace_dir
        self.spans: list[Span] = []
        self.started_at = time.time()
        self.llm_calls: list[dict] = []
        self.error: str | None = None

    @contextmanager
    def span(self, name: str, **metadata):
        s = Span(name=name, start=time.time(), metadata=metadata)
        try:
            yield s
        except Exception as exc:
            s.metadata["error"] = str(exc)
            self.error = str(exc)
            raise
        finally:
            s.end = time.time()
            self.spans.append(s)

    def record_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int,
                         latency_ms: float, cost_usd: float = 0.0) -> None:
        self.llm_calls.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": cost_usd,
        })

    def finish_and_save(self) -> str:
        os.makedirs(self.trace_dir, exist_ok=True)
        total_ms = round((time.time() - self.started_at) * 1000, 2)
        payload = {
            "correlation_id": self.correlation_id,
            "request_type": self.request_type,
            "total_duration_ms": total_ms,
            "error": self.error,
            "spans": [s.to_dict() for s in self.spans],
            "llm_calls": self.llm_calls,
            "total_tokens": sum(c["total_tokens"] for c in self.llm_calls),
            "total_cost_usd": sum(c["cost_usd"] for c in self.llm_calls),
        }
        path = os.path.join(self.trace_dir, f"{self.correlation_id}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path
