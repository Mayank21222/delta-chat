"""
Grounded answer: retrieve -> build a citation-forcing prompt -> call LLM ->
parse which citations were actually used.

Grounding is enforced two ways:
  1. Prompt-level: the system prompt requires every claim to carry a
     [chunk_id] citation drawn only from the provided context, and requires
     an explicit "not found in the provided documents" when unsupported.
  2. Post-hoc, checkable: `cited_chunk_ids` is parsed out of the raw answer
     text and cross-checked against what was actually retrieved (used by
     eval/metrics.py's groundedness score) -- so "did it cite something
     real" is a mechanical check, not a vibe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.chat.index import Chunk, RetrievalIndex
from src.chat.llm import LLMClient
from src.observability.tracing import Trace

SYSTEM_PROMPT = """You are a grounded technical assistant answering questions about \
piping & instrumentation drawings (P&IDs) and a delta report describing changes \
between two revisions of such a drawing.

You have been given context chunks from two P&ID revisions and a delta report. \
Each chunk is labeled with an ID like [pid_a:1:441] or [delta:D0001].

Rules:
- Use ALL information in the context to answer the question. Read every chunk carefully.
- Every factual claim MUST end with a citation in square brackets using the exact \
chunk id given in the context, e.g. "The setpoint is 230.0 bar(g) [delta:D0001]".
- If the context does not contain the answer, say exactly: \
"Not found in the provided documents." Do not guess.
- Be concise. This is an engineering review tool, not a chat companion.
- If you see delta entries (labeled delta:), they describe what changed between revisions. \
Use them to answer "what changed" questions."""

CITATION_RE = re.compile(r"\[(pid_[ab]:[^\]]+|delta:[^\]]+)\]")


@dataclass
class ChatAnswer:
    question: str
    answer_text: str
    cited_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    grounded: bool  # True if every citation in the answer maps to a retrieved chunk
    retrieved_chunks: list[Chunk] = None  # full chunks for retrieval quality eval


def _build_context(retrieved: list[tuple[Chunk, float]]) -> str:
    lines = []
    for chunk, score in retrieved:
        lines.append(f"[{chunk.chunk_id}] (source: {chunk.citation}, relevance: {score:.2f})\n{chunk.text}")
    return "\n\n".join(lines)


def ask(question: str, index: RetrievalIndex, llm: LLMClient, trace: Trace, top_k: int = 15) -> ChatAnswer:
    with trace.span("retrieve", top_k=top_k) as span:
        retrieved = index.search(question, top_k=top_k)
        span.metadata["num_retrieved"] = len(retrieved)
        span.metadata["chunk_ids"] = [c.chunk_id for c, _ in retrieved]

    context = _build_context(retrieved)
    user_prompt = f"Context:\n\n{context}\n\nQuestion: {question}"

    with trace.span("llm_generate", model=getattr(llm, "model", "unknown")):
        response = llm.chat(SYSTEM_PROMPT, user_prompt)

    trace.record_llm_call(
        model=response.model,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_ms=response.latency_ms,
    )

    cited = CITATION_RE.findall(response.text)
    retrieved_ids = {c.chunk_id for c, _ in retrieved}
    grounded = all(c in retrieved_ids for c in cited) if cited else False

    if not retrieved:
        return ChatAnswer(
            question=question,
            answer_text="Not found in the provided documents. No relevant context was retrieved.",
            cited_chunk_ids=[],
            retrieved_chunk_ids=[],
            grounded=True,
        )

    return ChatAnswer(
        question=question,
        answer_text=response.text,
        cited_chunk_ids=cited,
        retrieved_chunk_ids=list(retrieved_ids),
        grounded=grounded,
        retrieved_chunks=[c for c, _ in retrieved],
    )
