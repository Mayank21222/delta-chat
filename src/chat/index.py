"""
Retrieval index over PID A, PID B, and the delta report -- the three source
pools grounded chat must draw from (per assignment section 3.D).

TF-IDF + cosine similarity via scikit-learn, not a vector DB. This is a
deliberate scope cut for the take-home: P&ID text content for one sheet
pair is a few hundred short chunks -- nowhere near the scale where TF-IDF's
lack of semantic matching becomes the bottleneck vs. embedding cost/infra.
Documented in README as "what I'd swap for a 500-sheet set": a real vector
store + embedding model, because at that scale TF-IDF's exact-token
dependence will visibly miss paraphrased questions.

Every chunk keeps a stable `citation` string that both the delta report and
the canonical documents already provide for free (page + bbox, or delta
entry id) -- grounding falls out of the data model rather than being
bolted on afterward.
"""
from __future__ import annotations

import re as _re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.canonical.model import CanonicalDocument
from src.delta.engine import DeltaEntry


@dataclass
class Chunk:
    chunk_id: str
    kind: str          # "pid_a" | "pid_b" | "delta"
    pid: str | None
    page: int
    text: str
    citation: str       # human-readable, used verbatim in answers


class RetrievalIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add_document(self, doc: CanonicalDocument, kind: str) -> None:
        for el in doc.all_elements():
            if not el.text.strip():
                continue
            self.chunks.append(Chunk(
                chunk_id=f"{kind}:{el.id}",
                kind=kind,
                pid=doc.pid,
                page=el.page,
                text=el.text,
                citation=f"{doc.pid} (rev {doc.revision_label or '?'}), sheet {el.page} @({el.bbox.x0:.0f},{el.bbox.y0:.0f})",
            ))

    def add_delta_report(self, entries: list[DeltaEntry]) -> None:
        for e in entries:
            change_word = e.change_type.value.capitalize()
            type_word = e.element_type if e.element_type else ""
            rich_text = f"{change_word} {type_word}: {e.description}"
            self.chunks.append(Chunk(
                chunk_id=f"delta:{e.id}",
                kind="delta",
                pid=None,
                page=e.location.page,
                text=rich_text,
                citation=f"delta report [{e.id}], sheet {e.location.page}",
            ))

    def build(self) -> None:
        texts = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        self._matrix = self._vectorizer.fit_transform(texts)

    def _neighbors(self, chunk_idx: int, radius: int = 3) -> list[int]:
        """Return indices of chunks spatially adjacent (same kind + page, nearby element index)."""
        if chunk_idx < 0 or chunk_idx >= len(self.chunks):
            return []
        anchor = self.chunks[chunk_idx]
        neighbors = []
        for i in range(max(0, chunk_idx - radius), min(len(self.chunks), chunk_idx + radius + 1)):
            if i == chunk_idx:
                continue
            c = self.chunks[i]
            if c.kind == anchor.kind and c.page == anchor.page:
                neighbors.append(i)
        return neighbors

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if self._vectorizer is None:
            raise RuntimeError("call .build() before .search()")
        qvec = self._vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: -sims[i])[:top_k]
        results = [(self.chunks[i], float(sims[i])) for i in ranked if sims[i] > 0]
        seen_ids = {c.chunk_id for c, _ in results}

        # Label/value proximity boost: find short chunks (<6 words) that
        # contain a query keyword. These are almost certainly labels or tags
        # on the P&ID (e.g. "VENDOR", "MOTOR DRIVE POWER"). Pull in their
        # same-page neighbors to catch the associated value that TF-IDF alone
        # can't link (e.g. "MAN ENERGY SOLUTIONS" next to "VENDOR").
        q_words = {w.lower() for w in _re.findall(r"\w{3,}", query.lower())}
        for i, c in enumerate(self.chunks):
            if c.kind == "delta" or c.chunk_id in seen_ids:
                continue
            c_words = set(_re.findall(r"\w{3,}", c.text.lower()))
            word_count = len(c.text.split())
            if word_count <= 5 and q_words & c_words:
                for nidx in self._neighbors(i, radius=5):
                    nc = self.chunks[nidx]
                    if nc.chunk_id not in seen_ids:
                        results.append((nc, 0.9))
                        seen_ids.add(nc.chunk_id)

        # Small spatial expansion from top TF-IDF results (single hop, radius 3)
        for i in ranked[:top_k]:
            if sims[i] <= 0:
                continue
            for nidx in self._neighbors(i, radius=3):
                c = self.chunks[nidx]
                if c.chunk_id not in seen_ids:
                    results.append((c, float(sims[i]) * 0.3))
                    seen_ids.add(c.chunk_id)

        # Always include delta chunks (critical for "what changed" questions)
        for c in self.chunks:
            if c.kind == "delta" and c.chunk_id not in seen_ids:
                results.append((c, 0.0))
                seen_ids.add(c.chunk_id)

        # Cap non-delta results to keep LLM context manageable
        results.sort(key=lambda x: -x[1])
        max_chunks = 80
        delta_chunks = [(c, s) for c, s in results if c.kind == "delta"]
        non_delta = [(c, s) for c, s in results if c.kind != "delta"][:max_chunks]
        return non_delta + delta_chunks
