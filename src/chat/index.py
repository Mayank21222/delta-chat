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
            self.chunks.append(Chunk(
                chunk_id=f"delta:{e.id}",
                kind="delta",
                pid=None,
                page=e.location.page,
                text=e.description,
                citation=f"delta report [{e.id}], sheet {e.location.page}",
            ))

    def build(self) -> None:
        texts = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if self._vectorizer is None:
            raise RuntimeError("call .build() before .search()")
        qvec = self._vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: -sims[i])[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in ranked if sims[i] > 0]
