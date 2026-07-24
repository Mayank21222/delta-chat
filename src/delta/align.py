"""
Alignment: matching is the hard part, not diffing (per the assignment brief).

Two-pass, deterministic, no LLM:

  Pass 1 (exact): elements with an identical fingerprint (normalized text,
  same page) are matched immediately and marked UNCHANGED. This handles the
  vast majority of a P&ID sheet, which does not change between revisions.

  Pass 2 (fuzzy): everything left unmatched on a page is scored pairwise
  using a blend of text similarity (difflib) and spatial proximity
  (normalized bbox-center distance) -- a "MODIFIED" candidate should read
  similarly to its predecessor AND sit roughly where it used to sit. Greedy
  assignment, highest score first, above a threshold. Below the threshold,
  or with nothing left to pair against, an element is unmatched: present
  only in A -> REMOVED, present only in B -> ADDED.

Why not embeddings/LLM alignment: this is deterministic, fast, free, and
directly inspectable (you can point at the exact score that produced a
match). The trade-off is real -- see README for where this breaks (e.g.
if a whole page is relabeled/renumbered, spatial proximity stops helping
and this degrades toward text-similarity-only matching).
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import Enum

from src.canonical.model import CanonicalDocument, CanonicalElement

FUZZY_MATCH_THRESHOLD = 0.45
TEXT_WEIGHT = 0.65
SPATIAL_WEIGHT = 0.35


class MatchKind(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    UNMATCHED_A = "unmatched_a"   # -> removed
    UNMATCHED_B = "unmatched_b"   # -> added


@dataclass
class Match:
    kind: MatchKind
    a: CanonicalElement | None
    b: CanonicalElement | None
    score: float


def _text_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.strip().upper(), b.strip().upper()).ratio()


def _spatial_sim(a: CanonicalElement, b: CanonicalElement, diag: float) -> float:
    if diag <= 0:
        return 0.0
    ax, ay = a.bbox.center()
    bx, by = b.bbox.center()
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return max(0.0, 1.0 - dist / diag)


def align(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> list[Match]:
    matches: list[Match] = []
    pages = sorted(set(p.number for p in doc_a.pages) | set(p.number for p in doc_b.pages))

    by_page_a = {p.number: p for p in doc_a.pages}
    by_page_b = {p.number: p for p in doc_b.pages}

    for page_no in pages:
        pa = by_page_a.get(page_no)
        pb = by_page_b.get(page_no)
        els_a = list(pa.elements) if pa else []
        els_b = list(pb.elements) if pb else []
        diag = ((pa.width if pa else pb.width) ** 2 + (pa.height if pa else pb.height) ** 2) ** 0.5

        # pass 1: exact fingerprint match
        fp_b_index: dict[str, list[CanonicalElement]] = {}
        for eb in els_b:
            fp_b_index.setdefault(eb.fingerprint(), []).append(eb)

        remaining_a: list[CanonicalElement] = []
        matched_b_ids: set[str] = set()
        for ea in els_a:
            candidates = fp_b_index.get(ea.fingerprint(), [])
            candidates = [c for c in candidates if c.id not in matched_b_ids]
            if candidates:
                eb = candidates[0]
                matched_b_ids.add(eb.id)
                matches.append(Match(MatchKind.EXACT, ea, eb, 1.0))
            else:
                remaining_a.append(ea)

        remaining_b = [eb for eb in els_b if eb.id not in matched_b_ids]

        # pass 2: fuzzy scoring + greedy assignment
        scored = []
        for ea in remaining_a:
            for eb in remaining_b:
                t = _text_sim(ea.text, eb.text)
                s = _spatial_sim(ea, eb, diag)
                score = TEXT_WEIGHT * t + SPATIAL_WEIGHT * s
                if score >= FUZZY_MATCH_THRESHOLD:
                    scored.append((score, ea, eb))
        scored.sort(key=lambda x: -x[0])

        used_a: set[str] = set()
        used_b: set[str] = set()
        for score, ea, eb in scored:
            if ea.id in used_a or eb.id in used_b:
                continue
            used_a.add(ea.id)
            used_b.add(eb.id)
            matches.append(Match(MatchKind.FUZZY, ea, eb, round(score, 3)))

        for ea in remaining_a:
            if ea.id not in used_a:
                matches.append(Match(MatchKind.UNMATCHED_A, ea, None, 0.0))
        for eb in remaining_b:
            if eb.id not in used_b:
                matches.append(Match(MatchKind.UNMATCHED_B, None, eb, 0.0))

    return matches
