"""
Alignment: matching is the hard part, not diffing (per the assignment brief).

Four-pass, deterministic, no LLM:

  Pass 1 (exact): elements with an identical fingerprint (normalized text,
  same page) are matched immediately and marked UNCHANGED. This handles the
  vast majority of a P&ID sheet, which does not change between revisions.

  Pass 2 (fuzzy, per-page): everything left unmatched on a page is scored
  pairwise using a blend of text similarity and spatial proximity. Greedy
  assignment above a threshold produces MODIFIED candidates.

  Pass 3 (moved, cross-page): unmatched elements across different pages
  with high text similarity are classified as MOVED. This catches content
  that was relocated between sheets between revisions -- a common pattern
  in drawing updates.

  Pass 4 (unmatched): anything still unmatched after passes 1-3 is
  classified as REMOVED (in A only) or ADDED (in B only).

Why not embeddings/LLM alignment: this is deterministic, fast, free, and
directly inspectable (you can point at the exact score that produced a
match). The trade-off is real -- see README for where this breaks.
"""
from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from enum import Enum

from src.canonical.model import CanonicalDocument, CanonicalElement


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


FUZZY_MATCH_THRESHOLD = _env_float("ALIGN_FUZZY_THRESHOLD", 0.45)
TEXT_WEIGHT = _env_float("ALIGN_TEXT_WEIGHT", 0.65)
SPATIAL_WEIGHT = _env_float("ALIGN_SPATIAL_WEIGHT", 0.35)
MOVED_TEXT_THRESHOLD = _env_float("ALIGN_MOVED_TEXT_THRESHOLD", 0.70)


class MatchKind(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    MOVED = "moved"
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

    # Collect unmatched elements across all pages for cross-page pass
    all_remaining_a: list[CanonicalElement] = []
    all_remaining_b: list[CanonicalElement] = []

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

        # pass 2: fuzzy scoring + greedy assignment (per-page)
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

        all_remaining_a.extend(ea for ea in remaining_a if ea.id not in used_a)
        all_remaining_b.extend(eb for eb in remaining_b if eb.id not in used_b)

    # pass 3: cross-page moved detection
    # High text similarity across different pages = content was relocated
    if all_remaining_a and all_remaining_b:
        moved_scored = []
        for ea in all_remaining_a:
            for eb in all_remaining_b:
                t = _text_sim(ea.text, eb.text)
                if t >= MOVED_TEXT_THRESHOLD:
                    moved_scored.append((t, ea, eb))
        moved_scored.sort(key=lambda x: -x[0])

        moved_a: set[str] = set()
        moved_b: set[str] = set()
        for t, ea, eb in moved_scored:
            if ea.id in moved_a or eb.id in moved_b:
                continue
            moved_a.add(ea.id)
            moved_b.add(eb.id)
            matches.append(Match(MatchKind.MOVED, ea, eb, round(t, 3)))

        all_remaining_a = [ea for ea in all_remaining_a if ea.id not in moved_a]
        all_remaining_b = [eb for eb in all_remaining_b if eb.id not in moved_b]

    # pass 4: everything still unmatched
    for ea in all_remaining_a:
        matches.append(Match(MatchKind.UNMATCHED_A, ea, None, 0.0))
    for eb in all_remaining_b:
        matches.append(Match(MatchKind.UNMATCHED_B, None, eb, 0.0))

    return matches
