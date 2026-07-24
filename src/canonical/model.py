"""
Canonical representation: the format-agnostic intermediate model.

Every ingestion adapter (native PDF, scanned PDF/OCR, DWG) normalizes its
input into this model. Everything downstream (delta engine, chat/retrieval,
markup) only ever talks to this model — it never knows or cares what the
original format was.

Design choice: elements are kept at "line" granularity (a cluster of nearby
words on the same visual baseline), not word or page granularity. Words are
too fine (every OCR/kerning wobble becomes a diff), pages are too coarse
(you lose location). Lines are the unit a human means when they say "that
changed" on a drawing like a P&ID.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class ElementType(str, Enum):
    """Coarse semantic type, guessed by ingestion-time heuristics.

    This is intentionally coarse and heuristic — it is NOT a claim of deep
    document understanding. It exists so the delta engine can report
    *what kind* of thing changed (a tag vs. a setpoint vs. a note), which is
    far more useful to a reviewer than "text changed".
    """
    TAG = "tag"                # equipment/instrument tag, e.g. 26-KA-902, PIT 9019
    SETPOINT = "setpoint"      # SP=..., HH:, LL:, H:, alarm/trip setpoints
    DIMENSION = "dimension"    # line size / rating strings, e.g. 3"-DC-26-9026
    NOTE = "note"              # numbered drawing notes / free text annotations
    TABLE_CELL = "table_cell"  # datasheet-style key/value block (duty, flow, etc.)
    TEXT = "text"              # unclassified text line
    GEOMETRY = "geometry"      # vector/graphic entity (native PDF paths, DWG entities)


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple:
        return (self.x0, self.y0, self.x1, self.y1)

    def center(self) -> tuple:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)


@dataclass
class CanonicalElement:
    id: str                    # stable id: f"{page}:{index}", assigned at ingest
    type: ElementType
    text: str
    bbox: BBox
    page: int                  # 1-indexed page/sheet number
    confidence: float = 1.0    # 1.0 for native extraction; OCR confidence in [0,1] for scans
    source: str = "native"     # "native" | "ocr" | "dwg"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "CanonicalElement":
        d = dict(d)
        d["type"] = ElementType(d["type"])
        d["bbox"] = BBox(**d["bbox"])
        return CanonicalElement(**d)

    def fingerprint(self) -> str:
        """Content hash used as a cheap exact-match key during alignment."""
        norm = " ".join(self.text.strip().upper().split())
        return hashlib.sha1(f"{self.page}:{norm}".encode()).hexdigest()[:12]


@dataclass
class CanonicalPage:
    number: int
    width: float
    height: float
    elements: list[CanonicalElement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "width": self.width,
            "height": self.height,
            "elements": [e.to_dict() for e in self.elements],
        }

    @staticmethod
    def from_dict(d: dict) -> "CanonicalPage":
        return CanonicalPage(
            number=d["number"],
            width=d["width"],
            height=d["height"],
            elements=[CanonicalElement.from_dict(e) for e in d["elements"]],
        )


@dataclass
class CanonicalDocument:
    """The normalized form of one PID (one document revision)."""
    pid: str                       # the PID handle this was resolved from
    source_format: str             # "pdf_native" | "pdf_scanned" | "dwg"
    revision_label: Optional[str]  # human label if known, e.g. "Rev A"
    pages: list[CanonicalPage] = field(default_factory=list)

    def all_elements(self):
        for page in self.pages:
            for el in page.elements:
                yield el

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "source_format": self.source_format,
            "revision_label": self.revision_label,
            "pages": [p.to_dict() for p in self.pages],
        }

    @staticmethod
    def from_dict(d: dict) -> "CanonicalDocument":
        return CanonicalDocument(
            pid=d["pid"],
            source_format=d["source_format"],
            revision_label=d.get("revision_label"),
            pages=[CanonicalPage.from_dict(p) for p in d["pages"]],
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "CanonicalDocument":
        with open(path) as f:
            return CanonicalDocument.from_dict(json.load(f))
