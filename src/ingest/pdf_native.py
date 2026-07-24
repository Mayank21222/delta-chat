"""
Native PDF adapter: born-digital PDFs with an extractable text layer.

Uses PyMuPDF's word extraction, which already segments words into
(block, line) groups. We roll words on the same (block, line) into one
CanonicalElement — that's our "line" granularity unit (see canonical/model.py
docstring for why lines and not words or pages).
"""
from __future__ import annotations

import re
from collections import defaultdict

import fitz  # PyMuPDF

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalPage, ElementType
from src.ingest.base import FormatAdapter, ResolvedPID

# --- element classification heuristics -------------------------------------
# Deliberately regex/heuristic, not ML: fast, deterministic, auditable, and
# good enough for P&ID-style structured text. Documented as a trade-off in
# the README rather than hidden.

_TAG_RE = re.compile(r"^\d{2}-[A-Z]{2,4}-\d{3,4}[A-Z]?$")                    # 26-KA-902
_INSTR_TAG_RE = re.compile(r"^(PI|TI|PIT|TIT|PDI|PDIT|FI|FE|FIT|PSV|PSE|PSI|XV|LC|LO)[\s-]?\d{3,4}[A-Z]?$")
_SETPOINT_RE = re.compile(r"(SP\s*=|SET PRESSURE|HH\s*:|LL\s*:|^H\s*:|^L\s*:)", re.IGNORECASE)
_DIMENSION_RE = re.compile(r'\d+(\.\d+)?["\']?\s*-[A-Z]{2}-\d+-\d+-[A-Z0-9]+-\d+|^\d+["\'"]\s*x\s*\d+["\']?$')
_NOTE_RE = re.compile(r"^\d{1,2}\.\s+\S")
_TABLE_LABEL_RE = re.compile(
    r"^(SERVICE|DUTY|FLOW RATE|TAG NUMBER|TYPE|VENDOR|QUANTITY|MATERIAL|"
    r"DISCHARGE|SUCTION)\b", re.IGNORECASE,
)


def classify(text: str) -> ElementType:
    t = text.strip()
    if _TAG_RE.match(t) or _INSTR_TAG_RE.match(t.replace(" ", "")):
        return ElementType.TAG
    if _SETPOINT_RE.search(t):
        return ElementType.SETPOINT
    if _NOTE_RE.match(t):
        return ElementType.NOTE
    if _TABLE_LABEL_RE.match(t):
        return ElementType.TABLE_CELL
    if _DIMENSION_RE.search(t):
        return ElementType.DIMENSION
    return ElementType.TEXT


class PDFNativeAdapter(FormatAdapter):
    format_name = "pdf_native"

    def ingest(self, resolved: ResolvedPID) -> CanonicalDocument:
        doc = fitz.open(stream=resolved.bytes_, filetype="pdf")
        pages: list[CanonicalPage] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            words = page.get_text("words")  # (x0,y0,x1,y1,text,block_no,line_no,word_no)

            lines: dict[tuple, list] = defaultdict(list)
            for x0, y0, x1, y1, text, block_no, line_no, word_no in words:
                lines[(block_no, line_no)].append((x0, y0, x1, y1, text, word_no))

            elements: list[CanonicalElement] = []
            el_idx = 0
            for (block_no, line_no), word_list in sorted(lines.items()):
                word_list.sort(key=lambda w: w[5])
                text = " ".join(w[4] for w in word_list).strip()
                if not text:
                    continue
                x0 = min(w[0] for w in word_list)
                y0 = min(w[1] for w in word_list)
                x1 = max(w[2] for w in word_list)
                y1 = max(w[3] for w in word_list)

                elements.append(CanonicalElement(
                    id=f"{page_index + 1}:{el_idx}",
                    type=classify(text),
                    text=text,
                    bbox=BBox(x0, y0, x1, y1),
                    page=page_index + 1,
                    confidence=1.0,
                    source="native",
                ))
                el_idx += 1

            pages.append(CanonicalPage(
                number=page_index + 1,
                width=page.rect.width,
                height=page.rect.height,
                elements=elements,
            ))

        return CanonicalDocument(
            pid=resolved.pid,
            source_format=self.format_name,
            revision_label=resolved.revision_label,
            pages=pages,
        )
