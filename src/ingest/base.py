"""
The ingestion seam.

A `PID` is a handle to one document revision. `resolve_pid` turns that handle
into bytes + metadata. `detect_format` sniffs the format. Each `FormatAdapter`
knows how to turn bytes of ONE format into a `CanonicalDocument`. Nothing
downstream of `ingest()` ever imports a format-specific module again.

Adding a 4th format = write one adapter class + register it. That's the
test this abstraction has to survive (see README "what a 4th format costs").
"""
from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.canonical.model import CanonicalDocument


@dataclass
class ResolvedPID:
    pid: str
    bytes_: bytes
    filename: str
    revision_label: str | None = None


def resolve_pid(pid: str, path: str, revision_label: str | None = None) -> ResolvedPID:
    """Resolve a PID handle to raw bytes + metadata.

    In this take-home a PID resolves to a local file path. In production this
    is where you'd hit a document management system / object store by PID and
    return the same shape — nothing else in the pipeline would need to change.
    """
    with open(path, "rb") as f:
        data = f.read()
    return ResolvedPID(pid=pid, bytes_=data, filename=os.path.basename(path), revision_label=revision_label)


def detect_format(resolved: ResolvedPID) -> str:
    """Sniff format from magic bytes + text-layer heuristics.

    For PDFs: inspects whether pages have extractable text. If fewer than
    half the pages produce meaningful text (>50 chars), the PDF is treated
    as scanned. The filename convention (_scanned suffix) is used as a
    fallback when text extraction fails (e.g. corrupted or unusual PDFs).
    """
    head = resolved.bytes_[:8]
    ext = os.path.splitext(resolved.filename)[1].lower()

    if head.startswith(b"%PDF"):
        if _has_text_layer(resolved.bytes_):
            return "pdf_native"
        return "pdf_scanned"
    if ext == ".dwg" or head[:4] in (b"AC10", b"AC15", b"AC18", b"AC21", b"AC24"):
        return "dwg"
    raise ValueError(f"Unrecognized format for PID {resolved.pid} ({resolved.filename})")


def _has_text_layer(pdf_bytes: bytes) -> bool:
    """Check whether a PDF has a meaningful extractable text layer.

    Opens the PDF with PyMuPDF and checks how many pages produce >50 chars
    of extractable text. If fewer than half do, the PDF is treated as
    scanned/rasterized. This is more robust than a filename convention.
    """
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_pages = 0
        total_pages = len(doc)
        if total_pages == 0:
            return False
        for i in range(min(total_pages, 5)):  # check first 5 pages max
            text = doc[i].get_text().strip()
            if len(text) > 50:
                text_pages += 1
        doc.close()
        return text_pages > total_pages / 2
    except Exception:
        # If we can't open with fitz, assume native (will fail gracefully downstream)
        return True


class FormatAdapter(ABC):
    """One adapter per format. Must produce a CanonicalDocument."""

    format_name: str

    @abstractmethod
    def ingest(self, resolved: ResolvedPID) -> CanonicalDocument:
        ...


class AdapterRegistry:
    _adapters: dict[str, FormatAdapter] = {}

    @classmethod
    def register(cls, format_name: str, adapter: FormatAdapter) -> None:
        cls._adapters[format_name] = adapter

    @classmethod
    def get(cls, format_name: str) -> FormatAdapter:
        if format_name not in cls._adapters:
            raise ValueError(
                f"No adapter registered for format '{format_name}'. "
                f"Registered: {list(cls._adapters)}"
            )
        return cls._adapters[format_name]


def ingest_pid(pid: str, path: str, revision_label: str | None = None) -> CanonicalDocument:
    """The one function everything downstream calls. Format-agnostic by construction."""
    resolved = resolve_pid(pid, path, revision_label)
    fmt = detect_format(resolved)
    adapter = AdapterRegistry.get(fmt)
    doc = adapter.ingest(resolved)
    doc.revision_label = revision_label
    return doc
