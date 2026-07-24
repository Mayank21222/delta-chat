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
    """Sniff format from magic bytes / extension. Cheap and deliberately dumb.

    Real-world hardening (not done here, see README cuts): detect "scanned"
    vs "native" PDF by checking whether >~90% of pages have an extractable
    text layer, rather than trusting a filename convention.
    """
    head = resolved.bytes_[:8]
    ext = os.path.splitext(resolved.filename)[1].lower()

    if head.startswith(b"%PDF"):
        if "_scanned" in resolved.filename.lower():
            return "pdf_scanned"
        return "pdf_native"
    if ext == ".dwg" or head[:4] in (b"AC10", b"AC15", b"AC18", b"AC21", b"AC24"):
        return "dwg"
    raise ValueError(f"Unrecognized format for PID {resolved.pid} ({resolved.filename})")


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
