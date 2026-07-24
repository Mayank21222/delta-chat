"""
DWG adapter — STUB, kept honest per the assignment FAQ ("keep DWG as a real
stub behind the adapter seam" is explicitly acceptable).

This is a real class implementing the real FormatAdapter interface and
producing a real CanonicalDocument shape — it is wired into detect_format()
and the AdapterRegistry exactly like the other two. What's missing is the
DWG parsing itself.

What the real implementation would do (cut for time, not because it's hard
to describe):
  1. DWG is a closed binary format. Two practical paths:
     a) Convert DWG -> DXF with the (free) ODA File Converter, then parse
        DXF with `ezdxf` (pure Python, reads TEXT/MTEXT/DIMENSION/INSERT
        entities with coordinates directly — very close to native PDF's
        word+bbox shape).
     b) If only a viewer/print is available, rasterize and fall back to
        the OCR path in pdf_scanned.py.
  2. Map DXF entities to CanonicalElement:
       TEXT/MTEXT         -> text line, classify() reused as-is
       DIMENSION entities -> ElementType.DIMENSION directly (DXF *tells*
                              you it's a dimension — better signal than the
                              regex guess the PDF adapters have to make)
       INSERT (block refs) -> ElementType.TAG when the block name matches
                              an instrument/equipment block naming convention
       LWPOLYLINE/LINE/CIRCLE -> ElementType.GEOMETRY (kept for markup
                              bonus / spatial context, mostly ignored by
                              the text-centric delta engine below)
     DXF space is in drawing units, not PDF points — bbox scale is a real
     wrinkle (paper space vs. model space, sheet-dependent scale factor).
  3. Layers matter in DWG in a way pages don't in PDF (a DWG can encode
     revision clouds / redlines on a dedicated layer) — worth a dedicated
     `layer` field on CanonicalElement if this were built out.
"""
from __future__ import annotations

from src.canonical.model import CanonicalDocument
from src.ingest.base import FormatAdapter, ResolvedPID


class DWGAdapter(FormatAdapter):
    format_name = "dwg"

    def ingest(self, resolved: ResolvedPID) -> CanonicalDocument:
        raise NotImplementedError(
            "DWG ingestion is stubbed for this take-home (see module docstring "
            "for the real implementation path: DWG->DXF via ODA File Converter, "
            "then ezdxf entity extraction). The adapter interface is real and "
            "registered; only the parsing body is missing."
        )
