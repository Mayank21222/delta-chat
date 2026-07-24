"""
Scanned PDF adapter: raster/image PDFs with no reliable text layer.

Rasterizes each page (pdf2image/poppler), runs Tesseract OCR with
per-word bounding boxes and confidence (pytesseract.image_to_data), then
groups words into lines using Tesseract's own (block, par, line) numbering
-- mirroring exactly what the native adapter does with PyMuPDF's
(block, line) numbering. Same canonical output shape, same classify()
heuristics reused from pdf_native. That reuse is the point of the seam:
the delta engine and chat layer downstream cannot tell these two adapters
apart.

OCR confidence is preserved per-element (0-1, normalized from Tesseract's
0-100) and threaded into the canonical model so a low-confidence OCR read
can be flagged distinctly from a low-confidence delta match.
"""
from __future__ import annotations

from collections import defaultdict

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalPage, ElementType
from src.ingest.base import FormatAdapter, ResolvedPID
from src.ingest.pdf_native import classify

OCR_DPI = 300


class PDFScannedAdapter(FormatAdapter):
    format_name = "pdf_scanned"

    def ingest(self, resolved: ResolvedPID) -> CanonicalDocument:
        images: list[Image.Image] = convert_from_bytes(resolved.bytes_, dpi=OCR_DPI)
        pages: list[CanonicalPage] = []
        scale = 72.0 / OCR_DPI  # convert pixel coords back to PDF points

        for page_index, image in enumerate(images):
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            lines: dict[tuple, list] = defaultdict(list)
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                if not text:
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines[key].append((
                    data["left"][i], data["top"][i],
                    data["width"][i], data["height"][i],
                    text, data["word_num"][i], int(data["conf"][i]),
                ))

            elements: list[CanonicalElement] = []
            el_idx = 0
            for key, word_list in sorted(lines.items()):
                word_list.sort(key=lambda w: w[5])
                text = " ".join(w[4] for w in word_list).strip()
                if not text:
                    continue
                confs = [w[6] for w in word_list if w[6] >= 0]
                avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0

                x0 = min(w[0] for w in word_list) * scale
                y0 = min(w[1] for w in word_list) * scale
                x1 = max(w[0] + w[2] for w in word_list) * scale
                y1 = max(w[1] + w[3] for w in word_list) * scale

                elements.append(CanonicalElement(
                    id=f"{page_index + 1}:{el_idx}",
                    type=classify(text),
                    text=text,
                    bbox=BBox(x0, y0, x1, y1),
                    page=page_index + 1,
                    confidence=round(avg_conf, 3),
                    source="ocr",
                ))
                el_idx += 1

            pages.append(CanonicalPage(
                number=page_index + 1,
                width=image.width * scale,
                height=image.height * scale,
                elements=elements,
            ))

        return CanonicalDocument(
            pid=resolved.pid,
            source_format=self.format_name,
            revision_label=resolved.revision_label,
            pages=pages,
        )
