"""
Synthesizes PID B (Rev B) from the real Rev A P&ID by applying a small,
deliberate set of edits via PyMuPDF redact-and-reinsert. This is the
provenance-documented approach the assignment FAQ explicitly sanctions
("export a PDF twice with edits").

The edits below ARE the ground truth used in eval/datasets/. Keep this
script and the ground-truth JSON in sync if you change one.
"""
import fitz

SRC = "data/samples/raw/pid_export_gas_compressor_revA.pdf"
OUT = "data/samples/raw/pid_export_gas_compressor_revB.pdf"

EDITS = [
    # (find_text, occurrence_index, replace_text_or_None, note)
    ("SP= 225.4 bar (g)", 1, "SP= 230.0 bar (g)",
     "PSV 9027B relief setpoint raised 225.4 -> 230.0 bar(g); PSV 9027A left unchanged"),
    ("HH : 214", 0, "HH : 220",
     "PIT 9023 high-high alarm raised 214 -> 220"),
    ("MECHANICAL", 0, None, "Removed 'MECHANICAL' label (interlock note deleted, part 1)"),
    ("INTERLOCK", 0, None, "Removed 'INTERLOCK' label (interlock note deleted, part 2)"),
    ("1835", 0, "1902", "Compressor duty revised 1835 kW -> 1902 kW"),
]

ADDED_TEXT = [
    # (x, y, text)
    (25, 610, "36. REV B: SUCTION STRAINER UPGRADED TO PERMANENT TYPE."),
]


def main():
    doc = fitz.open(SRC)
    page = doc[0]

    for needle, occ_idx, replacement, note in EDITS:
        hits = page.search_for(needle)
        if occ_idx >= len(hits):
            raise RuntimeError(f"Expected occurrence {occ_idx} of {needle!r}, found {len(hits)}")
        rect = hits[occ_idx]
        page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()
        if replacement:
            page.insert_text((rect.x0, rect.y1 - 1.5), replacement, fontsize=5.2, color=(0, 0, 0))
        print(f"applied: {note}")

    for x, y, text in ADDED_TEXT:
        page.insert_text((x, y), text, fontsize=5.2, color=(0, 0, 0))
        print(f"added: {text}")

    doc.save(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
