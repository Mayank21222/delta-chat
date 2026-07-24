"""
Delta report: the same delta rendered two ways.
  - JSON: machine-parseable, used by the eval harness and as a retrievable
    source for chat (each entry becomes a citeable chunk, see chat/index.py).
  - Markdown: human-readable, grouped by change type then page, with a
    summary count up top.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from src.delta.engine import DeltaEntry, save_delta


def render_markdown(entries: list[DeltaEntry], pid_a: str, pid_b: str) -> str:
    counts = Counter(e.change_type.value for e in entries)
    lines = [
        f"# Delta Report: {pid_a} -> {pid_b}",
        "",
        f"**Summary:** {len(entries)} changes — "
        f"{counts.get('added', 0)} added, {counts.get('removed', 0)} removed, "
        f"{counts.get('modified', 0)} modified.",
        "",
    ]

    by_type: dict[str, list[DeltaEntry]] = defaultdict(list)
    for e in entries:
        by_type[e.change_type.value].append(e)

    for change_type in ["modified", "added", "removed"]:
        group = by_type.get(change_type, [])
        if not group:
            continue
        lines.append(f"## {change_type.capitalize()} ({len(group)})")
        lines.append("")
        by_page: dict[int, list[DeltaEntry]] = defaultdict(list)
        for e in group:
            by_page[e.location.page].append(e)
        for page in sorted(by_page):
            lines.append(f"### Sheet {page}")
            for e in sorted(by_page[page], key=lambda x: -x.confidence):
                loc = f"({e.location.bbox[0]:.0f}, {e.location.bbox[1]:.0f})"
                lines.append(
                    f"- **[{e.id}]** {e.description}  \n"
                    f"  _type: {e.element_type} · location: sheet {page} {loc} "
                    f"· confidence: {e.confidence:.2f}_"
                )
            lines.append("")

    return "\n".join(lines)


def write_report(entries: list[DeltaEntry], pid_a: str, pid_b: str, out_dir: str) -> tuple[str, str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "delta.json")
    md_path = os.path.join(out_dir, "delta_report.md")

    save_delta(entries, json_path)
    with open(md_path, "w") as f:
        f.write(render_markdown(entries, pid_a, pid_b))

    return json_path, md_path
