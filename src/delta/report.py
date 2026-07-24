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
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total changes | {len(entries)} |",
        f"| Added | {counts.get('added', 0)} |",
        f"| Removed | {counts.get('removed', 0)} |",
        f"| Modified | {counts.get('modified', 0)} |",
    ]

    if entries:
        confs = [e.confidence for e in entries]
        avg_conf = sum(confs) / len(confs)
        min_conf = min(confs)
        max_conf = max(confs)
        lines.append(f"| Avg confidence | {avg_conf:.2f} |")
        lines.append(f"| Min confidence | {min_conf:.2f} |")
        lines.append(f"| Max confidence | {max_conf:.2f} |")

        # Element type breakdown
        type_counts = Counter(e.element_type for e in entries)
        lines.append("")
        lines.append("### Changes by Element Type")
        lines.append("")
        for etype, count in type_counts.most_common():
            lines.append(f"- **{etype}**: {count}")

        # Page breakdown
        pages = sorted(set(e.location.page for e in entries))
        if len(pages) > 1:
            lines.append("")
            lines.append("### Changes by Page")
            lines.append("")
            for p in pages:
                pcount = sum(1 for e in entries if e.location.page == p)
                lines.append(f"- **Sheet {p}**: {pcount} changes")

    lines.append("")
    lines.append("---")
    lines.append("")

    by_type: dict[str, list[DeltaEntry]] = defaultdict(list)
    for e in entries:
        by_type[e.change_type.value].append(e)

    for change_type in ["modified", "moved", "added", "removed"]:
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
                before_after = ""
                if e.change_type.value in ("modified", "moved") and e.before_text and e.after_text:
                    before_after = f"\n  Before: `{e.before_text[:80]}`\n  After: `{e.after_text[:80]}`"
                elif e.change_type.value == "added" and e.after_text:
                    before_after = f"\n  Content: `{e.after_text[:80]}`"
                elif e.change_type.value == "removed" and e.before_text:
                    before_after = f"\n  Content: `{e.before_text[:80]}`"
                lines.append(
                    f"- **[{e.id}]** {e.description}{before_after}  \n"
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
