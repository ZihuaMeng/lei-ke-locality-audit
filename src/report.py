"""
report.py
Generate outputs/audit_report.md from outputs/results.json
Milestone: M0.5
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_report(results_path: Path, output_dir: Path) -> Path:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    meta = data["meta"]
    summary = data["summary"]

    lines = []
    lines.append("# KE Locality Audit Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Mode:** `{meta['mode']}`")
    lines.append(f"\n**Timestamp (run):** {meta['timestamp']}")

    lines.append("\n---\n")
    lines.append("## Summary")

    if meta["mode"] == "NO_EDIT":
        lines.append(f"\n- Smoke test: **{summary.get('smoke_test', 'N/A')}**")
        lines.append("- Rates not computed in NO_EDIT mode.")
    else:
        lines.append(f"\n| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Efficacy rate | {summary.get('efficacy_rate', 'N/A'):.2%} |")
        lines.append(f"| Locality rate | {summary.get('locality_rate', 'N/A'):.2%} |")
        lines.append(f"| Distractor unchanged rate | {summary.get('distractor_unchanged_rate', 'N/A'):.2%} |")

    lines.append("\n---\n")
    lines.append("## Prompt Counts")
    lines.append(f"\n- Rewrite: {meta['n_rewrite']}")
    lines.append(f"- Neighborhood: {meta['n_neighborhood']}")
    lines.append(f"- Distractor: {meta['n_distractor']}")

    lines.append("\n---\n")
    lines.append("## Rewrite Results (Efficacy)")
    lines.append("\n| # | Subject | Query | C_orig | C_star | Edit Success |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(data["rewrite"], 1):
        lines.append(
            f"| {i} | {r['subject']} | {r['query']} | {r['c_orig']} | {r['c_star']} | {r['edit_success']} |"
        )

    lines.append("\n---\n")
    lines.append("## Neighborhood Results (Locality)")
    lines.append("\n| # | Query | Expected | Answer | Stable |")
    lines.append("| --- | --- | --- | --- | --- |")
    for i, r in enumerate(data["neighborhood"], 1):
        lines.append(f"| {i} | {r['query']} | {r['expected']} | {r['answer']} | {r['stable']} |")

    lines.append("\n---\n")
    lines.append("## Distractor Results (Side Effects)")
    lines.append("\n| # | Query | Expected | Answer | Unchanged |")
    lines.append("| --- | --- | --- | --- | --- |")
    for i, r in enumerate(data["distractor"], 1):
        lines.append(f"| {i} | {r['query']} | {r['expected']} | {r['answer']} | {r['unchanged']} |")

    report_path = output_dir / "audit_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("outputs/results.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    generate_report(args.results, args.output_dir)
