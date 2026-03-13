#!/usr/bin/env python3
"""Generate markdown report from benchmark results JSON."""

import argparse
import json
from pathlib import Path
from typing import Any


def generate_markdown(report_data: dict[str, Any]) -> str:
    """Generate a markdown report from benchmark result data."""
    lines: list[str] = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- **Score**: {report_data['overall_score']}")
    lines.append(f"- **Pass^k**: {report_data['overall_pass_k']}")
    lines.append(f"- **Cost (USD)**: {report_data['total_cost_usd']}")
    lines.append(f"- **Time (ms)**: {report_data['total_time_ms']}")
    lines.append("")

    by_difficulty = report_data.get("by_difficulty", {})
    if by_difficulty:
        lines.append("## By Difficulty")
        lines.append("")
        lines.append("| Difficulty | Score |")
        lines.append("|-----------|-------|")
        for diff, score in by_difficulty.items():
            lines.append(f"| {diff} | {score} |")
        lines.append("")

    by_category = report_data.get("by_category", {})
    if by_category:
        lines.append("## By Category")
        lines.append("")
        lines.append("| Category | Score |")
        lines.append("|----------|-------|")
        for cat, score in by_category.items():
            lines.append(f"| {cat} | {score} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report from results")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: stdout)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        report_data = json.load(f)

    md = generate_markdown(report_data)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"Report written to {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
