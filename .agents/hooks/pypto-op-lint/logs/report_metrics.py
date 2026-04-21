#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
"""Summarize pypto-op-lint metric events into JSON and Markdown reports.

The lint runtime appends one JSON event per line to `lint_events.jsonl`.
This helper rolls those events up into:

- `summary_latest.json` -- machine-readable totals (consumed by CI dashboards)
- `summary_latest.md`   -- human-readable breakdown

Tests override `BASE`, `EVENTS`, `OUT_JSON`, `OUT_MD` directly on the module,
then call `load_events()` / `build_summary()` / `write_outputs()`.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

BASE: Path = Path(__file__).resolve().parent
EVENTS: Path = BASE / "lint_events.jsonl"
OUT_JSON: Path = BASE / "summary_latest.json"
OUT_MD: Path = BASE / "summary_latest.md"


def load_events() -> list[dict[str, Any]]:
    """Read the JSONL events file; return an empty list if missing."""
    target = Path(EVENTS)
    if not target.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def build_summary(events: list[dict]) -> dict[str, Any]:
    total_rule_checks = 0
    total_gate_blocks = 0
    status_counts: Counter = Counter()
    per_rule: dict[str, dict[str, int]] = {}
    for event in events:
        kind = event.get("event_type")
        if kind == "rule_check":
            total_rule_checks += 1
            status = event.get("status", "UNKNOWN")
            status_counts[status] += 1
            rule_id = event.get("rule_id", "")
            if rule_id:
                slot = per_rule.setdefault(rule_id, {})
                slot[status] = slot.get(status, 0) + 1
        elif kind == "gate_decision" and event.get("blocked"):
            total_gate_blocks += 1
        elif kind == "gate_decision":
            # gate_decision without "blocked": ignored
            continue
    return {
        "total_rule_checks": total_rule_checks,
        "total_gate_blocks": total_gate_blocks,
        "rule_status_counts": dict(status_counts),
        "per_rule": per_rule,
    }


def write_outputs(summary: dict[str, Any]) -> None:
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PyPTO op-lint metrics",
        "",
        f"- Rule checks: **{summary['total_rule_checks']}**",
        f"- Gate blocks: **{summary['total_gate_blocks']}**",
        "",
        "## Status counts",
    ]
    for status, count in sorted(summary.get("rule_status_counts", {}).items()):
        lines.append(f"- {status}: {count}")
    if summary["per_rule"]:
        lines.append("")
        lines.append("## Per-rule breakdown")
        for rule_id, breakdown in sorted(summary["per_rule"].items()):
            lines.append(f"- **{rule_id}**: " + ", ".join(f"{k}={v}" for k, v in breakdown.items()))
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    write_outputs(build_summary(load_events()))
