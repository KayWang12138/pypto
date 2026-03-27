#!/usr/bin/env python3
"""get_progress.py — 聚合统计当前开发进度。

用法:
    python get_progress.py --csv autodev/scan_results.csv
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

if __package__:
    from .data.csv_ops import read_csv
else:
    module_path = Path(__file__).parent / "data" / "csv_ops.py"
    spec = importlib.util.spec_from_file_location("pypto_op_autodev_csv_ops", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load csv_ops from {module_path}")
    csv_ops = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, csv_ops)
    spec.loader.exec_module(csv_ops)
    read_csv = csv_ops.read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    rows = read_csv(args.csv)
    total = len(rows)

    by_status = {}
    by_category = {}
    fps_total = 0
    fps_confirmed = 0
    completed = 0
    terminal = 0

    for r in rows:
        s = r.get("status", "")
        by_status[s] = by_status.get(s, 0) + 1

        cat = r.get("category", "other") or "other"
        by_category[cat] = by_category.get(cat, 0) + 1

        fps_total += int(r.get("fps_total") or 0)
        fps_confirmed += int(r.get("fps_confirmed") or 0)

        if s == "completed":
            completed += 1
            terminal += 1
        elif s == "failed":
            terminal += 1

    success_rate = completed / terminal if terminal > 0 else 0

    print(json.dumps({
        "total": total,
        "by_status": by_status,
        "by_category": by_category,
        "success_rate": round(success_rate, 4),
        "fps_total": fps_total,
        "fps_confirmed": fps_confirmed,
    }))


if __name__ == "__main__":
    main()
