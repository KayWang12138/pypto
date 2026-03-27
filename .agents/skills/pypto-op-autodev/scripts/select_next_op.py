#!/usr/bin/env python3
"""select_next_op.py — 优先级分数计算 + 算子选择。

用法:
    python select_next_op.py --csv autodev/scan_results.csv [--dry-run]

退出码:
    0: 成功选择算子（score >= DISCOVERY_THRESHOLD）
    1: 需要触发 discover（无候选 或 最高分 < DISCOVERY_THRESHOLD）
    2: 参数错误
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
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

DISCOVERY_THRESHOLD = 25

COMPLEXITY_PENALTY = {"easy": 0, "medium": 15, "hard": 30}


def now():
    return datetime.now(timezone.utc)


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def score(row, recent_categories):
    base = 100

    c_pen = COMPLEXITY_PENALTY.get(row.get("complexity", "easy"), 0)

    fail_count = int(row.get("fail_count") or 0)
    f_pen = fail_count * 20

    decay = 0
    if row.get("status") == "failed":
        end = parse_dt(row.get("end_time"))
        if end:
            hours_since = (now() - end).total_seconds() / 3600
            decay = max(0, 30 - hours_since)

    create = parse_dt(row.get("create_time"))
    if create:
        days = (now() - create).total_seconds() / 86400
        age_b = min(20, days * 2)
    else:
        age_b = 0

    cat_pen = 10 if row.get("category") in recent_categories else 0

    return base - c_pen - f_pen - decay + age_b - cat_pen


def is_candidate(row):
    """检查是否为有效候选。

    排除条件：
    1. 状态不是 pending 或 failed
    2. dev_result 为 NOT_IMPLEMENTABLE
    """
    status = row.get("status", "")
    dev_result = row.get("dev_result", "")
    if status not in ("pending", "failed"):
        return False
    if dev_result == "NOT_IMPLEMENTABLE":
        return False

    return True


def get_recent_categories(all_rows, n=3):
    completed = [r for r in all_rows if r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("end_time", ""), reverse=True)
    return {r["category"] for r in completed[:n]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_rows = read_csv(args.csv)
    recent_cats = get_recent_categories(all_rows)
    candidates = [r for r in all_rows if is_candidate(r)]

    if not candidates:
        print(json.dumps({"action": "trigger_discovery", "selected": None, "need_discovery": True}))
        sys.exit(1)

    scored = [(score(r, recent_cats), r) for r in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_row = scored[0]

    if best_score < DISCOVERY_THRESHOLD:
        print(json.dumps({"action": "trigger_discovery", "selected": None, "need_discovery": True}))
        sys.exit(1)

    result = {
        "action": "select_from_csv",
        "selected": {
            "op_name": best_row["op_name"],
            "complexity": best_row.get("complexity", ""),
            "category": best_row.get("category", ""),
            "fail_count": int(best_row.get("fail_count") or 0),
            "score": round(best_score, 2),
        },
        "need_discovery": False,
    }
    print(json.dumps(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
