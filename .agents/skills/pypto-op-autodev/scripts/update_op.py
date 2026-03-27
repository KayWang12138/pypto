#!/usr/bin/env python3
"""update_op.py — 三种模式的 CSV 状态更新。

模式 1: --op {name} --status {status} [--dev-result X] [--fps-total N] [--fps-confirmed M]
模式 2: --op {name} --reset
模式 3: --reset-stale --timeout-hours 6

退出码（仅模式 3）: 0=无活跃, 1=有活跃 in_progress
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.csv_ops import read_csv, upsert_row

VALID_STATUSES = {"pending", "in_progress", "completed", "failed"}


def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def mode1(args):
    existing = read_csv(args.csv, op_name=args.op)
    if not existing:
        print(json.dumps({"error": f"op '{args.op}' not found"}), file=sys.stderr)
        sys.exit(2)

    row = existing[0]
    fields = {"status": args.status}

    if args.status == "in_progress":
        fields["start_time"] = now_str()
        fields["end_time"] = ""

    if args.status in ("completed", "failed"):
        fields["end_time"] = now_str()
        if args.dev_result is not None:
            fields["dev_result"] = args.dev_result
        if args.fps_total is not None:
            fields["fps_total"] = str(args.fps_total)
        if args.fps_confirmed is not None:
            fields["fps_confirmed"] = str(args.fps_confirmed)
        if args.status == "failed":
            fields["fail_count"] = str(int(row.get("fail_count") or 0) + 1)
        if args.blocked_stage is not None:
            fields["blocked_stage"] = str(args.blocked_stage)
        # 自动计算开发耗时
        start = parse_dt(row.get("start_time"))
        if start:
            duration = (datetime.now(timezone.utc) - start).total_seconds() / 60
            fields["duration_min"] = str(round(duration, 1))

    result = upsert_row(args.csv, args.op, **fields)
    print(json.dumps(result))


def mode2(args):
    existing = read_csv(args.csv, op_name=args.op)
    if not existing:
        print(json.dumps({"error": f"op '{args.op}' not found"}), file=sys.stderr)
        sys.exit(2)
    result = upsert_row(
        args.csv, args.op,
        status="pending",
        dev_result="",
        start_time="",
        end_time="",
    )
    print(json.dumps(result))


def mode3(args):
    timeout = timedelta(hours=args.timeout_hours)
    now = datetime.now(timezone.utc)

    all_rows = read_csv(args.csv, status="in_progress")
    reset_ops = []
    active_op = None

    for row in all_rows:
        start = parse_dt(row.get("start_time"))
        if start is None or (now - start) > timeout:
            upsert_row(args.csv, row["op_name"], status="pending", start_time="", end_time="")
            reset_ops.append(row["op_name"])
        else:
            if active_op is None:
                active_op = row["op_name"]

    has_active = active_op is not None
    print(json.dumps({
        "reset_ops": reset_ops,
        "has_active": has_active,
        "active_op": active_op,
    }))
    sys.exit(1 if has_active else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--op")
    parser.add_argument("--status", choices=list(VALID_STATUSES))
    parser.add_argument("--dev-result")
    parser.add_argument("--fps-total", type=int)
    parser.add_argument("--fps-confirmed", type=int)
    parser.add_argument("--blocked-stage", type=int, help="Stage number where blocked (1-7)")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--reset-stale", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    args = parser.parse_args()

    if args.reset_stale:
        mode3(args)
    elif args.reset and args.op:
        mode2(args)
    elif args.op and args.status:
        mode1(args)
    else:
        parser.error("必须指定 --reset-stale，或 --op --reset，或 --op --status")


if __name__ == "__main__":
    main()
