#!/usr/bin/env python3
"""add_op.py — 添加新算子到 CSV（去重 + 默认值）。

用法:
    python add_op.py --csv {csv_path} \\
        --op tanh_linear --source manual --complexity easy --category elementwise

输出 JSON:
    {"success": true, "op_name": "tanh_linear", "is_new": true}
    {"success": true, "op_name": "softmax", "is_new": false}   # 已存在，不覆盖
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .data.csv_ops import read_csv, upsert_row
else:
    module_path = Path(__file__).parent / "data" / "csv_ops.py"
    spec = importlib.util.spec_from_file_location("pypto_op_autodev_csv_ops", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load csv_ops from {module_path}")
    csv_ops = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, csv_ops)
    spec.loader.exec_module(csv_ops)
    read_csv = csv_ops.read_csv
    upsert_row = csv_ops.upsert_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--source", required=True, choices=["manual", "auto_discovered"])
    parser.add_argument("--complexity", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    # 检查算子目录是否已有完整工件（impl + test 都存在才视为已完成）
    csv_path = Path(args.csv)
    custom_dir = csv_path.parent / "custom"  # {csv_path}/../custom/
    op_dir = custom_dir / args.op
    if op_dir.exists():
        has_impl = (op_dir / f"{args.op}_impl.py").exists()
        has_test = (op_dir / f"test_{args.op}.py").exists()
        if has_impl and has_test:
            print(json.dumps({
                "success": True,
                "op_name": args.op,
                "is_new": False,
                "reason": "complete_artifacts_exist"
            }))
            return

    existing = read_csv(args.csv, op_name=args.op)
    if existing:
        print(json.dumps({"success": True, "op_name": args.op, "is_new": False}))
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    upsert_row(
        args.csv, args.op,
        source=args.source,
        status="pending",
        complexity=args.complexity,
        category=args.category,
        dev_result="",
        fail_count="0",
        fps_total="0",
        fps_confirmed="0",
        create_time=now,
        start_time="",
        end_time="",
        note="",
    )
    print(json.dumps({"success": True, "op_name": args.op, "is_new": True}))


if __name__ == "__main__":
    main()
