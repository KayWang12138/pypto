#!/usr/bin/env python3
"""add_op.py — 添加新算子到 CSV（去重 + 默认值）。

用法:
    python add_op.py --csv {csv_path} \\
        --op tanh_linear --source manual --complexity easy --category elementwise \\
        --description "y = tanh(x) * x, elementwise"

输出 JSON:
    {"action": "created", "op_name": "tanh_linear"}
    {"action": "skipped", "op_name": "softmax", "reason": "already_in_csv"}
    {"action": "skipped", "op_name": "relu",    "reason": "complete_artifacts_exist"}
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from csv_ops import read_csv, upsert_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--source", required=True, choices=["manual", "auto_discovered"])
    parser.add_argument("--complexity", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--requirement", default="", help="需求文件相对路径（如 sources/gated_delta_net.md）")
    parser.add_argument("--reference", default="", help="参考实现位置（如 torch.nn.functional.softmax）")
    parser.add_argument("--description", required=True, help="一句话需求描述（必填）")
    args = parser.parse_args()

    # 检查算子目录是否已有完整工件（impl + test 都存在才视为已完成）
    csv_path = Path(args.csv)
    custom_dir = csv_path.parent / "custom"
    op_dir = custom_dir / args.op
    if op_dir.exists():
        has_impl = (op_dir / f"{args.op}_impl.py").exists()
        has_test = (op_dir / f"test_{args.op}.py").exists()
        if has_impl and has_test:
            print(json.dumps({
                "action": "skipped",
                "op_name": args.op,
                "reason": "complete_artifacts_exist",
            }))
            return

    existing = read_csv(args.csv, op_name=args.op)
    if existing:
        print(json.dumps({"action": "skipped", "op_name": args.op, "reason": "already_in_csv"}))
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    upsert_row(
        args.csv, args.op,
        source=args.source,
        status="pending",
        complexity=args.complexity,
        category=args.category,
        requirement=args.requirement,
        reference=args.reference,
        description=args.description,
        dev_result="",
        fail_count="0",
        fps_total="0",
        fps_confirmed="0",
        create_time=now,
        start_time="",
        end_time="",
        note="",
    )
    print(json.dumps({"action": "created", "op_name": args.op}))


if __name__ == "__main__":
    main()
