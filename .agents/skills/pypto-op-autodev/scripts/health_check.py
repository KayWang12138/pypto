#!/usr/bin/env python3
"""health_check.py — autodev 流程健康检查（替代 LLM 自我回顾）。

检查单次 autodev 执行的关键产物和状态，仅在有异常时输出详细信息。

用法:
    python health_check.py --op gelu --op-dir {work_dir}/gelu

退出码:
    0: 全部检查通过
    1: 存在异常（需 LLM 回顾）
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def check(name, condition, detail=""):
    return {"name": name, "ok": condition, "detail": detail}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, help="Operator name")
    parser.add_argument("--op-dir", required=True, help="Operator directory")
    parser.add_argument("--status", default="unknown", help="Final status (completed/failed)")
    parser.add_argument("--dev-result", default="", help="Dev result")
    parser.add_argument("--start-time", default="", help="Start time ISO format")
    parser.add_argument("--strategy", default="", help="Dev strategy used (orchestrator/workflow)")
    parser.add_argument("--enable-fracture", action="store_true", default=False,
                        help="Whether fracture detection was enabled")
    args = parser.parse_args()

    op_dir = Path(args.op_dir)
    results = []

    # 1. 算子目录存在
    results.append(check("op_dir_exists", op_dir.exists()))

    # 2. 开发结果文件存在（.dev_result.json 是所有策略的标准输出）
    dev_result_file = op_dir / ".dev_result.json"
    results.append(check("dev_result_exists", dev_result_file.exists(),
                         "所有策略完成后都应写入 .dev_result.json"))

    # 3. dev-log.md 存在且非空
    dev_log = op_dir / "dev-log.md"
    if dev_log.exists():
        content = dev_log.read_text(encoding="utf-8").strip()
        results.append(check("dev_log_nonempty", len(content) > 50,
                             f"dev-log.md has {len(content)} chars"))
    else:
        results.append(check("dev_log_nonempty", False, "dev-log.md not found"))

    # 4. 关键产物存在（仅 completed 时检查）
    if args.status == "completed":
        for artifact in ["spec.md", f"{args.op}_impl.py", f"test_{args.op}.py"]:
            results.append(check(f"artifact_{artifact}", (op_dir / artifact).exists()))

    # 5. 断裂点检测是否执行（仅 enable_fracture 时检查）
    if args.enable_fracture:
        fps = list(op_dir.glob("fracture-point-*.md")) + list(op_dir.glob("fracture-summary.json"))
        results.append(check("fracture_detection_ran", len(fps) > 0))

    # 6. 耗时合理性（< 90 min）
    if args.start_time:
        try:
            start = datetime.fromisoformat(args.start_time)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            else:
                start = start.astimezone(timezone.utc)
            duration_min = (datetime.now(timezone.utc) - start).total_seconds() / 60
            results.append(check("duration_reasonable", duration_min < 90,
                                 f"{duration_min:.1f} min"))
        except ValueError:
            pass

    # Output
    issues = [r for r in results if not r["ok"]]
    output = {
        "op_name": args.op,
        "status": args.status,
        "checks_total": len(results),
        "checks_passed": len(results) - len(issues),
        "issues": issues,
        "needs_review": len(issues) > 0,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
