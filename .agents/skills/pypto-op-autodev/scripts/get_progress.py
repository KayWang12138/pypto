#!/usr/bin/env python3
"""get_progress.py — 聚合统计当前开发进度，支持 JSON 和 Markdown 输出。

用法:
    python get_progress.py --csv autodev/scan_results.csv
    python get_progress.py --csv autodev/scan_results.csv --format markdown
    python get_progress.py --csv autodev/scan_results.csv --format markdown --output autodev/PROGRESS.md
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


def collect_stats(rows):
    total = len(rows)

    by_status = {}
    by_category = {}
    by_result = {}
    fps_total = 0
    fps_confirmed = 0
    completed = 0
    terminal = 0
    total_duration = 0
    duration_count = 0

    for r in rows:
        s = r.get("status", "")
        by_status[s] = by_status.get(s, 0) + 1

        cat = r.get("category", "other") or "other"
        by_category[cat] = by_category.get(cat, 0) + 1

        dev_result = r.get("dev_result", "")
        if dev_result:
            by_result[dev_result] = by_result.get(dev_result, 0) + 1

        fps_total += int(r.get("fps_total") or 0)
        fps_confirmed += int(r.get("fps_confirmed") or 0)

        dur = r.get("duration_min", "")
        if dur:
            try:
                total_duration += float(dur)
                duration_count += 1
            except ValueError:
                pass

        if s == "completed":
            completed += 1
            terminal += 1
        elif s == "failed":
            terminal += 1

    success_rate = completed / terminal if terminal > 0 else 0
    avg_duration = round(total_duration / duration_count, 1) if duration_count > 0 else 0

    return {
        "total": total,
        "by_status": by_status,
        "by_category": by_category,
        "by_result": by_result,
        "success_rate": round(success_rate, 4),
        "avg_duration_min": avg_duration,
        "fps_total": fps_total,
        "fps_confirmed": fps_confirmed,
    }


def format_markdown(rows, stats):
    lines = []
    lines.append("# PyPTO 算子开发进度报告\n")

    by_status = stats["by_status"]
    completed = by_status.get("completed", 0)
    pending = by_status.get("pending", 0)
    failed = by_status.get("failed", 0)
    in_progress = by_status.get("in_progress", 0)

    lines.append("## 概览\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|------|")
    lines.append(f"| 总算子数 | {stats['total']} |")
    lines.append(f"| 已完成 | {completed} |")
    lines.append(f"| 进行中 | {in_progress} |")
    lines.append(f"| 待开发 | {pending} |")
    lines.append(f"| 失败 | {failed} |")
    lines.append(f"| 成功率 | {stats['success_rate']:.1%} |")
    if stats["avg_duration_min"] > 0:
        lines.append(f"| 平均耗时 | {stats['avg_duration_min']} min |")
    lines.append(f"| 断裂点总数 | {stats['fps_total']} |")
    lines.append(f"| 高置信断裂点 | {stats['fps_confirmed']} |")
    lines.append("")

    lines.append("## 类别覆盖\n")
    lines.append("| 类别 | 数量 | 已完成 |")
    lines.append("|------|------|--------|")
    cat_completed = {}
    for r in rows:
        if r.get("status") == "completed":
            cat = r.get("category", "other") or "other"
            cat_completed[cat] = cat_completed.get(cat, 0) + 1
    for cat, count in sorted(stats["by_category"].items()):
        done = cat_completed.get(cat, 0)
        lines.append(f"| {cat} | {count} | {done} |")
    lines.append("")

    lines.append("## 算子详情\n")
    lines.append("| 算子 | 状态 | 类别 | 复杂度 | 结果 | 失败次数 | 断裂点 |")
    lines.append("|------|------|------|--------|------|----------|--------|")
    for r in sorted(rows, key=lambda x: (
        {"completed": 0, "in_progress": 1, "pending": 2, "failed": 3}.get(x.get("status", ""), 4),
        x.get("op_name", ""),
    )):
        status_icon = {
            "completed": "done", "in_progress": "wip",
            "pending": "todo", "failed": "fail",
        }.get(r.get("status", ""), r.get("status", ""))
        fps = int(r.get("fps_total") or 0)
        fps_str = str(fps) if fps > 0 else "-"
        fail_count = int(r.get("fail_count") or 0)
        fail_str = str(fail_count) if fail_count > 0 else "-"
        lines.append(
            f"| {r.get('op_name', '')} "
            f"| {status_icon} "
            f"| {r.get('category', '')} "
            f"| {r.get('complexity', '')} "
            f"| {r.get('dev_result', '-') or '-'} "
            f"| {fail_str} "
            f"| {fps_str} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", help="Write output to file instead of stdout")
    args = parser.parse_args()

    rows = read_csv(args.csv)
    stats = collect_stats(rows)

    if args.format == "markdown":
        output = format_markdown(rows, stats)
    else:
        output = json.dumps(stats)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(json.dumps({"written_to": args.output, "total": stats["total"]}))
    else:
        print(output)


if __name__ == "__main__":
    main()
