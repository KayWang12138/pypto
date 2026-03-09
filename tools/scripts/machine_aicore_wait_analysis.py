#!/usr/bin/env python3
"""
AICore 等待 AICPU 时间分析脚本。

功能:
1) 读取 output 目录中的 aicpu_dev_pref.json，计算并打印:
   - CTRL-AICPU 总耗时: EXIT - BEGIN
   - 各 SCHED-AICPU-blockIdx 总耗时: WAIT_CORE_EXIT - BEGIN
   - AICore 等待 CTRL-AICPU / 对应 SCHED-AICPU 的时间
2) 读取 tilefwk_L1_prof_data.json，打印 AICore 执行摘要

使用示例:
    python3 machine_aicore_wait_analysis.py
    python3 machine_aicore_wait_analysis.py output/output_xxx
    python3 machine_aicore_wait_analysis.py --detailed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_output_dirs(base_dir: str = "output") -> List[Path]:
    output_base = Path(base_dir)
    if not output_base.exists():
        return []
    dirs = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("output_")]
    dirs.sort(key=lambda x: x.name, reverse=True)
    return dirs


def load_json(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def to_us(cycles: float, freq: float) -> float:
    # 当前 trace 里 time_convert_denominator 使用 freq，脚本转换逻辑为 cycle / freq。
    return safe_div(cycles, freq)


def get_task_cycle(tasks: List[Dict[str, Any]], name: str) -> Optional[float]:
    for task in tasks:
        if task.get("name") == name:
            return float(task.get("end", 0))
    return None


def parse_sched_idx_from_core_type(core_type: str) -> Optional[int]:
    # 形如 SCHED1-AIC / SCHED2-AIV
    m = re.match(r"^SCHED(\d+)-", core_type)
    if not m:
        return None
    return int(m.group(1))


def print_wait_stats(label: str, values: List[float], freq: float) -> None:
    if not values:
        print(f"- {label}: 无可用数据")
        return
    avg_cycles = sum(values) / len(values)
    print(
        f"- {label}: count={len(values)}, min={min(values):.0f} cyc ({to_us(min(values), freq):.2f} us), "
        f"avg={avg_cycles:.2f} cyc ({to_us(avg_cycles, freq):.2f} us), "
        f"max={max(values):.0f} cyc ({to_us(max(values), freq):.2f} us)"
    )


def wait_stats_row(label: str, values: List[float], freq: float) -> List[str]:
    if not values:
        return [label, "0", "-", "-", "-"]
    avg_cycles = sum(values) / len(values)
    return [
        label,
        str(len(values)),
        f"{min(values):.0f}/{to_us(min(values), freq):.2f}",
        f"{avg_cycles:.2f}/{to_us(avg_cycles, freq):.2f}",
        f"{max(values):.0f}/{to_us(max(values), freq):.2f}",
    ]


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    print(fmt_row(headers))
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        print(fmt_row(row))


def collect_tilefwk_exec_cycles(
    output_dir: Path,
) -> Tuple[Dict[Tuple[str, int], float], Dict[str, Dict[str, Any]]]:
    file_path = output_dir / "tilefwk_L1_prof_data.json"
    if not file_path.exists():
        return {}, {}
    data = load_json(file_path)
    if not isinstance(data, list):
        return {}, {}

    exec_cycles: Dict[Tuple[str, int], float] = {}
    per_type: Dict[str, Dict[str, Any]] = {}
    for core in data:
        core_type = str(core.get("coreType", "UNKNOWN"))
        if core_type not in ("AIC", "AIV"):
            continue
        block_idx = int(core.get("blockIdx", -1))
        total = 0.0
        for task in core.get("tasks", []):
            s = float(task.get("execStart", 0))
            e = float(task.get("execEnd", 0))
            if e > s:
                total += e - s
                per_type.setdefault(core_type, {"core_entries": 0, "task_count": 0, "durations": []})
                per_type[core_type]["task_count"] += 1
                per_type[core_type]["durations"].append(e - s)
        if block_idx >= 0:
            exec_cycles[(core_type, block_idx)] = total
        per_type.setdefault(core_type, {"core_entries": 0, "task_count": 0, "durations": []})
        per_type[core_type]["core_entries"] += 1
    return exec_cycles, per_type


def analyze_wait_time(
    aicpu_dev_pref: List[Dict[str, Any]],
    detailed: bool,
) -> None:
    ctrl_begin: Optional[float] = None
    ctrl_exit: Optional[float] = None
    ctrl_freq: Optional[float] = None
    sched_begin: Dict[int, float] = {}
    sched_exit: Dict[int, float] = {}
    sched_freq: Dict[int, float] = {}
    aicore_records: List[Dict[str, Any]] = []

    for core in aicpu_dev_pref:
        core_type = str(core.get("coreType", ""))
        block_idx = int(core.get("blockIdx", -1))
        freq = float(core.get("freq", 0))
        tasks = core.get("tasks", [])
        begin_cycle = get_task_cycle(tasks, "BEGIN")
        if begin_cycle is None:
            continue

        if core_type == "AICPU-CTRL":
            ctrl_begin = begin_cycle
            ctrl_exit = get_task_cycle(tasks, "EXIT")
            ctrl_freq = freq
        elif core_type == "AICPU-SCHED":
            sched_begin[block_idx] = begin_cycle
            sched_exit[block_idx] = get_task_cycle(tasks, "WAIT_CORE_EXIT")
            sched_freq[block_idx] = freq
        elif core_type.startswith("SCHED") and ("-AIC" in core_type or "-AIV" in core_type):
            wait_exit_notify = get_task_cycle(tasks, "WAIT_EXIT_NOTIFY")
            aicore_records.append(
                {
                    "block_idx": block_idx,
                    "core_type": core_type,
                    "freq": freq,
                    "begin": begin_cycle,
                    "wait_exit_notify": wait_exit_notify,
                    "sched_idx": parse_sched_idx_from_core_type(core_type),
                }
            )

    print("\n=== AICPU 执行时间分析 ===")
    exec_rows: List[List[str]] = []

    if ctrl_begin is not None and ctrl_exit is not None:
        ctrl_dur = ctrl_exit - ctrl_begin
        ctrl_us = to_us(ctrl_dur, ctrl_freq if ctrl_freq else 1.0)
        exec_rows.append(
            [
                "CTRL-AICPU",
                "-",
                f"{ctrl_us:.2f}",
            ]
        )

    for idx in sorted(sched_begin.keys()):
        begin = sched_begin[idx]
        end = sched_exit.get(idx)
        freq = sched_freq.get(idx, 1.0) or 1.0
        if end is None:
            exec_rows.append(
                [f"SCHED-AICPU-{idx}", str(idx), "-"]
            )
            continue
        dur = end - begin
        exec_rows.append(
            [
                f"SCHED-AICPU-{idx}",
                str(idx),
                f"{to_us(dur, freq):.2f}",
            ]
        )

    if exec_rows:
        print_table(
            ["AICPU核心", "blockIdx", "耗时(us)"],
            exec_rows,
        )
    else:
        print("- 无可用 AICPU 执行数据")

    print("\n=== AICore 等待 AICPU 时间分析 ===")
    wait_ctrl: List[float] = []
    wait_sched: List[float] = []
    per_core_rows: List[Dict[str, Any]] = []

    for item in aicore_records:
        aicore_begin = float(item["begin"])
        freq = float(item["freq"]) if item["freq"] > 0 else 1.0
        sched_idx = item["sched_idx"]

        w_ctrl = None
        if ctrl_begin is not None:
            w_ctrl = aicore_begin - ctrl_begin
            wait_ctrl.append(w_ctrl)

        w_sched = None
        if sched_idx is not None and sched_idx in sched_begin:
            w_sched = aicore_begin - sched_begin[sched_idx]
            wait_sched.append(w_sched)

        per_core_rows.append(
            {
                "block_idx": int(item["block_idx"]),
                "core_type": str(item["core_type"]),
                "wait_ctrl": w_ctrl,
                "wait_sched": w_sched,
                "freq": freq,
                "begin": aicore_begin,
                "wait_exit_notify": item.get("wait_exit_notify"),
                "sched_idx": sched_idx,
            }
        )

    ref_freq = ctrl_freq if ctrl_freq else (aicore_records[0]["freq"] if aicore_records else 1.0)
    ref_freq = float(ref_freq) if ref_freq else 1.0
    print_table(
        ["指标", "count", "min(cyc/us)", "avg(cyc/us)", "max(cyc/us)"],
        [
            ["AICore 核心数", str(len(aicore_records)), "-", "-", "-"],
            ["CTRL BEGIN 可用", "1" if ctrl_begin is not None else "0", "-", "-", "-"],
            ["SCHED BEGIN 数量", str(len(sched_begin)), "-", "-", "-"],
            wait_stats_row("等待 CTRL-AICPU", wait_ctrl, ref_freq),
            wait_stats_row("等待 SCHED-AICPU", wait_sched, ref_freq),
        ],
    )

    if detailed:
        print("\n--- 按 AICore 明细 ---")
        rows: List[List[str]] = []
        for row in sorted(per_core_rows, key=lambda x: x["block_idx"]):
            block_idx = row["block_idx"]
            core_type = row["core_type"]
            w_ctrl = row["wait_ctrl"]
            w_sched = row["wait_sched"]
            freq = row["freq"]

            ctrl_str = "-" if w_ctrl is None else f"{w_ctrl:.0f}/{to_us(w_ctrl, freq):.2f}"
            sched_str = "-" if w_sched is None else f"{w_sched:.0f}/{to_us(w_sched, freq):.2f}"
            wait_exit_notify = row.get("wait_exit_notify")
            exec_cyc = None
            if wait_exit_notify is not None:
                exec_cyc = float(wait_exit_notify) - float(row["begin"])
            exec_str = "-" if exec_cyc is None else f"{exec_cyc:.0f}/{to_us(exec_cyc, freq):.2f}"

            rows.append(
                [
                    str(block_idx),
                    core_type,
                    ctrl_str,
                    sched_str,
                    exec_str,
                ]
            )
        print_table(
            ["blockIdx", "coreType", "wait_ctrl(cyc/us)", "wait_sched(cyc/us)", "exec(cyc/us)"],
            rows,
        )


def analyze_l1_prof(output_dir: Path, per_type: Dict[str, Dict[str, Any]]) -> None:
    file_path = output_dir / "tilefwk_L1_prof_data.json"
    print("\n=== tilefwk_L1_prof_data.json (AICore 执行摘要) ===")
    if not file_path.exists():
        print("- 文件不存在")
        return

    rows: List[List[str]] = []
    for core_type in ("AIC", "AIV"):
        item = per_type.get(core_type)
        if not item:
            rows.append([core_type, "0", "0", "-", "-", "-", "-"])
            continue
        durs = item["durations"]
        if durs:
            min_cyc = f"{min(durs):.0f}"
            avg_cyc = f"{sum(durs) / len(durs):.2f}"
            max_cyc = f"{max(durs):.0f}"
            total_cyc = f"{sum(durs):.0f}"
        else:
            min_cyc = avg_cyc = max_cyc = total_cyc = "-"
        rows.append(
            [
                core_type,
                str(item["core_entries"]),
                str(item["task_count"]),
                min_cyc,
                avg_cyc,
                max_cyc,
                total_cyc,
            ]
        )

    print_table(
        ["coreType", "核心条目数", "任务数", "min(cyc)", "avg(cyc)", "max(cyc)", "total(cyc)"],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AICore 等待 AICPU 时间分析")
    parser.add_argument("output_dir", nargs="?", type=str, help="要分析的 output 目录，不传则自动选最新 output_*")
    parser.add_argument("--detailed", action="store_true", help="打印按 AICore 的等待明细")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dirs = get_output_dirs()
        if not output_dirs:
            print("错误: 没有找到 output_* 目录")
            return
        output_dir = output_dirs[0]

    if not output_dir.exists():
        print(f"错误: 目录不存在: {output_dir}")
        return

    print(f"分析目录: {output_dir}")
    aicpu_pref_file = output_dir / "aicpu_dev_pref.json"
    if not aicpu_pref_file.exists():
        print("错误: 缺少 aicpu_dev_pref.json，无法计算 AICore 等待 AICPU 时间")
        return

    aicpu_dev_pref = load_json(aicpu_pref_file)
    _, per_type = collect_tilefwk_exec_cycles(output_dir)
    analyze_wait_time(aicpu_dev_pref, args.detailed)
    analyze_l1_prof(output_dir, per_type)


if __name__ == "__main__":
    main()
