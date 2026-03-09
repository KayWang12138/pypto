#!/usr/bin/env python3
"""
AICore/AICPU 分阶段性能分析脚本。

阶段1：
- CTRL 前期：DEV_TASK_BUILD(0)-BEGIN
- SCHED 前期：ALLOC_THREAD_ID / INIT / CORE_HAND_SHAKE / DEV_TASK_RCV(0)
- AICore 前期：BEGIN 到 DEV_TASK_RCV_MODEL(0)（仅统计存在 DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK 的核）

阶段2：
- AICore 端到端、AIC/AIV 数量与总耗时、AICore 利用率

阶段3：
- AICore 执行完成后退出等待：DEV_TASK_WAIT_SYNC_STOP_NOTIFY(0)-DEV_TASK_ALL_CALLOP_TASK_EXEC(0)

阶段4：
- SCHED 后处理：WAIT_CORE_EXIT-DEV_TASK_SCHED_EXEC(0)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_output_dirs() -> List[Path]:
    # 兼容两种目录布局：
    # - ./output/output_xxx
    # - ./output_xxx
    dirs: List[Path] = []
    root = Path(".")
    dirs.extend([d for d in root.iterdir() if d.is_dir() and d.name.startswith("output_")])
    nested_output = root / "output"
    if nested_output.exists():
        dirs.extend([d for d in nested_output.iterdir() if d.is_dir() and d.name.startswith("output_")])
    # 去重并按目录名降序
    uniq = {str(d.resolve()): d for d in dirs}
    result = list(uniq.values())
    result.sort(key=lambda x: x.name, reverse=True)
    return result


def load_json(file_path: Path) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def to_us(cycles: float, freq: float) -> float:
    return safe_div(cycles, freq)


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


def parse_task_name(name: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^([A-Z0-9_]+)(?:\((\d+)\))?$", str(name))
    if not m:
        return str(name), None
    base = m.group(1)
    idx = int(m.group(2)) if m.group(2) is not None else None
    return base, idx


def get_task_cycle(tasks: List[Dict[str, Any]], task_name: str, idx: Optional[int] = None) -> Optional[float]:
    # 支持模糊匹配：DEV_TASK_BUILD 可匹配 DEV_TASK_BUILD(0)
    for task in tasks:
        base, num = parse_task_name(task.get("name", ""))
        if base != task_name:
            continue
        if idx is not None and num != idx:
            continue
        return float(task.get("end", 0))
    return None


def calc_duration_from_ends(start_end: Optional[float], end_end: Optional[float]) -> Optional[float]:
    if start_end is None or end_end is None:
        return None
    return end_end - start_end


def get_task_duration(
    tasks: List[Dict[str, Any]],
    start_task_name: str,
    end_task_name: str,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> Optional[float]:
    start_end = get_task_cycle(tasks, start_task_name, start_idx)
    end_end = get_task_cycle(tasks, end_task_name, end_idx)
    return calc_duration_from_ends(start_end, end_end)


def format_us(v: Optional[float], freq: float) -> str:
    if v is None:
        return "-"
    return f"{to_us(v, freq):.2f}"


def summarize_us(values: List[float], freq: float) -> List[str]:
    if not values:
        return ["0", "-", "-", "-"]
    avg = sum(values) / len(values)
    return [str(len(values)), f"{to_us(min(values), freq):.2f}", f"{to_us(avg, freq):.2f}", f"{to_us(max(values), freq):.2f}"]


def analyze_stage1(aicpu_dev_pref: List[Dict[str, Any]], aicore_wait_rows: List[Dict[str, Any]]) -> None:
    print("\n=== 第一阶段：AICPU前期准备与AICore首个任务等待 ===")
    analyze_stage1_ctrl(aicpu_dev_pref)
    analyze_stage1_sched(aicpu_dev_pref)
    analyze_stage1_aicore(aicore_wait_rows)


def analyze_stage1_ctrl(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    print("\n--- 第一阶段-CTRL AICPU ---")
    ctrl = next((x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-CTRL"), None)
    if ctrl is None:
        print("- 未找到 AICPU-CTRL 数据")
        return
    tasks = ctrl.get("tasks", [])
    freq = float(ctrl.get("freq", 0)) or 1.0
    build_dur = get_task_duration(tasks, "BEGIN", "DEV_TASK_BUILD", None, 0)
    print_table(["阶段", "耗时(us)"], [["DEV_TASK_BUILD", format_us(build_dur, freq)]])


def analyze_stage1_sched(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    print("\n--- 第一阶段-SCHED AICPU ---")
    scheds = [x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-SCHED"]
    if not scheds:
        print("- 未找到 AICPU-SCHED 数据")
        return

    scheds.sort(key=lambda x: int(x.get("blockIdx", 0)))
    rows: List[List[str]] = []

    for s in scheds:
        block_idx = int(s.get("blockIdx", -1))
        tasks = s.get("tasks", [])
        freq = float(s.get("freq", 0)) or 1.0

        alloc_dur = get_task_duration(tasks, "BEGIN", "ALLOC_THREAD_ID")
        init_dur = get_task_duration(tasks, "ALLOC_THREAD_ID", "INIT")
        handshake_dur = get_task_duration(tasks, "INIT", "CORE_HAND_SHAKE")
        dev_task_rcv = get_task_duration(tasks, "CORE_HAND_SHAKE", "DEV_TASK_RCV", None, 0)
        total_dur = None
        if alloc_dur is not None and init_dur is not None and handshake_dur is not None and dev_task_rcv is not None:
            total_dur = alloc_dur + init_dur + handshake_dur + dev_task_rcv

        rows.append(
            [
                str(block_idx),
                format_us(alloc_dur, freq),
                format_us(init_dur, freq),
                format_us(handshake_dur, freq),
                format_us(dev_task_rcv, freq),
                format_us(total_dur, freq),
            ]
        )

    print_table(
        ["blockIdx", "ALLOC_THREAD_ID(us)", "INIT(us)", "CORE_HAND_SHAKE(us)", "DEV_TASK_RCV(us)", "TOTAL(us)"],
        rows,
    )


def analyze_stage1_aicore(aicore_wait_rows: List[Dict[str, Any]]) -> None:
    print("\n--- 第一阶段-AICore 首个callop前等待 ---")
    values: List[float] = []
    ref_freq = 1.0
    for row in aicore_wait_rows:
        wait_dev_task = row.get("first_wait")
        freq = float(row.get("freq", 1.0)) or 1.0
        ref_freq = freq
        if wait_dev_task is not None:
            values.append(wait_dev_task)

    if not aicore_wait_rows:
        print("- 未找到满足条件的 AICore 数据")
        return

    stat = summarize_us(values, ref_freq)
    print_table(["统计", "count", "min(us)", "avg(us)", "max(us)"], [["AICore等待接收dev task", stat[0], stat[1], stat[2], stat[3]]])


def analyze_stage2(output_dir: Path, aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    print("\n=== 第二阶段：AICore整体执行 ===")
    file_path = output_dir / "tilefwk_L1_prof_data.json"
    if not file_path.exists():
        print("- 缺少 tilefwk_L1_prof_data.json")
        return

    data = load_json(file_path)
    if not isinstance(data, list):
        print("- tilefwk_L1_prof_data.json 格式异常")
        return

    all_starts: List[float] = []
    all_ends: List[float] = []
    aic_total = 0.0
    aiv_total = 0.0
    aic_count = 0
    aiv_count = 0

    for core in data:
        ctype = str(core.get("coreType", ""))
        if ctype not in ("AIC", "AIV"):
            continue
        if ctype == "AIC":
            aic_count += 1
        else:
            aiv_count += 1
        for task in core.get("tasks", []):
            s = float(task.get("execStart", 0))
            e = float(task.get("execEnd", 0))
            if e > s:
                dur = (e - s)
                if ctype == "AIC":
                    aic_total += dur
                else:
                    aiv_total += dur
                all_starts.append(s)
                all_ends.append(e)

    if not all_starts or not all_ends:
        print("- AICore 无有效任务执行数据")
        return

    freq = 1.0
    if aicpu_dev_pref:
        freq = float(aicpu_dev_pref[0].get("freq", 1.0)) or 1.0

    e2e_cycles = max(all_ends) - min(all_starts)
    total_core_count = aic_count + aiv_count
    total_exec_cycles = aic_total + aiv_total
    util = safe_div(total_exec_cycles, total_core_count * e2e_cycles) * 100 if total_core_count > 0 else 0.0

    print_table(
        ["指标", "值"],
        [
            ["AICore端到端耗时(us)", f"{to_us(e2e_cycles, freq):.2f}"],
            ["总执行耗时(us)", f"{to_us(total_exec_cycles, freq):.2f}"],
            ["AICore利用率", f"{util:.2f}%"],
        ],
    )
    print_table(
        ["类型", "数量", "总耗时(us)"],
        [
            ["AIC", str(aic_count), f"{to_us(aic_total, freq):.2f}"],
            ["AIV", str(aiv_count), f"{to_us(aiv_total, freq):.2f}"],
        ],
    )


def analyze_stage3(aicore_wait_rows: List[Dict[str, Any]]) -> None:
    print("\n=== 第三阶段：AICore执行后退出等待 ===")
    values: List[float] = []
    ref_freq = 1.0
    for row in aicore_wait_rows:
        gap = row.get("exit_wait")
        freq = float(row.get("freq", 1.0)) or 1.0
        ref_freq = freq
        if gap is None:
            continue
        values.append(gap)
    if not aicore_wait_rows:
        print("- 未找到 AICore 退出等待数据")
        return
    stat = summarize_us(values, ref_freq)
    print_table(["统计", "count", "min(us)", "avg(us)", "max(us)"], [["AICore退出等待", stat[0], stat[1], stat[2], stat[3]]])


def collect_aicore_wait_rows(aicpu_dev_pref: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for core in aicpu_dev_pref:
        core_type = str(core.get("coreType", ""))
        if not (core_type.startswith("SCHED") and ("-AIC" in core_type or "-AIV" in core_type)):
            continue
        tasks = core.get("tasks", [])
        # 仅保留存在首个callop等待标记的核，保持与原统计口径一致
        wait_first = get_task_cycle(tasks, "DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK", 0)
        if wait_first is None:
            continue
        first_wait = get_task_duration(tasks, "BEGIN", "DEV_TASK_RCV_MODEL", None, 0)
        exit_wait = get_task_duration(tasks, "DEV_TASK_ALL_CALLOP_TASK_EXEC", "DEV_TASK_WAIT_SYNC_STOP_NOTIFY", 0, 0)
        rows.append(
            {
                "core_type": core_type,
                "block_idx": int(core.get("blockIdx", -1)),
                "freq": float(core.get("freq", 0)) or 1.0,
                "first_wait": first_wait,
                "exit_wait": exit_wait,
            }
        )
    rows.sort(key=lambda x: x["block_idx"])
    return rows


def analyze_aicore_wait_detail(aicore_wait_rows: List[Dict[str, Any]]) -> None:
    print("\n=== 附录：AICore等待明细（首callop前 + 执行后退出） ===")
    if not aicore_wait_rows:
        print("- 无可用 AICore 明细数据")
        return
    rows: List[List[str]] = []
    for item in aicore_wait_rows:
        freq = float(item.get("freq", 1.0)) or 1.0
        rows.append(
            [
                str(item["core_type"]),
                str(item["block_idx"]),
                format_us(item.get("first_wait"), freq),
                format_us(item.get("exit_wait"), freq),
            ]
        )
    print_table(
        ["coreType", "blockIdx", "DEV_TASK_RCV_MODEL-BEGIN(us)", "WAIT_SYNC_STOP_NOTIFY-ALL_CALLOP_EXEC(us)"],
        rows,
    )


def analyze_stage4(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    print("\n=== 第四阶段：schedule aicpu等待退出时间 ===")
    scheds = [x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-SCHED"]
    if not scheds:
        print("- 未找到 AICPU-SCHED 数据")
        return
    rows: List[List[str]] = []
    values: List[float] = []
    ref_freq = 1.0
    for s in sorted(scheds, key=lambda x: int(x.get("blockIdx", 0))):
        tasks = s.get("tasks", [])
        gap = get_task_duration(tasks, "DEV_TASK_SCHED_EXEC", "WAIT_CORE_EXIT", 0, None)
        freq = float(s.get("freq", 0)) or 1.0
        ref_freq = freq
        if gap is not None:
            values.append(gap)
        rows.append([str(int(s.get("blockIdx", -1))), format_us(gap, freq)])
    print_table(["blockIdx", "WAIT_CORE_EXIT-DEV_TASK_SCHED_EXEC(us)"], rows)
    stat = summarize_us(values, ref_freq)
    print_table(
        ["统计", "count", "min(us)", "avg(us)", "max(us)"],
        [
            ["SCHED后处理", stat[0], stat[1], stat[2], stat[3]],
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AICore 等待 AICPU 性能分析脚本")
    parser.add_argument("output_dir", nargs="?", type=str, help="要分析的 output 目录，不传则自动分析最新目录")
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

    aicpu_pref_file = output_dir / "aicpu_dev_pref.json"
    if not aicpu_pref_file.exists():
        print(f"错误: {aicpu_pref_file} 不存在")
        return

    print(f"分析目录: {output_dir}")
    aicpu_dev_pref = load_json(aicpu_pref_file)
    if not isinstance(aicpu_dev_pref, list):
        print("错误: aicpu_dev_pref.json 格式异常，期望 list")
        return

    aicore_wait_rows = collect_aicore_wait_rows(aicpu_dev_pref)
    analyze_stage1(aicpu_dev_pref, aicore_wait_rows)
    analyze_stage2(output_dir, aicpu_dev_pref)
    analyze_stage3(aicore_wait_rows)
    analyze_stage4(aicpu_dev_pref)
    analyze_aicore_wait_detail(aicore_wait_rows)
    print()


if __name__ == "__main__":
    main()
