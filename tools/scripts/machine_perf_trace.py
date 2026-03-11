#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import json
import re
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def parse_log_file(log_file_path):
    all_blocks_data = []
    current_block_aicpu = []
    current_block_aicore = []
    in_perf_trace_block = False
    block_start_line = 0

    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if "Begin dump machine perf trace:" in line:
                if in_perf_trace_block:
                    print(f"Error: Nested perf trace block at line {line_num}. Block started at line {block_start_line} is not properly closed.")
                    return {"error": f"Nested perf trace block at line {line_num}"}

                in_perf_trace_block = True
                block_start_line = line_num
                current_block_aicpu = []
                current_block_aicore = []
                print(f"Found perf trace block start at line {line_num}")
                continue

            if "Finish dump machine perf trace." in line:
                if not in_perf_trace_block:
                    print(f"Error: Finish without matching begin at line {line_num}")
                    return {"error": f"Finish without matching begin at line {line_num}"}

                in_perf_trace_block = False
                if current_block_aicpu and current_block_aicore:
                    block_json_str = ''.join(current_block_aicpu) + ',' + ''.join(current_block_aicore)
                    block_json_str = block_json_str.replace(",]", "]")
                    all_blocks_data.append(block_json_str)
                    print(f"Successfully parsed performance trace block from line {block_start_line} to {line_num}")
                else:
                    print(f"Warning: Empty performance trace block from line {block_start_line} to {line_num}")

                current_block_aicpu = []
                current_block_aicore = []
                continue

            if in_perf_trace_block:
                if "tile_fwk aicpu prof:" in line:
                    match = re.search(r'tile_fwk aicpu prof:(.*)', line)
                    if match:
                        content = match.group(1).strip()
                        if content.endswith('"'):
                            content = content[:-1]
                        current_block_aicpu.append(content)
                elif "tile_fwk aicore prof:" in line:
                    match = re.search(r'tile_fwk aicore prof:(.*)', line)
                    if match:
                        content = match.group(1).strip()
                        if content.endswith('"'):
                            content = content[:-1]
                        current_block_aicore.append(content)
            else:
                if "tile_fwk aicpu prof:" in line or "tile_fwk aicore prof:" in line:
                    print(f"Warning: Ignoring prof data outside of perf trace block at line {line_num}")

    if in_perf_trace_block:
        print(f"Error: Unclosed perf trace block started at line {block_start_line}. Discarding incomplete block data.")
        return {"error": f"Unclosed perf trace block started at line {block_start_line}"}

    if all_blocks_data:
        full_json_str = '[' + ','.join(all_blocks_data) + ']'
        try:
            parsed_data = json.loads(full_json_str)
            print(f"Successfully parsed {len(all_blocks_data)} performance trace blocks")
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw JSON string: {full_json_str[:500]}...")  # Print first 500 chars for debugging
            return {"error": str(e), "raw_blocks": all_blocks_data}
    else:
        print("No valid performance trace blocks found in log file")
        return []


def save_json(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        if isinstance(data, dict) and "raw_aicpu" in data:
            file.write('[' + data["raw_aicpu"] + ',' + data["raw_aicore"] + ']')
        else:
            json.dump(data, file, indent=2, ensure_ascii=False)


def convert_to_perfetto_format(input_json: List[Dict]) -> List[Dict]:
    trace_events = []
    thread_id = 0
    for block in input_json:
        block_idx = block.get("blockIdx", 0)
        core_type = block.get("coreType", "UNKNOWN")
        freq = block.get("freq", 0)
        thread_name = f"{core_type}-{block_idx}"
        trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": 0,
            "tid": thread_id,
            "args": {
                "name": thread_name
            }
        })

        tasks = block.get("tasks", [])
        sorted_tasks = sorted(tasks, key=lambda x: x.get("end", 0))

        # 处理相同end_time的情况
        adjusted_tasks = []
        prev_end = None
        for task in sorted_tasks:
            task_copy = task.copy()
            current_end = task_copy.get("end", 0)

            # 如果当前end与上一个相同，则递增1
            if prev_end is not None and current_end == prev_end:
                task_copy["end"] = prev_end + 1
                current_end = prev_end + 1

            adjusted_tasks.append(task_copy)
            prev_end = current_end

        prev_end = None
        for task in adjusted_tasks:
            task_name = task.get("name", "UNKNOWN")
            end_time = task.get("end", 0)
            if prev_end is None:
                start_time = end_time - 1
            else:
                start_time = prev_end
            ts = start_time / freq
            dur = (end_time - start_time) / freq
            if dur <= 0:
                dur = 1 / freq
            perfetto_event = {
                "name": f"{task_name}",
                "cat": core_type,
                "ph": "X",
                "ts": ts,
                "dur": dur,
                "pid": 0,
                "tid": thread_id,
                "freq": freq
            }
            trace_events.append(perfetto_event)
            prev_end = end_time
        thread_id += 1
    trace_events_pypto = {"traceEvents": trace_events}
    return trace_events_pypto


def parse_log_command(input_file, output_file):
    parsed_data = parse_log_file(input_file)
    save_json(parsed_data, output_file)
    print(f"Parsing completed, result saved to: {output_file}")


def gen_perfetto_command(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        perfetto_data = convert_to_perfetto_format(input_data)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(perfetto_data, f, ensure_ascii=False, indent=2)
        print(f"Success to generate perfetto file: {output_file}")

        complete_events = [e for e in perfetto_data["traceEvents"] if e.get('ph') == 'X']
        metadata_events = [e for e in perfetto_data["traceEvents"] if e.get('ph') == 'M']
        print(f"Process {len(complete_events)} task events, {len(metadata_events)} meta data events")

        print("\ninfo: upload this json file to https://ui.perfetto.dev/")
    except FileNotFoundError:
        print(f"error: cannot find input file {input_file}")
    except json.JSONDecodeError:
        print(f"error: input file {input_file} is not valid json format")
    except Exception as e:
        print(f"process exception info: {str(e)}")


def gen_perfetto_example():
    sample_data = [
        {"blockIdx": 0, "coreType": "AICPU-SCHED", "freq": 50, "tasks": [
            {"name": "BEGIN", "end": 5236903326282},
            {"name": "ALLOC_THREAD_ID", "end": 5236903326381},
            {"name": "INIT", "end": 5236903329385},
            {"name": "HAND_SHAKE", "end": 5236903330212},
            {"name": "WAIT_ALL_TASK_FIN", "end": 5236903331821},
            {"name": "SEND_STOP", "end": 5236903332087},
            {"name": "EXIT", "end": 5236903332854}
        ]},
        {"blockIdx": 1, "coreType": "AICPU-SCHED", "freq": 50, "tasks": [
            {"name": "BEGIN", "end": 5236903326282},
            {"name": "ALLOC_THREAD_ID", "end": 5236903326383},
            {"name": "INIT", "end": 5236903329389},
            {"name": "HAND_SHAKE", "end": 5236903330219},
            {"name": "WAIT_SEND_FIRST_TASK", "end": 5236903331446},
            {"name": "WAIT_ALL_TASK_FIN", "end": 5236903331829},
            {"name": "SEND_STOP", "end": 5236903332102},
            {"name": "EXIT", "end": 5236903333765}
        ]}
    ]

    result = convert_to_perfetto_format(sample_data)
    with open('perfetto_output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("example have saved to perfetto_output.json")
    print("You can check it by upload this file to https://ui.perfetto.dev/")


def get_output_dirs() -> List[Path]:
    dirs: List[Path] = []
    root = Path(".")
    dirs.extend([d for d in root.iterdir() if d.is_dir() and d.name.startswith("output_")])
    nested_output = root / "output"
    if nested_output.exists():
        dirs.extend([d for d in nested_output.iterdir() if d.is_dir() and d.name.startswith("output_")])
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


def display_width(text: str) -> int:
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
    return width


def pad_cell(text: str, width: int) -> str:
    pad = max(width - display_width(text), 0)
    return text + (" " * pad)


def render_table_lines(headers: List[str], rows: List[List[str]]) -> List[str]:
    line_rows: List[List[str]] = [[str(x) for x in headers]]
    line_rows.extend([[str(x) for x in row] for row in rows])

    widths = [display_width(h) for h in headers]
    for row in line_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], display_width(cell))

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(pad_cell(cell, widths[i]) for i, cell in enumerate(row)) + " |"

    def fmt_border() -> str:
        return "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"

    lines = [fmt_border(), fmt_row(line_rows[0]), fmt_border()]
    for row in line_rows[1:]:
        lines.append(fmt_row(row))
    lines.append(fmt_border())
    return lines


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    for line in render_table_lines(headers, rows):
        print(line)


def print_tables_side_by_side(
    left_headers: List[str],
    left_rows: List[List[str]],
    right_headers: List[str],
    right_rows: List[List[str]],
    gap: int = 4,
) -> None:
    left_lines = render_table_lines(left_headers, left_rows)
    right_lines = render_table_lines(right_headers, right_rows)
    left_width = max(display_width(line) for line in left_lines) if left_lines else 0
    total_lines = max(len(left_lines), len(right_lines))

    for i in range(total_lines):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        left_padded = pad_cell(left, left_width)
        print(left_padded + (" " * gap) + right)


def print_section(title: str) -> None:
    line = "=" * 24
    print(f"\n{line} {title} {line}")


def print_subsection(title: str) -> None:
    line = "-" * 20
    print(f"\n{line} {title} {line}")


def parse_task_name(name: str) -> Tuple[str, Optional[int]]:
    m = re.match(r"^([A-Z0-9_]+)(?:\((\d+)\))?$", str(name))
    if not m:
        return str(name), None
    base = m.group(1)
    idx = int(m.group(2)) if m.group(2) is not None else None
    return base, idx


def get_task_cycle(tasks: List[Dict[str, Any]], task_name: str, idx: Optional[int] = None) -> Optional[float]:
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
    print_section("Stage 1: AICPU Prep and AICore First-task Wait")
    analyze_stage1_ctrl(aicpu_dev_pref)
    analyze_stage1_sched(aicpu_dev_pref)
    analyze_stage1_aicore(aicore_wait_rows)


def analyze_stage1_ctrl(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    print_subsection("Stage 1 - CTRL AICPU")
    ctrl = next((x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-CTRL"), None)
    if ctrl is None:
        print("- No AICPU-CTRL data found")
        return
    tasks = ctrl.get("tasks", [])
    freq = float(ctrl.get("freq", 0)) or 1.0
    build_dur = get_task_duration(tasks, "BEGIN", "DEV_TASK_BUILD", None, 0)
    print_table(["Phase", "Time(us)"], [["DEV_TASK_BUILD", format_us(build_dur, freq)]])


def analyze_stage1_sched(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    print_subsection("Stage 1 - SCHED AICPU")
    scheds = [x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-SCHED"]
    if not scheds:
        print("- No AICPU-SCHED data found")
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
    print_subsection("Stage 1 - AICore Wait Before First Callop")
    values: List[float] = []
    ref_freq = 1.0
    for row in aicore_wait_rows:
        wait_dev_task = row.get("first_wait")
        freq = float(row.get("freq", 1.0)) or 1.0
        ref_freq = freq
        if wait_dev_task is not None:
            values.append(wait_dev_task)

    if not aicore_wait_rows:
        print("- No eligible AICore data found")
        return

    stat = summarize_us(values, ref_freq)
    print_table(["Stats", "count", "min(us)", "avg(us)", "max(us)"], [["AICore wait for dev task", stat[0], stat[1], stat[2], stat[3]]])


def analyze_stage2(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    print_section("Stage 2: AICore Overall Execution")
    all_wait_first: List[float] = []
    all_exec_done: List[float] = []
    aic_total = 0.0
    aiv_total = 0.0
    aic_count = 0
    aiv_count = 0

    for core in aicpu_dev_pref:
        ctype = str(core.get("coreType", ""))
        if not (ctype.startswith("SCHED") and ("-AIC" in ctype or "-AIV" in ctype)):
            continue
        tasks = core.get("tasks", [])
        wait_first = get_task_cycle(tasks, "DEV_TASK_WAIT_RCV_FIRST_CALLOP_TASK", 0)
        all_exec = get_task_cycle(tasks, "DEV_TASK_ALL_CALLOP_TASK_EXEC", 0)
        if wait_first is None or all_exec is None or all_exec <= wait_first:
            continue

        dur = all_exec - wait_first
        if "-AIC" in ctype:
            aic_count += 1
            aic_total += dur
        else:
            aiv_count += 1
            aiv_total += dur
        all_wait_first.append(wait_first)
        all_exec_done.append(all_exec)

    if not all_wait_first or not all_exec_done:
        print("- No valid AICore execution data")
        return

    freq = 1.0
    if aicpu_dev_pref:
        freq = float(aicpu_dev_pref[0].get("freq", 1.0)) or 1.0

    e2e_cycles = max(all_exec_done) - min(all_wait_first)
    total_exec_cycles = aic_total + aiv_total

    print_tables_side_by_side(
        ["Metric", "Time(us)"],
        [
            ["AICore End-to-End time", f"{to_us(e2e_cycles, freq):.2f}"],
            ["AICore total exec time", f"{to_us(total_exec_cycles, freq):.2f}"],
        ],
        ["Type", "Count", "Total time(us)"],
        [
            ["AIC", str(aic_count), f"{to_us(aic_total, freq):.2f}"],
            ["AIV", str(aiv_count), f"{to_us(aiv_total, freq):.2f}"],
        ],
    )


def analyze_stage3(aicore_wait_rows: List[Dict[str, Any]]) -> None:
    print_section("Stage 3: AICore Exit Wait After Execution")
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
        print("- No AICore exit-wait data found")
        return
    stat = summarize_us(values, ref_freq)
    print_table(["Stats", "count", "min(us)", "avg(us)", "max(us)"], [["AICore exit wait", stat[0], stat[1], stat[2], stat[3]]])


def collect_aicore_wait_rows(aicpu_dev_pref: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for core in aicpu_dev_pref:
        core_type = str(core.get("coreType", ""))
        if not (core_type.startswith("SCHED") and ("-AIC" in core_type or "-AIV" in core_type)):
            continue
        tasks = core.get("tasks", [])
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
    print_section("Appendix: AICore Wait Details")
    if not aicore_wait_rows:
        print("- No AICore detail data available")
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
        ["coreType", "blockIdx", "Wait before first callop(us)", "Exit wait after exec(us)"],
        rows,
    )


def analyze_stage4(aicpu_dev_pref: List[Dict[str, Any]]) -> None:
    print_section("Stage 4: SCHED AICPU Post-processing")
    scheds = [x for x in aicpu_dev_pref if str(x.get("coreType")) == "AICPU-SCHED"]
    if not scheds:
        print("- No AICPU-SCHED data found")
        return
    rows: List[List[str]] = []
    for s in sorted(scheds, key=lambda x: int(x.get("blockIdx", 0))):
        tasks = s.get("tasks", [])
        gap = get_task_duration(tasks, "DEV_TASK_SCHED_EXEC", "WAIT_CORE_EXIT", 0, None)
        freq = float(s.get("freq", 0)) or 1.0
        rows.append([str(int(s.get("blockIdx", -1))), format_us(gap, freq)])
    print_table(["blockIdx", "SCHED AICPU post-process time(us)"], rows)


def analyze_output_command(output_dir_arg: Optional[str]) -> None:
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
    else:
        output_dirs = get_output_dirs()
        if not output_dirs:
            print("Error: no output_* directory found")
            return
        output_dir = output_dirs[0]

    if not output_dir.exists():
        print(f"Error: directory does not exist: {output_dir}")
        return

    aicpu_pref_file = output_dir / "aicpu_dev_pref.json"
    if not aicpu_pref_file.exists():
        print(f"Error: {aicpu_pref_file} does not exist")
        return

    print(f"Analyzing directory: {output_dir}")
    aicpu_dev_pref = load_json(aicpu_pref_file)
    if not isinstance(aicpu_dev_pref, list):
        print("Error: invalid aicpu_dev_pref.json format, expected list")
        return

    aicore_wait_rows = collect_aicore_wait_rows(aicpu_dev_pref)
    analyze_stage1(aicpu_dev_pref, aicore_wait_rows)
    analyze_stage2(aicpu_dev_pref)
    analyze_stage3(aicore_wait_rows)
    analyze_stage4(aicpu_dev_pref)
    analyze_aicore_wait_detail(aicore_wait_rows)
    print()


def main():
    parser = argparse.ArgumentParser(description='Performance data processing tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # parse_log 子命令
    parse_parser = subparsers.add_parser('parse_log', help='Parse device log and generate performance JSON')
    parse_parser.add_argument('input_file', help='Path to input log file')
    parse_parser.add_argument('output_file', help='Path to output JSON file')

    # gen_perfetto 子命令
    perfetto_parser = subparsers.add_parser('gen_perfetto', help='Convert performance JSON to Perfetto format')
    perfetto_parser.add_argument('input_file', help='Input JSON file path')
    perfetto_parser.add_argument('output_file', help='Output Perfetto JSON file path')

    # gen_perfetto_example 子命令
    example_parser = subparsers.add_parser('gen_perfetto_example', help='Generate example Perfetto data')
    # analyze 子命令
    analyze_parser = subparsers.add_parser('analyze', help='Analyze aicpu_dev_pref.json in output directory')
    analyze_parser.add_argument('output_dir', nargs='?', help='Path to output directory; latest output_* if omitted')

    args = parser.parse_args()

    if args.command == 'parse_log':
        parse_log_command(args.input_file, args.output_file)
    elif args.command == 'gen_perfetto':
        gen_perfetto_command(args.input_file, args.output_file)
    elif args.command == 'gen_perfetto_example':
        gen_perfetto_example()
    elif args.command == 'analyze':
        analyze_output_command(args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
