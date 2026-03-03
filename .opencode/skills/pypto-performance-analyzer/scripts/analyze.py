#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""
PyPTO Performance Analyzer - Swimlane & Computation Graph Analysis

Analyzes merged_swimlane.json (Chrome Trace Format) and optional computation graph JSON
to identify performance bottlenecks and provide actionable optimization recommendations.

Usage:
    python3 analyze.py --dry-run
    python3 analyze.py --pypto-repo /workspace/code/pypto
    python3 analyze.py --output-dir /path/to/output --report-out /path/to/report.md
"""
import argparse
import ast
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_DIR = SCRIPT_DIR.parent
REFERENCES_DIR = SKILL_DIR / "references"

# Default paths
DEFAULT_PYPTO_REPO = "/workspace/code/pypto"
DEFAULT_REPORT_OUT = "/workspace/code/.sisyphus/evidence/pypto-performance-report.md"

logger = logging.getLogger(__name__)


@dataclass
class PerformanceData:
    aic_util: float
    aiv_util: float
    core_stats: Dict[int, Dict[str, Any]]
    bubble_result: Dict[str, Any]
    memory_result: Dict[str, Any]
    trace_result: Dict[str, Any]
    timeline_length: int


@dataclass
class RecommendationData:
    aic_util: float
    aiv_util: float
    top_gaps: List[Dict[str, Any]]
    core_stats: Dict[int, Dict[str, Any]]
    rating_result: Optional[Dict[str, Any]] = None
    memory_result: Optional[Dict[str, Any]] = None
    trace_result: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisSummaryInput:
    output_path: str
    aic_util: float
    aiv_util: float
    timeline_length: int
    core_stats: Dict[int, Dict[str, Any]]
    rating: Dict[str, Any]
    recommendations: List[Dict[str, str]]
    bubble_stats: Optional[Dict[str, Any]] = None
    memory_stats: Optional[Dict[str, Any]] = None
    trace_stats: Optional[Dict[str, Any]] = None


def extract_first_float(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


class SwimlaneAnalyzer:
    """Analyzes Chrome Trace Format swimlane JSON for performance bottlenecks."""

    def __init__(self, trace_data: Dict[str, Any]):
        self.trace_data = trace_data
        self.events = trace_data.get("traceEvents", [])
        self.thread_names: Dict[int, str] = {}  # tid -> name
        self.task_events: List[Dict[str, Any]] = []  # ph == "X" events
        self.counter_events: List[Dict[str, Any]] = []  # ph == "C" events
        self._parse_events()

    def _parse_events(self):
        """Parse and categorize trace events."""
        for event in self.events:
            ph = event.get("ph", "")
            if ph == "M" and event.get("name") == "thread_name":
                tid = event.get("tid")
                name = event.get("args", {}).get("name", "")
                if tid is not None:
                    self.thread_names[tid] = name
            elif ph == "X":
                self.task_events.append(event)
            elif ph == "C":
                self.counter_events.append(event)

    @staticmethod
    def extract_value(text: str, key: str) -> float:
        for line in text.split("\n"):
            if key in line:
                try:
                    value_text = line.split(":", 1)[1].strip()
                except IndexError:
                    return 0.0
                value = extract_first_float(value_text)
                return value if value is not None else 0.0
        return 0.0

    @staticmethod
    def parse_operand_hint(hint: str) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "shape": "",
            "dtype": "",
            "mem_usage": 0,
            "raw": hint,
            "format": "unknown",
        }

        shape_start = hint.find("shape:")
        dtype_start = hint.find("dtype:")
        mem_usage_start = hint.find("mem_usage:")

        if shape_start >= 0:
            shape_end = len(hint)
            if dtype_start > shape_start:
                shape_end = min(shape_end, dtype_start)
            elif mem_usage_start > shape_start:
                shape_end = min(shape_end, mem_usage_start)
            parsed["shape"] = hint[shape_start + len("shape:"):shape_end].strip().rstrip(",")

        if dtype_start >= 0:
            dtype_end = len(hint)
            if mem_usage_start > dtype_start:
                dtype_end = mem_usage_start
            parsed["dtype"] = hint[dtype_start + len("dtype:"):dtype_end].strip().rstrip(",")

        if mem_usage_start >= 0:
            mem_text = hint[mem_usage_start + len("mem_usage:"):].strip().rstrip(",")
            mem_token = mem_text.split(",", 1)[0].strip()
            try:
                parsed["mem_usage"] = int(mem_token)
            except ValueError:
                parsed["mem_usage"] = 0
        if parsed["shape"] or parsed["dtype"] or parsed["mem_usage"] > 0:
            parsed["format"] = "kv"
            return parsed

        try:
            structured = ast.literal_eval(hint)
        except (SyntaxError, ValueError):
            return parsed

        entries: List[Any]
        if isinstance(structured, list):
            entries = structured
        elif isinstance(structured, dict):
            entries = [structured]
        else:
            return parsed

        dtype_candidates: List[str] = []
        shape_candidates: List[str] = []
        mem_candidates: List[int] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            dtype_val = entry.get("dtype") or entry.get("dataType") or entry.get("type")
            if dtype_val is not None:
                dtype_candidates.append(str(dtype_val))

            shape_val = entry.get("shape") or entry.get("dims") or entry.get("originShape")
            if shape_val is not None:
                shape_candidates.append(str(shape_val))

            for mem_key in ("mem_usage", "bytes", "nbytes", "byteSize", "size"):
                if mem_key not in entry:
                    continue
                value = extract_first_float(str(entry.get(mem_key)))
                if value is not None and value > 0:
                    mem_candidates.append(int(value))
                    break

        if dtype_candidates:
            parsed["dtype"] = ",".join(dtype_candidates[:2])
        if shape_candidates:
            parsed["shape"] = "; ".join(shape_candidates[:2])
        if mem_candidates:
            parsed["mem_usage"] = sum(mem_candidates)
        parsed["format"] = "structured"
        return parsed

    def get_core_type(self, tid: int) -> str:
        """Determine core type (AIC/AIV) from thread name."""
        name = self.thread_names.get(tid, "")
        if name.startswith("AIC"):
            return "AIC"
        if name.startswith("AIV"):
            return "AIV"
        return "UNKNOWN"

    def calculate_timeline_length(self) -> int:
        """Calculate timeline length from task events (in microseconds)."""
        if not self.task_events:
            return 0

        min_ts = min(e.get("ts", 0) for e in self.task_events)
        max_end = max(e.get("ts", 0) + e.get("dur", 0) for e in self.task_events)
        return max_end - min_ts

    def calculate_per_core_stats(self) -> Dict[int, Dict[str, Any]]:
        """Calculate per-core statistics including bubble ratio."""
        timeline_length = self.calculate_timeline_length()
        if timeline_length == 0:
            return {}

        core_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "total_dur": 0,
            "task_count": 0,
            "tasks": [],
            "core_type": "UNKNOWN",
            "bubble": 0.0,
            "utilization": 0.0
        })

        for event in self.task_events:
            tid = event.get("tid")
            dur = event.get("dur", 0)
            if tid is not None:
                core_stats[tid]["total_dur"] += dur
                core_stats[tid]["task_count"] += 1
                core_stats[tid]["tasks"].append(event)
                core_stats[tid]["core_type"] = self.get_core_type(tid)

        for _, stats in core_stats.items():
            stats["bubble"] = 1.0 - (stats["total_dur"] / timeline_length) if timeline_length > 0 else 0.0
            stats["utilization"] = stats["total_dur"] / timeline_length if timeline_length > 0 else 0.0

        return dict(core_stats)

    def calculate_aicore_utilization(self) -> Tuple[float, int]:
        """Calculate overall AICore utilization."""
        timeline_length = self.calculate_timeline_length()
        if timeline_length == 0:
            return 0.0, 0

        core_stats = self.calculate_per_core_stats()
        aic_cores = [tid for tid, stats in core_stats.items() if stats["core_type"] == "AIC"]

        if not aic_cores:
            return 0.0, 0

        total_aic_dur = sum(core_stats[tid]["total_dur"] for tid in aic_cores)
        utilization = total_aic_dur / (timeline_length * len(aic_cores))

        return utilization, len(aic_cores)

    def calculate_aivector_utilization(self) -> Tuple[float, int]:
        """Calculate overall AIVector utilization."""
        timeline_length = self.calculate_timeline_length()
        if timeline_length == 0:
            return 0.0, 0

        core_stats = self.calculate_per_core_stats()
        aiv_cores = [tid for tid, stats in core_stats.items() if stats["core_type"] == "AIV"]

        if not aiv_cores:
            return 0.0, 0

        total_aiv_dur = sum(core_stats[tid]["total_dur"] for tid in aiv_cores)
        utilization = total_aiv_dur / (timeline_length * len(aiv_cores))

        return utilization, len(aiv_cores)

    def get_top_hot_tasks(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top-K hot tasks by duration."""
        sorted_tasks = sorted(self.task_events, key=lambda e: e.get("dur", 0), reverse=True)
        return sorted_tasks[:top_k]

    def get_aggregated_hot_tasks(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get aggregated hot tasks by name (count, sum, median, p95)."""
        task_groups: Dict[str, List[int]] = defaultdict(list)

        for event in self.task_events:
            name = event.get("name", "Unknown")
            dur = event.get("dur", 0)
            task_groups[name].append(dur)

        aggregated = []
        for name, durs in task_groups.items():
            sorted_durs = sorted(durs)
            n = len(sorted_durs)
            median = sorted_durs[n // 2] if n > 0 else 0
            p95_idx = int(n * 0.95)
            p95 = sorted_durs[min(p95_idx, n - 1)] if n > 0 else 0

            aggregated.append({
                "name": name,
                "count": n,
                "sum_dur": sum(durs),
                "median_dur": median,
                "p95_dur": p95
            })

        return sorted(aggregated, key=lambda x: x["sum_dur"], reverse=True)[:top_k]

    def get_top_gaps(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top-K idle gaps per core lane."""
        gaps = []
        core_stats = self.calculate_per_core_stats()

        for tid, stats in core_stats.items():
            tasks = sorted(stats["tasks"], key=lambda e: e.get("ts", 0))
            for i in range(1, len(tasks)):
                prev_end = tasks[i - 1].get("ts", 0) + tasks[i - 1].get("dur", 0)
                curr_start = tasks[i].get("ts", 0)
                gap = curr_start - prev_end

                if gap > 0:
                    gaps.append({
                        "tid": tid,
                        "core_name": self.thread_names.get(tid, f"tid_{tid}"),
                        "gap_us": gap,
                        "prev_task": tasks[i - 1].get("name", "Unknown"),
                        "next_task": tasks[i].get("name", "Unknown")
                    })

        return sorted(gaps, key=lambda x: x["gap_us"], reverse=True)[:top_k]

    def analyze_execution_time_stats(self) -> Dict[str, Any]:
        records: List[Dict[str, Any]] = []
        stats = {
            "count": 0,
            "avg_time_mean": 0.0,
            "max_time_max": 0.0,
            "min_time_min": 0.0,
            "records": records,
        }
        avg_values: List[float] = []
        max_values: List[float] = []
        min_values: List[float] = []

        for event in self.task_events:
            args = event.get("args", {})
            if not isinstance(args, dict):
                continue
            hint = args.get("execution-hint")
            if not isinstance(hint, str) or not hint.strip():
                continue

            avg_time = SwimlaneAnalyzer.extract_value(hint, "Average Execution Time")
            max_time = SwimlaneAnalyzer.extract_value(hint, "Max Execution Time")
            min_time = SwimlaneAnalyzer.extract_value(hint, "Min Execution Time")

            avg_values.append(avg_time)
            max_values.append(max_time)
            min_values.append(min_time)
            records.append(
                {
                    "task": event.get("name", "Unknown"),
                    "tid": event.get("tid"),
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "min_time": min_time,
                }
            )

        if not avg_values:
            return stats

        stats["count"] = len(avg_values)
        stats["avg_time_mean"] = sum(avg_values) / len(avg_values)
        stats["max_time_max"] = max(max_values)
        stats["min_time_min"] = min(min_values)
        return stats

    def analyze_memory(self) -> Dict[str, Any]:
        memory_info: Dict[str, Any] = {
            "peak_ub_bytes": 0,
            "ub_events": [],
            "operand_hints": [],
            "total_operand_mem_usage": 0,
            "peak_operand_mem_usage": 0,
            "memory_efficiency": None,
        }

        for event in self.task_events:
            args = event.get("args", {})
            if not isinstance(args, dict):
                continue

            task_mem_sum = 0

            for hint_key in ("ioperand-hint", "ooperand-hint"):
                hint_value = args.get(hint_key)
                if not isinstance(hint_value, str) or not hint_value.strip():
                    continue
                parsed_hint = SwimlaneAnalyzer.parse_operand_hint(hint_value)
                memory_info["operand_hints"].append(
                    {
                        "task": event.get("name", "Unknown"),
                        "tid": event.get("tid"),
                        "hint_type": hint_key,
                        "shape": parsed_hint["shape"],
                        "dtype": parsed_hint["dtype"],
                        "mem_usage": parsed_hint["mem_usage"],
                        "format": parsed_hint["format"],
                    }
                )
                memory_info["total_operand_mem_usage"] += parsed_hint["mem_usage"]
                task_mem_sum += parsed_hint["mem_usage"]

            memory_info["peak_operand_mem_usage"] = max(memory_info["peak_operand_mem_usage"], task_mem_sum)

        for event in self.events:
            name = str(event.get("name", ""))
            if "OOO_Mem_Usage(UB)" not in name:
                continue
            args = event.get("args", {})
            if not isinstance(args, dict) or "/byte" not in args:
                continue
            try:
                ub_bytes = int(args.get("/byte", 0))
            except (TypeError, ValueError):
                ub_bytes = 0
            memory_info["peak_ub_bytes"] = max(memory_info["peak_ub_bytes"], ub_bytes)
            memory_info["ub_events"].append(
                {
                    "name": name,
                    "ts": event.get("ts", 0),
                    "ub_bytes": ub_bytes,
                }
            )

        peak = memory_info["peak_ub_bytes"]
        peak_operand = memory_info["peak_operand_mem_usage"]
        if peak > 0 and peak_operand > 0:
            efficiency = min(peak_operand / peak, 1.0)
            memory_info["memory_efficiency"] = efficiency

        return memory_info


class BubbleAnalyzer:
    def __init__(self, output_dir: Optional[str]):
        self.output_dir = Path(output_dir) if output_dir else None
        self.bubble_file: Optional[Path] = None
        if self.output_dir:
            direct = self.output_dir / "bubble_analysis.log"
            if direct.exists():
                self.bubble_file = direct
            else:
                candidates = list(self.output_dir.rglob("bubble_analysis.log"))
                if candidates:
                    self.bubble_file = candidates[0]

    def analyze(self) -> Dict[str, Any]:
        if not self.bubble_file or not self.bubble_file.exists():
            return {}

        with open(self.bubble_file, "r", encoding="utf-8") as f:
            content = f.read()

        result: Dict[str, Any] = {
            "threads": [],
            "total_wait_time": 0.0,
            "total_span_time": 0.0,
            "total_busy_time": 0.0,
            "source": str(self.bubble_file),
        }
        lines = content.split("\n")
        for line in lines:
            if "Execute task num" in line and "[" in line and "]" in line:
                thread_name = line.split("[", 1)[1].split("]", 1)[0]
                result["threads"].append(
                    {
                        "name": thread_name,
                        "span_time": 0.0,
                        "busy_time": 0.0,
                        "wait_time": 0.0,
                        "wait_schedule": 0.0,
                        "wait_predecessor": 0.0,
                        "utilization": 0.0,
                    }
                )
            elif "Core Total Work Time" in line and result["threads"]:
                value = float(line.split(":", 1)[1].strip().rstrip("us").strip())
                result["threads"][-1]["span_time"] = value
                result["total_span_time"] += value
            elif "Total Wait Time" in line and result["threads"]:
                value = float(line.split(":", 1)[1].strip().rstrip("us").strip())
                result["threads"][-1]["wait_time"] = value
                result["total_wait_time"] += value
            elif "Wait Schedule Time" in line and result["threads"]:
                result["threads"][-1]["wait_schedule"] = float(
                    line.split(":", 1)[1].strip().rstrip("us").strip()
                )
            elif "Wait Predecessor Time" in line and result["threads"]:
                result["threads"][-1]["wait_predecessor"] = float(
                    line.split(":", 1)[1].strip().rstrip("us").strip()
                )

        for thread in result["threads"]:
            span_time = thread["span_time"]
            busy_time = max(span_time - thread["wait_time"], 0.0)
            thread["busy_time"] = busy_time
            result["total_busy_time"] += busy_time
            thread["utilization"] = (busy_time / span_time) if span_time > 0 else 0.0

        return result


class TraceAnalyzer:
    def __init__(self, output_dir: Optional[str]):
        self.output_dir = Path(output_dir) if output_dir else None
        self.trace_file: Optional[Path] = None
        self.aicpu_perf_file: Optional[Path] = None
        if self.output_dir:
            for pattern in (
                "machine_runtime_operator_trace.json",
                "*machine_runtime_operator_trace*.json",
            ):
                candidates = list(self.output_dir.rglob(pattern))
                if candidates:
                    self.trace_file = candidates[0]
                    break

            for pattern in ("aicpu_dev_pref.json", "*aicpu_dev_pref*.json"):
                candidates = list(self.output_dir.rglob(pattern))
                if candidates:
                    self.aicpu_perf_file = candidates[0]
                    break

    @staticmethod
    def analyze_perfetto_trace(trace_path: Path) -> Dict[str, Any]:
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tid_name_map: Dict[Any, str] = {}
        for event in data.get("traceEvents", []):
            if event.get("ph") == "M" and event.get("name") == "thread_name":
                tid = event.get("tid")
                if tid is not None:
                    tid_name_map[tid] = str(event.get("args", {}).get("name", ""))

        result: Dict[str, Any] = {
            "stages": {},
            "total_time": 0.0,
            "source": str(trace_path),
            "source_type": "perfetto",
        }
        for event in data.get("traceEvents", []):
            if event.get("ph") != "X" or "dur" not in event:
                continue
            name = str(event.get("name", ""))
            cat = str(event.get("cat", ""))
            tid = event.get("tid")
            thread_name = tid_name_map.get(tid, "")
            if "AICPU-CTRL" not in cat and "AICPU-CTRL" not in thread_name:
                continue
            dur = float(event.get("dur", 0.0))
            result["stages"][name] = result["stages"].get(name, 0.0) + dur
            result["total_time"] += dur

        return result

    @staticmethod
    def analyze_aicpu_pref(pref_path: Path) -> Dict[str, Any]:
        with open(pref_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result: Dict[str, Any] = {
            "stages": {},
            "total_time": 0.0,
            "source": str(pref_path),
            "source_type": "aicpu_dev_pref",
        }
        if not isinstance(data, list):
            return result

        for block in data:
            if not isinstance(block, dict):
                continue
            if str(block.get("coreType", "")) != "AICPU-CTRL":
                continue
            tasks = block.get("tasks", [])
            if not isinstance(tasks, list):
                continue

            prev_end: Optional[float] = None
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                name = str(task.get("name", "UNKNOWN"))
                end_value = extract_first_float(str(task.get("end", 0)))
                if end_value is None:
                    continue
                if prev_end is None:
                    dur = 1.0
                else:
                    dur = max(end_value - prev_end, 0.0)
                prev_end = end_value
                result["stages"][name] = result["stages"].get(name, 0.0) + dur
                result["total_time"] += dur

        return result

    def analyze(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.trace_file and self.trace_file.exists():
            result = TraceAnalyzer.analyze_perfetto_trace(self.trace_file)

        has_valid_result = bool(result) and result.get("total_time", 0.0) > 0.0
        if not has_valid_result and self.aicpu_perf_file and self.aicpu_perf_file.exists():
            result = TraceAnalyzer.analyze_aicpu_pref(self.aicpu_perf_file)

        if not result:
            return {}

        total_time = float(result.get("total_time", 0.0))
        stage_ratios: Dict[str, float] = {}
        if total_time > 0:
            for stage_name, stage_dur in result.get("stages", {}).items():
                stage_ratios[stage_name] = float(stage_dur) / total_time
        result["stage_ratios"] = stage_ratios
        return result


class OutputArtifactsAnalyzer:
    def __init__(self, output_dir: Optional[str]):
        self.output_dir = Path(output_dir) if output_dir else None

    def _find_file(self, patterns: List[str]) -> Optional[Path]:
        if not self.output_dir:
            return None
        for pattern in patterns:
            candidates = list(self.output_dir.rglob(pattern))
            if candidates:
                return candidates[0]
        return None

    def analyze_execute_json(self) -> Dict[str, Any]:
        execute_file = self._find_file(["execute.json", "*execute*.json"])
        if not execute_file:
            return {}
        with open(execute_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return {}

        exec_times: List[float] = []
        core_type_counts: Dict[str, int] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            core_type = str(entry.get("coreType", "UNKNOWN"))
            core_type_counts[core_type] = core_type_counts.get(core_type, 0) + 1
            value = extract_first_float(str(entry.get("execTime", 0)))
            if value is not None:
                exec_times.append(value)

        if not exec_times and not core_type_counts:
            return {}

        return {
            "source": str(execute_file),
            "task_count": len(data),
            "avg_exec_time": (sum(exec_times) / len(exec_times)) if exec_times else 0.0,
            "max_exec_time": max(exec_times) if exec_times else 0.0,
            "core_type_counts": core_type_counts,
        }

    def analyze_pipe_usage_csv(self) -> Dict[str, Any]:
        pipe_file = self._find_file(["pipe_usage.csv", "*pipe_usage*.csv"])
        if not pipe_file:
            return {}

        result: Dict[str, Any] = {
            "source": str(pipe_file),
            "total_core_num": None,
            "aic_num": None,
            "aiv_num": None,
            "total_pipe_usage": {},
        }

        with open(pipe_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        in_total_pipe_usage = False
        for line in lines:
            if line.startswith("Total Core Num:"):
                result["total_core_num"] = int(extract_first_float(line) or 0)
                continue
            if line.startswith("AIC:"):
                result["aic_num"] = int(extract_first_float(line) or 0)
                continue
            if line.startswith("AIV:"):
                result["aiv_num"] = int(extract_first_float(line) or 0)
                continue
            if line == "Total Pipe Usage":
                in_total_pipe_usage = True
                continue
            if in_total_pipe_usage and line.startswith("Pipe,"):
                continue
            if in_total_pipe_usage and "," in line:
                cols = [c.strip() for c in line.split(",")]
                if len(cols) < 4:
                    continue
                pipe_name = cols[0]
                avg_time = extract_first_float(cols[1]) or 0.0
                total_execute = extract_first_float(cols[2]) or 0.0
                usage = extract_first_float(cols[3]) or 0.0
                result["total_pipe_usage"][pipe_name] = {
                    "avg_time": avg_time,
                    "total_execute_time": total_execute,
                    "usage_percent": usage,
                }

        if not result["total_pipe_usage"]:
            return {}
        return result

    def analyze_topo_json(self) -> Dict[str, Any]:
        topo_file = self._find_file(["topo.json", "*topo*.json"])
        if not topo_file:
            return {}
        with open(topo_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        task_like_nodes = 0
        edge_count = 0

        stack: List[Any] = [data]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                if "taskId" in current:
                    task_like_nodes += 1
                successors = current.get("successors")
                if isinstance(successors, list):
                    edge_count += len(successors)
                for value in current.values():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(current, list):
                for value in current:
                    if isinstance(value, (dict, list)):
                        stack.append(value)

        if task_like_nodes == 0 and edge_count == 0:
            return {}
        return {
            "source": str(topo_file),
            "task_like_nodes": task_like_nodes,
            "edge_count": edge_count,
        }

    def analyze_program_json(self) -> Dict[str, Any]:
        program_file = self._find_file(["program.json", "*program*.json"])
        if not program_file:
            return {}
        with open(program_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = {
            "source": str(program_file),
            "function_count": 0,
            "tensor_count": 0,
        }
        if isinstance(data, dict):
            funcs = data.get("functions")
            if isinstance(funcs, list):
                result["function_count"] = len(funcs)
            tensors = data.get("tensors")
            if isinstance(tensors, list):
                result["tensor_count"] = len(tensors)
        if result["function_count"] == 0 and result["tensor_count"] == 0:
            return {}
        return result

    def analyze_tilefwk_l1_prof_data(self) -> Dict[str, Any]:
        tile_file = self._find_file(["tilefwk_L1_prof_data.json", "*tilefwk*prof*.json"])
        if not tile_file:
            return {}
        with open(tile_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return {}

        task_count = 0
        max_task_cycles = 0.0
        core_type_counts: Dict[str, int] = {}
        for block in data:
            if not isinstance(block, dict):
                continue
            core_type = str(block.get("coreType", "UNKNOWN"))
            core_type_counts[core_type] = core_type_counts.get(core_type, 0) + 1
            tasks = block.get("tasks", [])
            if not isinstance(tasks, list):
                continue
            task_count += len(tasks)
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                start = extract_first_float(str(task.get("execStart", 0)))
                end = extract_first_float(str(task.get("execEnd", 0)))
                if start is None or end is None:
                    continue
                max_task_cycles = max(max_task_cycles, end - start, 0.0)

        if task_count == 0 and not core_type_counts:
            return {}
        return {
            "source": str(tile_file),
            "block_count": len(data),
            "task_count": task_count,
            "max_task_cycles": max_task_cycles,
            "core_type_counts": core_type_counts,
        }

    def analyze_all(self) -> Dict[str, Any]:
        return {
            "execute": self.analyze_execute_json(),
            "pipe_usage": self.analyze_pipe_usage_csv(),
            "topo": self.analyze_topo_json(),
            "program": self.analyze_program_json(),
            "tilefwk": self.analyze_tilefwk_l1_prof_data(),
        }


def calculate_performance_rating(perf_data: PerformanceData) -> Dict[str, Any]:
    utilization_candidates: List[float] = []
    if perf_data.aic_util > 0:
        utilization_candidates.append(perf_data.aic_util)
    if perf_data.aiv_util > 0:
        utilization_candidates.append(perf_data.aiv_util)
    utilization = (
        sum(utilization_candidates) / len(utilization_candidates)
        if utilization_candidates
        else 0.0
    )

    bubble_ratio: Optional[float] = None
    total_span = float(perf_data.bubble_result.get("total_span_time", 0.0)) if perf_data.bubble_result else 0.0
    total_wait = float(perf_data.bubble_result.get("total_wait_time", 0.0)) if perf_data.bubble_result else 0.0
    if total_span > 0:
        bubble_ratio = total_wait / total_span
    elif perf_data.core_stats:
        bubble_ratio = sum(stats.get("bubble", 0.0) for stats in perf_data.core_stats.values()) / len(
            perf_data.core_stats
        )

    memory_eff = perf_data.memory_result.get("memory_efficiency") if perf_data.memory_result else None

    control_overhead: Optional[float] = None
    if perf_data.trace_result and perf_data.timeline_length > 0:
        control_overhead = float(perf_data.trace_result.get("total_time", 0.0)) / float(perf_data.timeline_length)

    metric_values = {
        "utilization": utilization,
        "bubble_ratio": bubble_ratio,
        "memory_efficiency": memory_eff,
        "control_overhead": control_overhead,
    }

    thresholds = [
        (
            5,
            {
                "utilization": (">", 0.98),
                "bubble_ratio": ("<", 0.02),
                "memory_efficiency": (">", 0.70),
                "control_overhead": ("<", 0.30),
            },
        ),
        (
            4,
            {
                "utilization": (">", 0.95),
                "bubble_ratio": ("<", 0.05),
                "memory_efficiency": (">", 0.50),
                "control_overhead": ("<", 0.50),
            },
        ),
        (
            3,
            {
                "utilization": (">", 0.90),
                "bubble_ratio": ("<", 0.10),
                "memory_efficiency": (">", 0.30),
                "control_overhead": ("<", 0.70),
            },
        ),
        (
            2,
            {
                "utilization": (">", 0.80),
                "bubble_ratio": ("<", 0.20),
                "memory_efficiency": (">", 0.20),
                "control_overhead": ("<", 0.80),
            },
        ),
    ]

    def _is_pass(metric_name: str, op: str, threshold: float) -> bool:
        value = metric_values.get(metric_name)
        if value is None:
            return True
        if op == ">":
            return float(value) > threshold
        return float(value) < threshold

    missing_optional_metrics = sum(
        1
        for metric_name in ("bubble_ratio", "memory_efficiency", "control_overhead")
        if metric_values.get(metric_name) is None
    )
    max_star_cap = max(2, 5 - missing_optional_metrics)

    stars = 1
    for star_level, checks in thresholds:
        if all(_is_pass(metric_name, op, threshold) for metric_name, (op, threshold) in checks.items()):
            stars = star_level
            break

    stars = min(stars, max_star_cap)

    return {
        "stars": stars,
        "label": "⭐" * stars,
        "metrics": metric_values,
        "available_metrics": {
            "utilization": True,
            "bubble_ratio": bubble_ratio is not None,
            "memory_efficiency": memory_eff is not None,
            "control_overhead": control_overhead is not None,
        },
        "max_star_cap": max_star_cap,
    }


def generate_recommendations(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Generate optimization recommendations based on analysis."""
    recommendations = []

    # Low AIC utilization recommendation
    if rec_data.aic_util < 0.6:
        recommendations.append({
            "issue": f"低 AICore 利用率 ({rec_data.aic_util * 100:.1f}%)",
            "knob": "cube_l1_reuse_mode / cube_nbuffer_mode",
            "suggestion": "启用 cube_l1_reuse_mode=1 或 cube_nbuffer_mode=1 增加子图合并",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
        })

    # Low AIV utilization recommendation
    if rec_data.aiv_util < 0.6:
        recommendations.append({
            "issue": f"低 AIVector 利用率 ({rec_data.aiv_util * 100:.1f}%)",
            "knob": "vec_nbuffer_mode / mg_vec_parallel_lb",
            "suggestion": "启用 vec_nbuffer_mode=1 或降低 mg_vec_parallel_lb 阈值",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
        })

    # Large gaps recommendation
    if rec_data.top_gaps and rec_data.top_gaps[0]["gap_us"] > 1000:
        recommendations.append({
            "issue": f"存在较大空闲间隔 ({rec_data.top_gaps[0]['gap_us']}us)",
            "knob": "device_sched_mode",
            "suggestion": "尝试 device_sched_mode=1 (L2亲和调度) 或 device_sched_mode=2 (公平调度)",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_runtime_options.md"
        })

    # High bubble recommendation
    high_bubble_cores = [
        (tid, stats)
        for tid, stats in rec_data.core_stats.items()
        if stats["bubble"] > 0.4
    ]
    if high_bubble_cores:
        recommendations.append({
            "issue": f"存在 {len(high_bubble_cores)} 个核心 bubble > 40%",
            "knob": "pg_lower_bound / pg_parallel_lower_bound",
            "suggestion": "降低 pg_lower_bound 或 pg_parallel_lower_bound 增加并行度",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
        })

    if rec_data.trace_result and rec_data.trace_result.get("total_time", 0) > 0:
        recommendations.append({
            "issue": "存在控制链路开销，可优化调度与控制流",
            "knob": "device_sched_mode / runtime_debug_mode",
            "suggestion": "优先检查 AICPU-CTRL 高占比阶段，结合调度模式减少控制面等待",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_runtime_options.md"
        })

    if rec_data.memory_result and rec_data.memory_result.get("peak_ub_bytes", 0) > 0:
        memory_eff = rec_data.memory_result.get("memory_efficiency")
        if memory_eff is not None and memory_eff < 0.3:
            recommendations.append({
                "issue": f"UB 内存效率偏低 ({memory_eff * 100:.1f}%)",
                "knob": "set_vec_tile_shapes / set_cube_tile_shapes",
                "suggestion": "调整 tile 形状提升数据复用，避免 UB 峰值高但有效载荷低",
                "doc_ref": "/workspace/code/pypto/docs/tutorials/debug/performance.md"
            })

    if rec_data.execution_result and rec_data.execution_result.get("count", 0) > 0:
        max_time = rec_data.execution_result.get("max_time_max", 0.0)
        min_time = rec_data.execution_result.get("min_time_min", 0.0)
        if max_time > 0 and min_time > 0 and (max_time / min_time) > 1.5:
            recommendations.append({
                "issue": "执行时间抖动较大",
                "knob": "device_sched_mode / vec_nbuffer_mode",
                "suggestion": (
                    "关注长尾算子并调整并行策略，"
                    "降低最大执行时间与最小执行时间比值"
                ),
                "doc_ref": "/workspace/code/pypto/docs/tutorials/debug/performance.md"
            })

    if rec_data.rating_result and rec_data.rating_result.get("stars", 0) <= 2:
        recommendations.append({
            "issue": f"综合性能评级偏低 ({rec_data.rating_result.get('label', '⭐')})",
            "knob": "组合调优",
            "suggestion": "优先处理利用率和 bubble 指标，再逐步优化控制开销与内存效率",
            "doc_ref": "/workspace/code/pypto/docs/tutorials/debug/performance.md"
        })

    # Default tile recommendation
    recommendations.append({
        "issue": "通用优化建议",
        "knob": "set_vec_tile_shapes / set_cube_tile_shapes",
        "suggestion": (
            "Vector: pypto.set_vec_tile_shapes(64, 512); "
            "Cube: pypto.set_cube_tile_shapes([128,128], [128,128], [128,128])"
        ),
        "doc_ref": "/workspace/code/pypto/docs/tutorials/debug/performance.md"
    })

    return recommendations


def find_latest_output_dir(pypto_repo: str) -> Optional[str]:
    """Find the latest output directory in pypto repo."""
    output_base = Path(pypto_repo) / "output"
    if not output_base.exists():
        return None

    subdirs = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("output_")]
    if not subdirs:
        return None

    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return str(latest)


def generate_report(
    analyzer: SwimlaneAnalyzer,
    output_path: str,
    pypto_repo: str,
    output_dir: Optional[str] = None
) -> str:
    """Generate markdown performance report."""

    timeline_length = analyzer.calculate_timeline_length()
    core_stats = analyzer.calculate_per_core_stats()
    aic_util, aic_count = analyzer.calculate_aicore_utilization()
    aiv_util, aiv_count = analyzer.calculate_aivector_utilization()
    execution_stats = analyzer.analyze_execution_time_stats()
    memory_stats = analyzer.analyze_memory()
    bubble_stats = BubbleAnalyzer(output_dir).analyze()
    trace_stats = TraceAnalyzer(output_dir).analyze()
    artifact_stats = OutputArtifactsAnalyzer(output_dir).analyze_all()
    rating = calculate_performance_rating(
        PerformanceData(
            aic_util=aic_util,
            aiv_util=aiv_util,
            core_stats=core_stats,
            bubble_result=bubble_stats,
            memory_result=memory_stats,
            trace_result=trace_stats,
            timeline_length=timeline_length,
        )
    )
    top_tasks = analyzer.get_top_hot_tasks(10)
    agg_tasks = analyzer.get_aggregated_hot_tasks(10)
    top_gaps = analyzer.get_top_gaps(10)
    recommendations = generate_recommendations(
        RecommendationData(
            aic_util=aic_util,
            aiv_util=aiv_util,
            top_gaps=top_gaps,
            core_stats=core_stats,
            rating_result=rating,
            memory_result=memory_stats,
            trace_result=trace_stats,
            execution_result=execution_stats,
        )
    )

    bubble_ratio = rating["metrics"]["bubble_ratio"]
    memory_efficiency = rating["metrics"]["memory_efficiency"]
    control_overhead = rating["metrics"]["control_overhead"]
    bubble_suffix = "(降级:缺失)" if bubble_ratio is None else ""
    memory_suffix = "(降级:缺失)" if memory_efficiency is None else ""
    control_suffix = "(降级:缺失)" if control_overhead is None else ""

    report_lines = [
        "# PyPTO 性能分析报告",
        f"\n**生成时间**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"**PyPTO 仓库**: `{pypto_repo}`",
        f"**输出目录**: `{output_dir or 'N/A'}`",
        "",
        "## 整体性能指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| Timeline 长度 | {timeline_length:,} µs ({timeline_length / 1000:.2f} ms) |",
        f"| AICore 数量 | {aic_count} |",
        f"| AIVector 数量 | {aiv_count} |",
        f"| AICore 利用率 | {aic_util * 100:.1f}% |",
        f"| AIVector 利用率 | {aiv_util * 100:.1f}% |",
        "",
        "## 性能评级",
        "",
        f"**综合评级**: {rating['label']} ({rating['stars']}/5)",
        "",
        "| 维度 | 指标值 |",
        "|------|--------|",
        f"| 利用率 | {rating['metrics']['utilization'] * 100:.1f}% |",
        f"| 气泡率 | {(bubble_ratio * 100 if bubble_ratio is not None else 0):.1f}% {bubble_suffix} |",
        (
            f"| 内存效率 | {(memory_efficiency * 100 if memory_efficiency is not None else 0):.1f}% "
            f"{memory_suffix} |"
        ),
        f"| 控制开销 | {(control_overhead * 100 if control_overhead is not None else 0):.1f}% {control_suffix} |",
        f"| 星级上限(缺失修正) | {rating.get('max_star_cap', 5)}/5 |",
        "",
        "## 核心级统计",
        "",
        "| 核心 | 类型 | 任务数 | 总耗时(µs) | Bubble | 利用率 |",
        "|------|------|--------|-----------|--------|--------|",
    ]

    for tid, stats in sorted(core_stats.items()):
        core_name = analyzer.thread_names.get(tid, f"tid_{tid}")
        report_lines.append(
            f"| {core_name} | {stats['core_type']} | {stats['task_count']} | "
            f"{stats['total_dur']:,} | {stats['bubble'] * 100:.1f}% | {stats['utilization'] * 100:.1f}% |"
        )

    report_lines.extend([
        "",
        "## 气泡分析",
        "",
    ])

    if bubble_stats and bubble_stats.get("threads"):
        report_lines.extend([
            f"数据源: `{bubble_stats.get('source', 'N/A')}`",
            "",
            f"- 总 Span Time: {bubble_stats.get('total_span_time', 0.0):.2f} us",
            f"- 总 Busy Time: {bubble_stats.get('total_busy_time', 0.0):.2f} us",
            f"- 总 Wait Time: {bubble_stats.get('total_wait_time', 0.0):.2f} us",
            "",
            "| 线程 | Span(us) | Busy(us) | Wait(us) | WaitSchedule(us) | WaitPred(us) | 利用率 |",
            "|------|----------|----------|----------|------------------|--------------|--------|",
        ])
        for thread in bubble_stats.get("threads", []):
            report_lines.append(
                f"| {thread['name']} | {thread['span_time']:.2f} | {thread['busy_time']:.2f} | "
                f"{thread['wait_time']:.2f} | "
                f"{thread['wait_schedule']:.2f} | {thread['wait_predecessor']:.2f} | "
                f"{thread['utilization'] * 100:.1f}% |"
            )
    else:
        report_lines.append("- 未找到 bubble_analysis.log，跳过该章节。")

    report_lines.extend([
        "",
        "## 执行时间统计",
        "",
    ])

    if execution_stats.get("count", 0) > 0:
        report_lines.extend([
            f"- 样本数: {execution_stats['count']}",
            f"- Average Execution Time(均值): {execution_stats['avg_time_mean']:.2f} us",
            f"- Max Execution Time(最大): {execution_stats['max_time_max']:.2f} us",
            f"- Min Execution Time(最小): {execution_stats['min_time_min']:.2f} us",
            "",
            "| 任务 | Avg(us) | Max(us) | Min(us) |",
            "|------|---------|---------|---------|",
        ])
        for row in execution_stats.get("records", [])[:5]:
            report_lines.append(
                f"| `{row['task']}` | {row['avg_time']:.2f} | {row['max_time']:.2f} | {row['min_time']:.2f} |"
            )
    else:
        report_lines.append("- swimlane 中未包含 execution-hint，跳过该章节。")

    report_lines.extend([
        "",
        "## 内存分析",
        "",
    ])

    if memory_stats.get("peak_ub_bytes", 0) > 0 or memory_stats.get("operand_hints"):
        mem_eff = memory_stats.get("memory_efficiency")
        report_lines.extend([
            f"- UB 峰值: {memory_stats.get('peak_ub_bytes', 0)} bytes",
            f"- Operand Mem Usage 总和: {memory_stats.get('total_operand_mem_usage', 0)} bytes",
            f"- Operand 单任务峰值: {memory_stats.get('peak_operand_mem_usage', 0)} bytes",
            (
                f"- 内存效率: {(mem_eff * 100):.1f}%"
                if mem_eff is not None
                else "- 内存效率: N/A (缺少足够数据)"
            ),
            "",
            "| 任务 | Hint类型 | Shape | DType | MemUsage(bytes) | 解析格式 |",
            "|------|----------|-------|-------|-----------------|----------|",
        ])
        for hint in memory_stats.get("operand_hints", [])[:6]:
            report_lines.append(
                f"| `{hint['task']}` | {hint['hint_type']} | {hint['shape'] or 'N/A'} | "
                f"{hint['dtype'] or 'N/A'} | {hint['mem_usage']} | {hint.get('format', 'unknown')} |"
            )
    else:
        report_lines.append("- 未找到 UB 内存与 operand hint 数据，跳过该章节。")

    report_lines.extend([
        "",
        "## 控制开销分析",
        "",
    ])

    if trace_stats and trace_stats.get("total_time", 0) > 0:
        control_ratio = (trace_stats.get("total_time", 0.0) / timeline_length) if timeline_length > 0 else 0.0
        report_lines.extend([
            f"数据源: `{trace_stats.get('source', 'N/A')}`",
            "",
            f"- AICPU-CTRL 总时长: {trace_stats.get('total_time', 0.0):.2f} (同源单位)",
            f"- 控制开销占比: {control_ratio * 100:.2f}%",
            f"- 数据来源类型: {trace_stats.get('source_type', 'unknown')}",
            "",
            "| Stage | 时长(us) | 占比 |",
            "|-------|----------|------|",
        ])
        sorted_stages = sorted(trace_stats.get("stages", {}).items(), key=lambda x: x[1], reverse=True)
        for stage_name, stage_dur in sorted_stages[:8]:
            ratio = trace_stats.get("stage_ratios", {}).get(stage_name, 0.0)
            report_lines.append(f"| `{stage_name}` | {stage_dur:.2f} | {ratio * 100:.2f}% |")
    else:
        report_lines.append("- 未找到 machine_runtime_operator_trace.json，跳过该章节。")

    report_lines.extend([
        "",
        "## 其他产物分析",
        "",
    ])

    has_artifact_data = any(bool(v) for v in artifact_stats.values())
    if not has_artifact_data:
        report_lines.append(
            "- 未检测到可解析的额外产物（execute.json / pipe_usage.csv / topo.json / program.json / "
            "tilefwk_L1_prof_data.json）。"
        )
    else:
        execute_stats = artifact_stats.get("execute", {})
        if execute_stats:
            report_lines.extend([
                "### execute.json",
                f"- 数据源: `{execute_stats.get('source', 'N/A')}`",
                f"- 任务数: {execute_stats.get('task_count', 0)}",
                f"- 平均执行时长: {execute_stats.get('avg_exec_time', 0.0):.2f} us",
                f"- 最大执行时长: {execute_stats.get('max_exec_time', 0.0):.2f} us",
            ])

        pipe_stats = artifact_stats.get("pipe_usage", {})
        if pipe_stats:
            report_lines.extend([
                "",
                "### pipe_usage.csv",
                f"- 数据源: `{pipe_stats.get('source', 'N/A')}`",
                (
                    f"- Core 数: {pipe_stats.get('total_core_num', 'N/A')} "
                    f"(AIC={pipe_stats.get('aic_num', 'N/A')}, AIV={pipe_stats.get('aiv_num', 'N/A')})"
                ),
                "- Total Pipe Usage:",
            ])
            for pipe_name, pipe_value in sorted(pipe_stats.get("total_pipe_usage", {}).items()):
                report_lines.append(
                    f"  - {pipe_name}: avg={pipe_value.get('avg_time', 0.0):.2f}, "
                    f"usage={pipe_value.get('usage_percent', 0.0):.2f}%"
                )

        topo_stats = artifact_stats.get("topo", {})
        if topo_stats:
            report_lines.extend([
                "",
                "### topo.json",
                f"- 数据源: `{topo_stats.get('source', 'N/A')}`",
                f"- 任务节点数(近似): {topo_stats.get('task_like_nodes', 0)}",
                f"- 依赖边数(近似): {topo_stats.get('edge_count', 0)}",
            ])

        program_stats = artifact_stats.get("program", {})
        if program_stats:
            report_lines.extend([
                "",
                "### program.json",
                f"- 数据源: `{program_stats.get('source', 'N/A')}`",
                f"- functions 数量: {program_stats.get('function_count', 0)}",
                f"- tensors 数量: {program_stats.get('tensor_count', 0)}",
            ])

        tilefwk_stats = artifact_stats.get("tilefwk", {})
        if tilefwk_stats:
            report_lines.extend([
                "",
                "### tilefwk_L1_prof_data.json",
                f"- 数据源: `{tilefwk_stats.get('source', 'N/A')}`",
                f"- Block 数: {tilefwk_stats.get('block_count', 0)}",
                f"- Task 数: {tilefwk_stats.get('task_count', 0)}",
                f"- 单 Task 最大 cycles: {tilefwk_stats.get('max_task_cycles', 0.0):.2f}",
            ])

    report_lines.extend([
        "",
        "## Top 热点任务 (单事件)",
        "",
        "| 排名 | 任务名称 | 耗时(µs) | 核心类型 |",
        "|------|----------|----------|----------|",
    ])

    for i, task in enumerate(top_tasks[:5], 1):
        name = task.get("name", "Unknown")
        dur = task.get("dur", 0)
        tid = task.get("tid", 0)
        core_type = analyzer.get_core_type(tid)
        report_lines.append(f"| {i} | `{name}` | {dur:,} | {core_type} |")

    report_lines.extend([
        "",
        "## Top 热点任务 (聚合)",
        "",
        "| 任务名称 | 次数 | 总耗时(µs) | 中位数(µs) | P95(µs) |",
        "|----------|------|-----------|-----------|---------|",
    ])

    for task in agg_tasks[:5]:
        report_lines.append(
            f"| `{task['name']}` | {task['count']} | {task['sum_dur']:,} | "
            f"{task['median_dur']:,} | {task['p95_dur']:,} |"
        )

    report_lines.extend([
        "",
        "## Top 空闲间隔",
        "",
        "| 核心 | 间隔(µs) | 前任务 | 后任务 |",
        "|------|----------|--------|--------|",
    ])

    for gap in top_gaps[:5]:
        report_lines.append(
            f"| {gap['core_name']} | {gap['gap_us']:,} | `{gap['prev_task']}` | `{gap['next_task']}` |"
        )

    report_lines.extend([
        "",
        "## 调优建议",
        "",
        "| 问题 | 相关 Knob | 建议 | 文档参考 |",
        "|------|-----------|------|----------|",
    ])

    for rec in recommendations:
        report_lines.append(
            f"| {rec['issue']} | `{rec['knob']}` | {rec['suggestion']} | `{rec['doc_ref']}` |"
        )

    report_lines.extend([
        "", "---", "*报告由 pypto-performance-analyzer 生成*"
    ])

    report_content = "\n".join(report_lines)

    # Write report
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")

    return report_content


def write_analysis_summary(summary_input: AnalysisSummaryInput) -> str:
    # Calculate bubble ratio from core stats
    bubble_ratios = [s["bubble"] for s in summary_input.core_stats.values() if s.get("bubble") is not None]
    avg_bubble_ratio = sum(bubble_ratios) / len(bubble_ratios) if bubble_ratios else None

    # Determine bottleneck labels
    bottleneck_labels = []
    if summary_input.aic_util < 0.6:
        bottleneck_labels.append("compute")
    if summary_input.aiv_util < 0.6:
        bottleneck_labels.append("compute")
    if avg_bubble_ratio is not None and avg_bubble_ratio > 0.3:
        bottleneck_labels.append("scheduling")
    if summary_input.memory_stats and summary_input.memory_stats.get("memory_efficiency") is not None:
        if summary_input.memory_stats["memory_efficiency"] < 0.3:
            bottleneck_labels.append("memory")
    if summary_input.trace_stats and summary_input.trace_stats.get("total_time", 0) > 0:
        control_ratio = (
            summary_input.trace_stats["total_time"] / summary_input.timeline_length
            if summary_input.timeline_length > 0
            else 0
        )
        if control_ratio > 0.1:
            bottleneck_labels.append("control_overhead")
    if summary_input.bubble_stats and summary_input.bubble_stats.get("total_wait_time", 0) > 0:
        wait_ratio = (
            summary_input.bubble_stats["total_wait_time"]
            / summary_input.bubble_stats.get("total_span_time", 1)
        )
        if wait_ratio > 0.3:
            bottleneck_labels.append("stitch")
    # Deduplicate while preserving order
    seen = set()
    unique_labels = []
    for label in bottleneck_labels:
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)
    bottleneck_labels = unique_labels

    # Extract suggested knobs from recommendations
    suggested_knobs = []
    for rec in summary_input.recommendations:
        if rec.get("knob") and rec.get("suggestion"):
            # Skip the generic "通用优化建议"
            if rec.get("issue") == "通用优化建议":
                continue
            suggested_knobs.append(
                {
                    "knob": rec["knob"],
                    "suggestion": rec["suggestion"],
                }
            )

    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "key_metrics": {
            "aic_utilization": round(summary_input.aic_util, 4),
            "aiv_utilization": round(summary_input.aiv_util, 4),
            "timeline_length_us": summary_input.timeline_length,
            "bubble_ratio": round(avg_bubble_ratio, 4) if avg_bubble_ratio is not None else None,
        },
        "bottleneck_labels": bottleneck_labels,
        "suggested_knobs": suggested_knobs,
        "rating": {
            "stars": summary_input.rating.get("stars", 0),
            "label": summary_input.rating.get("label", ""),
        },
    }

    summary_path = Path(summary_input.output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return str(summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO Performance Analyzer - Swimlane & Graph Analysis"
    )
    parser.add_argument(
        "--pypto-repo",
        default=DEFAULT_PYPTO_REPO,
        help="Path to PyPTO repository"
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output directory (auto-detect latest if not specified)"
    )
    parser.add_argument(
        "--report-out",
        default=DEFAULT_REPORT_OUT,
        help="Path to output report file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use sample data for dry-run validation"
    )

    args = parser.parse_args()

    # Determine swimlane path
    if args.dry_run:
        swimlane_path = REFERENCES_DIR / "sample_swimlane.json"
        output_dir = None
    else:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = find_latest_output_dir(args.pypto_repo)

        if not output_dir:
            logger.error("No output directory found in %s", args.pypto_repo)
            logger.error("Hint: Run a PyPTO example first or specify --output-dir")
            sys.exit(1)

        swimlane_path = Path(output_dir) / "merged_swimlane.json"

        # Also check subdirectories
        if not swimlane_path.exists():
            swimlane_files = list(Path(output_dir).rglob("merged_swimlane.json"))
            if swimlane_files:
                swimlane_path = swimlane_files[0]

    if not swimlane_path.exists():
        logger.error("Swimlane file not found: %s", swimlane_path)
        sys.exit(1)

    logger.info("Loading swimlane: %s", swimlane_path)

    try:
        with open(swimlane_path, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in swimlane file: %s", e)
        sys.exit(1)

    # Analyze
    analyzer = SwimlaneAnalyzer(trace_data)

    # Generate report
    generate_report(
        analyzer,
        args.report_out,
        args.pypto_repo,
        output_dir
    )

    aic_util, aic_count = analyzer.calculate_aicore_utilization()
    aiv_util, aiv_count = analyzer.calculate_aivector_utilization()
    core_stats = analyzer.calculate_per_core_stats()
    bubble_stats = BubbleAnalyzer(output_dir).analyze()
    trace_stats = TraceAnalyzer(output_dir).analyze()
    memory_stats = analyzer.analyze_memory()
    rating = calculate_performance_rating(
        PerformanceData(
            aic_util=aic_util,
            aiv_util=aiv_util,
            core_stats=core_stats,
            bubble_result=bubble_stats,
            memory_result=memory_stats,
            trace_result=trace_stats,
            timeline_length=analyzer.calculate_timeline_length(),
        )
    )
    recommendations = generate_recommendations(
        RecommendationData(
            aic_util=aic_util,
            aiv_util=aiv_util,
            top_gaps=analyzer.get_top_gaps(10),
            core_stats=core_stats,
            rating_result=rating,
            memory_result=memory_stats,
            trace_result=trace_stats,
            execution_result=analyzer.analyze_execution_time_stats(),
        )
    )

    summary_path = Path(args.report_out).parent / "analysis_summary.json"
    write_analysis_summary(
        AnalysisSummaryInput(
            output_path=str(summary_path),
            aic_util=aic_util,
            aiv_util=aiv_util,
            timeline_length=analyzer.calculate_timeline_length(),
            core_stats=core_stats,
            rating=rating,
            recommendations=recommendations,
            bubble_stats=bubble_stats,
            memory_stats=memory_stats,
            trace_stats=trace_stats,
        )
    )
    logger.info("Analysis summary saved to: %s", summary_path)

    logger.info("\n%s", "=" * 60)
    logger.info("PERFORMANCE ANALYSIS SUMMARY")
    logger.info("%s", "=" * 60)
    logger.info("Timeline: %s µs", f"{analyzer.calculate_timeline_length():,}")
    logger.info("AICore: %d cores, %.1f%% utilization", aic_count, aic_util * 100)
    logger.info("AIVector: %d cores, %.1f%% utilization", aiv_count, aiv_util * 100)
    logger.info("\nReport saved to: %s", args.report_out)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
