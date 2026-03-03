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

US_TO_MS = 1000.0
TOP_K_DEFAULT = 10
TOP_LIST_LIMIT = 5
TOP_HINT_LIMIT = 6
TOP_STAGE_LIMIT = 8
P95_QUANTILE = 0.95

RATING_FIVE_STARS = 5
RATING_FOUR_STARS = 4
RATING_THREE_STARS = 3
RATING_TWO_STARS = 2
RATING_ONE_STAR = 1

UTIL_LOW_THRESHOLD = 0.6
LARGE_GAP_THRESHOLD_US = 1000
HIGH_BUBBLE_THRESHOLD = 0.4
MEMORY_EFFICIENCY_LOW_THRESHOLD = 0.3
EXEC_JITTER_RATIO_THRESHOLD = 1.5
LOW_RATING_THRESHOLD = RATING_TWO_STARS
CONTROL_OVERHEAD_LABEL_THRESHOLD = 0.1
WAIT_RATIO_LABEL_THRESHOLD = 0.3
DEFAULT_AICPU_STAGE_DURATION = 1.0

RATING_THRESHOLDS = [
    (
        RATING_FIVE_STARS,
        {
            "utilization": (">", 0.98),
            "bubble_ratio": ("<", 0.02),
            "memory_efficiency": (">", 0.70),
            "control_overhead": ("<", 0.30),
        },
    ),
    (
        RATING_FOUR_STARS,
        {
            "utilization": (">", 0.95),
            "bubble_ratio": ("<", 0.05),
            "memory_efficiency": (">", 0.50),
            "control_overhead": ("<", 0.50),
        },
    ),
    (
        RATING_THREE_STARS,
        {
            "utilization": (">", 0.90),
            "bubble_ratio": ("<", 0.10),
            "memory_efficiency": (">", 0.30),
            "control_overhead": ("<", 0.70),
        },
    ),
    (
        RATING_TWO_STARS,
        {
            "utilization": (">", 0.80),
            "bubble_ratio": ("<", 0.20),
            "memory_efficiency": (">", 0.20),
            "control_overhead": ("<", 0.80),
        },
    ),
]
OPTIONAL_RATING_METRICS = (
    "bubble_ratio",
    "memory_efficiency",
    "control_overhead",
)

BUBBLE_THREAD_MARKER = "Execute task num"
BUBBLE_FIELD_PATTERNS = {
    "Core Total Work Time": ("span_time", "total_span_time"),
    "Total Wait Time": ("wait_time", "total_wait_time"),
    "Wait Schedule Time": ("wait_schedule", None),
    "Wait Predecessor Time": ("wait_predecessor", None),
}

PIPE_USAGE_PREFIX_HANDLERS = {
    "Total Core Num:": "total_core_num",
    "AIC:": "aic_num",
    "AIV:": "aiv_num",
}

STRUCTURED_MEM_KEYS = ("mem_usage", "bytes", "nbytes", "byteSize", "size")

logger = logging.getLogger(__name__)


@dataclass
class PerformanceData:
    """Aggregated performance metrics for rating calculation."""

    aic_util: float
    aiv_util: float
    core_stats: Dict[int, Dict[str, Any]]
    bubble_result: Dict[str, Any]
    memory_result: Dict[str, Any]
    trace_result: Dict[str, Any]
    timeline_length: int


@dataclass
class RecommendationData:
    """Input data for generating optimization recommendations."""

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
    """Input data for writing the analysis summary JSON."""

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
    """Extract the first floating-point number from a text string."""
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
        """Initialize SwimlaneAnalyzer with trace data."""
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
    def _extract_structured_entries(hint: str) -> Optional[List[Dict]]:
        """Parse hint string via ast.literal_eval and return list of dict entries."""
        try:
            structured = ast.literal_eval(hint)
        except (SyntaxError, ValueError):
            return None
        if isinstance(structured, dict):
            return [structured]
        if isinstance(structured, list):
            return [e for e in structured if isinstance(e, dict)]
        return None

    @staticmethod
    def extract_value(text: str, key: str) -> float:
        """Extract a numeric value associated with a key from multi-line text."""
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
    def parse_kv_hint(
        hint: str, shape_start: int, dtype_start: int, mem_usage_start: int
    ) -> Dict[str, Any]:
        """Parse key-value style operand hint."""
        parsed: Dict[str, Any] = {"shape": "", "dtype": "", "mem_usage": 0}
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
        return parsed

    @staticmethod
    def parse_structured_hint(hint: str) -> Dict[str, Any]:
        """Parse structured (dict/list) operand hint via ast.literal_eval."""
        parsed: Dict[str, Any] = {"shape": "", "dtype": "", "mem_usage": 0}
        dict_entries = SwimlaneAnalyzer._extract_structured_entries(hint)
        if dict_entries is None:
            return parsed
        dtype_candidates = _extract_field_candidates(
            dict_entries, ("dtype", "dataType", "type")
        )
        shape_candidates = _extract_field_candidates(
            dict_entries, ("shape", "dims", "originShape")
        )
        mem_candidates = _extract_mem_candidates(dict_entries)
        parsed["dtype"] = ",".join(dtype_candidates[:2]) if dtype_candidates else ""
        parsed["shape"] = "; ".join(shape_candidates[:2]) if shape_candidates else ""
        parsed["mem_usage"] = sum(mem_candidates) if mem_candidates else 0
        return parsed

    @staticmethod
    def parse_operand_hint(hint: str) -> Dict[str, Any]:
        """Parse an operand hint string into structured shape, dtype, and memory info."""
        result: Dict[str, Any] = {
            "shape": "", "dtype": "", "mem_usage": 0, "raw": hint, "format": "unknown",
        }
        shape_start = hint.find("shape:")
        dtype_start = hint.find("dtype:")
        mem_usage_start = hint.find("mem_usage:")
        kv = SwimlaneAnalyzer.parse_kv_hint(hint, shape_start, dtype_start, mem_usage_start)
        if kv["shape"] or kv["dtype"] or kv["mem_usage"] > 0:
            result.update(kv)
            result["format"] = "kv"
            return result
        structured = SwimlaneAnalyzer.parse_structured_hint(hint)
        if structured["shape"] or structured["dtype"] or structured["mem_usage"] > 0:
            result.update(structured)
            result["format"] = "structured"
        return result

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

    def analyze_execution_time_stats(self) -> Dict[str, Any]:
        """Aggregate execution time statistics from swimlane execution hints."""
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
        """Analyze UB memory usage and operand hints from swimlane events."""
        memory_info: Dict[str, Any] = {
            "peak_ub_bytes": 0,
            "ub_events": [],
            "operand_hints": [],
            "total_operand_mem_usage": 0,
            "peak_operand_mem_usage": 0,
            "memory_efficiency": None,
        }

        self._collect_operand_memory(memory_info)
        self._collect_ub_memory(memory_info)

        peak = memory_info["peak_ub_bytes"]
        peak_operand = memory_info["peak_operand_mem_usage"]
        if peak > 0 and peak_operand > 0:
            efficiency = min(peak_operand / peak, 1.0)
            memory_info["memory_efficiency"] = efficiency

        return memory_info

    def _collect_operand_memory(
        self, memory_info: Dict[str, Any]
    ) -> None:
        """Collect operand hint memory data from task events."""
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
            memory_info["peak_operand_mem_usage"] = max(
                memory_info["peak_operand_mem_usage"], task_mem_sum
            )

    def _collect_ub_memory(self, memory_info: Dict[str, Any]) -> None:
        """Collect UB memory usage events."""
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


def _extract_mem_candidates(dict_entries: List[Dict]) -> List[int]:
    """Extract memory usage candidates from structured dict entries."""
    mem_candidates = []
    for entry in dict_entries:
        for mem_key in STRUCTURED_MEM_KEYS:
            if mem_key not in entry:
                continue
            parsed_value = extract_first_float(str(entry.get(mem_key)))
            if parsed_value is not None and parsed_value > 0:
                mem_candidates.append(int(parsed_value))
                break
    return mem_candidates


def _extract_field_candidates(
    dict_entries: List[Dict], keys: Tuple[str, ...]
) -> List[str]:
    """Extract first matching field value from each entry for given key candidates."""
    candidates: List[str] = []
    for entry in dict_entries:
        for key in keys:
            value = entry.get(key)
            if value is not None:
                candidates.append(str(value))
                break
    return candidates


class BubbleAnalyzer:
    """Analyzes bubble_analysis.log for thread wait-time statistics."""

    def __init__(self, output_dir: Optional[str]):
        """Initialize BubbleAnalyzer with output directory path."""
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
        """Parse bubble_analysis.log and return thread-level wait statistics."""
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
        self._parse_bubble_lines(lines, result)

        for thread in result["threads"]:
            span_time = thread["span_time"]
            busy_time = max(span_time - thread["wait_time"], 0.0)
            thread["busy_time"] = busy_time
            result["total_busy_time"] += busy_time
            thread["utilization"] = (busy_time / span_time) if span_time > 0 else 0.0

        return result

    def _parse_bubble_lines(
        self, lines: List[str], result: Dict[str, Any]
    ) -> None:
        """Parse all lines from bubble_analysis.log into result dict."""
        for line in lines:
            if BUBBLE_THREAD_MARKER in line and "[" in line and "]" in line:
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
            elif result["threads"]:
                self._parse_thread_field(line, result["threads"][-1], result)

    @staticmethod
    def _parse_thread_field(
        line: str, thread: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Parse a single bubble field line and update thread/result dicts."""
        for field_key, (thread_key, total_key) in BUBBLE_FIELD_PATTERNS.items():
            if field_key not in line:
                continue
            value = float(line.split(":", 1)[1].strip().rstrip("us").strip())
            thread[thread_key] = value
            if total_key is not None:
                result[total_key] += value
            return


class TraceAnalyzer:
    """Analyzes trace data for AICPU control overhead."""

    def __init__(self, output_dir: Optional[str]):
        """Initialize TraceAnalyzer with output directory path."""
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
    def _parse_aicpu_block_tasks(
        tasks: List, result: Dict[str, Any]
    ) -> None:
        """Parse tasks from a single AICPU-CTRL block."""
        prev_end: Optional[float] = None
        for task in tasks:
            if not isinstance(task, dict):
                continue
            name = str(task.get("name", "UNKNOWN"))
            end_value = extract_first_float(str(task.get("end", 0)))
            if end_value is None:
                continue
            dur = DEFAULT_AICPU_STAGE_DURATION if prev_end is None else max(end_value - prev_end, 0.0)
            prev_end = end_value
            result["stages"][name] = result["stages"].get(name, 0.0) + dur
            result["total_time"] += dur

    @staticmethod
    def analyze_perfetto_trace(trace_path: Path) -> Dict[str, Any]:
        """Analyze a Perfetto trace JSON for AICPU-CTRL stage durations."""
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
        """Analyze aicpu_dev_pref.json for AICPU-CTRL task durations."""
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
            TraceAnalyzer._parse_aicpu_block_tasks(tasks, result)

        return result

    def analyze(self) -> Dict[str, Any]:
        """Analyze available trace data and return control overhead statistics."""
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
    """Analyzes output artifacts (execute, pipe_usage, topo, program, tilefwk)."""

    def __init__(self, output_dir: Optional[str]):
        """Initialize OutputArtifactsAnalyzer with output directory path."""
        self.output_dir = Path(output_dir) if output_dir else None

    @staticmethod
    def _parse_pipe_header_line(
        line: str,
    ) -> Optional[Tuple[str, int]]:
        """Parse a pipe_usage header line (core counts). Returns (key, value) or None."""
        for prefix, key in PIPE_USAGE_PREFIX_HANDLERS.items():
            if line.startswith(prefix):
                return key, int(extract_first_float(line) or 0)
        return None

    @staticmethod
    def _parse_pipe_data_line(line: str) -> Optional[Tuple[str, Dict[str, float]]]:
        """Parse a single pipe data line. Returns (pipe_name, stats) or None."""
        if line.startswith("Pipe,") or "," not in line:
            return None
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 4:
            return None
        return cols[0], {
            "avg_time": extract_first_float(cols[1]) or 0.0,
            "total_execute_time": extract_first_float(cols[2]) or 0.0,
            "usage_percent": extract_first_float(cols[3]) or 0.0,
        }

    @staticmethod
    def _process_topo_dict(
        node: dict, stack: List[Any]
    ) -> Tuple[int, int]:
        """Process a single dict node in topo traversal. Returns (tasks, edges)."""
        tasks = 1 if "taskId" in node else 0
        successors = node.get("successors")
        edges = len(successors) if isinstance(successors, list) else 0
        for value in node.values():
            if isinstance(value, (dict, list)):
                stack.append(value)
        return tasks, edges

    def analyze_all(self) -> Dict[str, Any]:
        """Run all artifact analyses and return combined results."""
        return {
            "execute": self.analyze_execute_json(),
            "pipe_usage": self.analyze_pipe_usage_csv(),
            "topo": self.analyze_topo_json(),
            "program": self.analyze_program_json(),
            "tilefwk": self.analyze_tilefwk_l1_prof_data(),
        }

    def analyze_execute_json(self) -> Dict[str, Any]:
        """Parse execute.json for task execution time and core type statistics."""
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
        """Parse pipe_usage.csv for core counts and pipe utilization data."""
        pipe_file = self._find_file(["pipe_usage.csv", "*pipe_usage*.csv"])
        if not pipe_file:
            return {}
        with open(pipe_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        total_core, aic, aiv, pipe_usage = self._parse_pipe_usage_lines(lines)
        if not pipe_usage:
            return {}
        return {
            "source": str(pipe_file),
            "total_core_num": total_core,
            "aic_num": aic,
            "aiv_num": aiv,
            "total_pipe_usage": pipe_usage,
        }

    @staticmethod
    def _parse_pipe_usage_lines(
        lines: List[str],
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Dict[str, Any]]:
        """Parse pipe_usage.csv lines into core counts and pipe usage dict."""
        counts = {"total_core_num": None, "aic_num": None, "aiv_num": None}
        pipe_usage: Dict[str, Any] = {}
        in_total_pipe = False
        for line in lines:
            header = OutputArtifactsAnalyzer._parse_pipe_header_line(line)
            if header is not None:
                counts[header[0]] = header[1]
                continue
            if line == "Total Pipe Usage":
                in_total_pipe = True
                continue
            if not in_total_pipe:
                continue
            entry = OutputArtifactsAnalyzer._parse_pipe_data_line(line)
            if entry is not None:
                pipe_usage[entry[0]] = entry[1]
        return counts["total_core_num"], counts["aic_num"], counts["aiv_num"], pipe_usage

    def analyze_topo_json(self) -> Dict[str, Any]:
        """Parse topo.json for task node and dependency edge counts."""
        topo_file = self._find_file(["topo.json", "*topo*.json"])
        if not topo_file:
            return {}
        with open(topo_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        task_nodes, edge_count = self._count_topo_elements(data)
        if task_nodes == 0 and edge_count == 0:
            return {}
        return {
            "source": str(topo_file),
            "task_like_nodes": task_nodes,
            "edge_count": edge_count,
        }

    @staticmethod
    def _count_topo_elements(data: Any) -> Tuple[int, int]:
        """Count task-like nodes and dependency edges via iterative traversal."""
        task_nodes = 0
        edges = 0
        stack: List[Any] = [data]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                t, e = OutputArtifactsAnalyzer._process_topo_dict(current, stack)
                task_nodes += t
                edges += e
            elif isinstance(current, list):
                stack.extend(v for v in current if isinstance(v, (dict, list)))
        return task_nodes, edges

    def analyze_program_json(self) -> Dict[str, Any]:
        """Parse program.json for function and tensor counts."""
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
        """Parse tilefwk_L1_prof_data.json for block and task cycle statistics."""
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
            count, cycles, core_type = self._parse_tilefwk_block(block)
            core_type_counts[core_type] = core_type_counts.get(core_type, 0) + 1
            task_count += count
            max_task_cycles = max(max_task_cycles, cycles)

        if task_count == 0 and not core_type_counts:
            return {}
        return {
            "source": str(tile_file),
            "block_count": len(data),
            "task_count": task_count,
            "max_task_cycles": max_task_cycles,
            "core_type_counts": core_type_counts,
        }

    def _find_file(self, patterns: List[str]) -> Optional[Path]:
        """Find the first matching file by glob patterns."""
        if not self.output_dir:
            return None
        for pattern in patterns:
            candidates = list(self.output_dir.rglob(pattern))
            if candidates:
                return candidates[0]
        return None

    @staticmethod
    def _parse_tilefwk_block(
        block: dict,
    ) -> Tuple[int, float, str]:
        """Parse a single tilefwk block. Returns (task_count, max_cycles, core_type)."""
        core_type = str(block.get("coreType", "UNKNOWN"))
        tasks = block.get("tasks", [])
        if not isinstance(tasks, list):
            return 0, 0.0, core_type
        max_cycles = 0.0
        for task in tasks:
            if not isinstance(task, dict):
                continue
            start = extract_first_float(str(task.get("execStart", 0)))
            end = extract_first_float(str(task.get("execEnd", 0)))
            if start is not None and end is not None:
                max_cycles = max(max_cycles, end - start, 0.0)
        return len(tasks), max_cycles, core_type

# ---------------------------------------------------------------------------
# Rating computation helpers
# ---------------------------------------------------------------------------


def _compute_metric_values(
    perf_data: PerformanceData,
) -> Tuple[Dict[str, Any], Optional[float], Optional[float]]:
    """Compute raw metric values from performance data.

    Returns (metric_values, bubble_ratio, memory_eff).
    """
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

    bubble_ratio = _compute_bubble_ratio(perf_data)
    memory_eff = perf_data.memory_result.get("memory_efficiency") if perf_data.memory_result else None

    control_overhead: Optional[float] = None
    if perf_data.trace_result and perf_data.timeline_length > 0:
        control_overhead = (
            float(perf_data.trace_result.get("total_time", 0.0))
            / float(perf_data.timeline_length)
        )

    metric_values = {
        "utilization": utilization,
        "bubble_ratio": bubble_ratio,
        "memory_efficiency": memory_eff,
        "control_overhead": control_overhead,
    }
    return metric_values, bubble_ratio, memory_eff


def _compute_bubble_ratio(perf_data: PerformanceData) -> Optional[float]:
    """Compute bubble ratio from bubble result or core stats."""
    total_span = (
        float(perf_data.bubble_result.get("total_span_time", 0.0))
        if perf_data.bubble_result else 0.0
    )
    total_wait = (
        float(perf_data.bubble_result.get("total_wait_time", 0.0))
        if perf_data.bubble_result else 0.0
    )
    if total_span > 0:
        return total_wait / total_span
    if perf_data.core_stats:
        return (
            sum(s.get("bubble", 0.0) for s in perf_data.core_stats.values())
            / len(perf_data.core_stats)
        )
    return None


def _determine_star_rating(
    metric_values: Dict[str, Any],
) -> Tuple[int, int]:
    """Determine star rating from metric values. Returns (stars, max_star_cap)."""

    def _is_pass(metric_name: str, op: str, threshold: float) -> bool:
        value = metric_values.get(metric_name)
        if value is None:
            return True
        if op == ">":
            return float(value) > threshold
        return float(value) < threshold

    missing_optional = sum(
        1 for m in OPTIONAL_RATING_METRICS
        if metric_values.get(m) is None
    )
    max_star_cap = max(2, 5 - missing_optional)

    stars = 1
    for star_level, checks in RATING_THRESHOLDS:
        if all(_is_pass(m, op, th) for m, (op, th) in checks.items()):
            stars = star_level
            break

    stars = min(stars, max_star_cap)
    return stars, max_star_cap


def calculate_performance_rating(perf_data: PerformanceData) -> Dict[str, Any]:
    """Calculate a star-based performance rating from aggregated metrics."""
    metric_values, bubble_ratio, memory_eff = _compute_metric_values(perf_data)
    stars, max_star_cap = _determine_star_rating(metric_values)

    return {
        "stars": stars,
        "label": "⭐" * stars,
        "metrics": metric_values,
        "available_metrics": {
            "utilization": True,
            "bubble_ratio": bubble_ratio is not None,
            "memory_efficiency": memory_eff is not None,
            "control_overhead": metric_values.get("control_overhead") is not None,
        },
        "max_star_cap": max_star_cap,
    }


# ---------------------------------------------------------------------------
# Recommendation helpers
# ---------------------------------------------------------------------------


def _check_utilization_recs(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Check utilization and return recommendations for low AIC/AIV."""
    recs: List[Dict[str, str]] = []
    if rec_data.aic_util < UTIL_LOW_THRESHOLD:
        recs.append({
            "issue": f"低 AICore 利用率 ({rec_data.aic_util * 100:.1f}%)",
            "knob": "cube_l1_reuse_mode / cube_nbuffer_mode",
            "suggestion": "启用 cube_l1_reuse_mode=1 或 cube_nbuffer_mode=1 增加子图合并",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
        })
    if rec_data.aiv_util < UTIL_LOW_THRESHOLD:
        recs.append({
            "issue": f"低 AIVector 利用率 ({rec_data.aiv_util * 100:.1f}%)",
            "knob": "vec_nbuffer_mode / mg_vec_parallel_lb",
            "suggestion": "启用 vec_nbuffer_mode=1 或降低 mg_vec_parallel_lb 阈值",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
        })
    return recs


def _check_gap_recs(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Check for large gaps and return recommendations."""
    recs: List[Dict[str, str]] = []
    if rec_data.top_gaps and rec_data.top_gaps[0]["gap_us"] > LARGE_GAP_THRESHOLD_US:
        recs.append({
            "issue": f"存在较大空闲间隔 ({rec_data.top_gaps[0]['gap_us']}us)",
            "knob": "device_sched_mode",
            "suggestion": "尝试 device_sched_mode=1 (L2亲和调度) 或 device_sched_mode=2 (公平调度)",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_runtime_options.md"
        })
    return recs


def _check_bubble_recs(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Check for high bubble cores and return recommendations."""
    high_bubble_cores = [
        (tid, stats)
        for tid, stats in rec_data.core_stats.items()
        if stats["bubble"] > HIGH_BUBBLE_THRESHOLD
    ]
    if not high_bubble_cores:
        return []
    return [{
        "issue": f"存在 {len(high_bubble_cores)} 个核心 bubble > 40%",
        "knob": "pg_lower_bound / pg_parallel_lower_bound",
        "suggestion": "降低 pg_lower_bound 或 pg_parallel_lower_bound 增加并行度",
        "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
    }]


def _check_trace_recs(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Check for trace control overhead and return recommendations."""
    if not rec_data.trace_result:
        return []
    if rec_data.trace_result.get("total_time", 0) <= 0:
        return []
    return [{
        "issue": "存在控制链路开销，可优化调度与控制流",
        "knob": "device_sched_mode / runtime_debug_mode",
        "suggestion": "优先检查 AICPU-CTRL 高占比阶段，结合调度模式减少控制面等待",
        "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_runtime_options.md"
    }]


def _check_memory_recs(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Check memory efficiency and return recommendations."""
    if not rec_data.memory_result:
        return []
    if rec_data.memory_result.get("peak_ub_bytes", 0) <= 0:
        return []
    memory_eff = rec_data.memory_result.get("memory_efficiency")
    if memory_eff is None or memory_eff >= MEMORY_EFFICIENCY_LOW_THRESHOLD:
        return []
    return [{
        "issue": f"UB 内存效率偏低 ({memory_eff * 100:.1f}%)",
        "knob": "set_vec_tile_shapes / set_cube_tile_shapes",
        "suggestion": "调整 tile 形状提升数据复用，避免 UB 峰值高但有效载荷低",
        "doc_ref": "/workspace/code/pypto/docs/tutorials/debug/performance.md"
    }]


def _check_execution_recs(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Check execution time jitter and return recommendations."""
    if not rec_data.execution_result:
        return []
    if rec_data.execution_result.get("count", 0) <= 0:
        return []
    max_time = rec_data.execution_result.get("max_time_max", 0.0)
    min_time = rec_data.execution_result.get("min_time_min", 0.0)
    has_jitter = max_time > 0 and min_time > 0 and (max_time / min_time) > EXEC_JITTER_RATIO_THRESHOLD
    if not has_jitter:
        return []
    return [{
        "issue": "执行时间抖动较大",
        "knob": "device_sched_mode / vec_nbuffer_mode",
        "suggestion": (
            "关注长尾算子并调整并行策略，"
            "降低最大执行时间与最小执行时间比值"
        ),
        "doc_ref": "/workspace/code/pypto/docs/tutorials/debug/performance.md"
    }]


def _check_rating_recs(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Check overall rating and return recommendations if low."""
    if not rec_data.rating_result:
        return []
    if rec_data.rating_result.get("stars", 0) > LOW_RATING_THRESHOLD:
        return []
    return [{
        "issue": f"综合性能评级偏低 ({rec_data.rating_result.get('label', '⭐')})",
        "knob": "组合调优",
        "suggestion": "优先处理利用率和 bubble 指标，再逐步优化控制开销与内存效率",
        "doc_ref": "/workspace/code/pypto/docs/tutorials/debug/performance.md"
    }]


def generate_recommendations(rec_data: RecommendationData) -> List[Dict[str, str]]:
    """Generate optimization recommendations based on analysis."""
    recommendations: List[Dict[str, str]] = []
    recommendations.extend(_check_utilization_recs(rec_data))
    recommendations.extend(_check_gap_recs(rec_data))
    recommendations.extend(_check_bubble_recs(rec_data))
    recommendations.extend(_check_trace_recs(rec_data))
    recommendations.extend(_check_memory_recs(rec_data))
    recommendations.extend(_check_execution_recs(rec_data))
    recommendations.extend(_check_rating_recs(rec_data))

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


# ---------------------------------------------------------------------------
# Report section builders
# ---------------------------------------------------------------------------


def _report_bubble_section(bubble_stats: Optional[Dict[str, Any]]) -> List[str]:
    """Build the bubble analysis section of the report."""
    lines: List[str] = ["", "## 气泡分析", ""]
    if not bubble_stats or not bubble_stats.get("threads"):
        lines.append("- 未找到 bubble_analysis.log，跳过该章节。")
        return lines
    lines.extend([
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
        lines.append(
            f"| {thread['name']} | {thread['span_time']:.2f} | "
            f"{thread['busy_time']:.2f} | {thread['wait_time']:.2f} | "
            f"{thread['wait_schedule']:.2f} | "
            f"{thread['wait_predecessor']:.2f} | "
            f"{thread['utilization'] * 100:.1f}% |"
        )
    return lines


def _report_execution_section(execution_stats: Dict[str, Any]) -> List[str]:
    """Build the execution time statistics section of the report."""
    lines: List[str] = ["", "## 执行时间统计", ""]
    if execution_stats.get("count", 0) <= 0:
        lines.append("- swimlane 中未包含 execution-hint，跳过该章节。")
        return lines
    lines.extend([
        f"- 样本数: {execution_stats['count']}",
        f"- Average Execution Time(均值): {execution_stats['avg_time_mean']:.2f} us",
        f"- Max Execution Time(最大): {execution_stats['max_time_max']:.2f} us",
        f"- Min Execution Time(最小): {execution_stats['min_time_min']:.2f} us",
        "",
        "| 任务 | Avg(us) | Max(us) | Min(us) |",
        "|------|---------|---------|---------|",
    ])
    for row in execution_stats.get("records", [])[:5]:
        lines.append(
            f"| `{row['task']}` | {row['avg_time']:.2f} | "
            f"{row['max_time']:.2f} | {row['min_time']:.2f} |"
        )
    return lines


def _report_memory_section(memory_stats: Dict[str, Any]) -> List[str]:
    """Build the memory analysis section of the report."""
    lines: List[str] = ["", "## 内存分析", ""]
    has_data = (
        memory_stats.get("peak_ub_bytes", 0) > 0
        or memory_stats.get("operand_hints")
    )
    if not has_data:
        lines.append("- 未找到 UB 内存与 operand hint 数据，跳过该章节。")
        return lines
    mem_eff = memory_stats.get("memory_efficiency")
    lines.extend([
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
    for hint in memory_stats.get("operand_hints", [])[:TOP_HINT_LIMIT]:
        lines.append(
            f"| `{hint['task']}` | {hint['hint_type']} | "
            f"{hint['shape'] or 'N/A'} | {hint['dtype'] or 'N/A'} | "
            f"{hint['mem_usage']} | {hint.get('format', 'unknown')} |"
        )
    return lines


def _report_trace_section(
    trace_stats: Optional[Dict[str, Any]], timeline_length: int
) -> List[str]:
    """Build the control overhead section of the report."""
    lines: List[str] = ["", "## 控制开销分析", ""]
    if not trace_stats or trace_stats.get("total_time", 0) <= 0:
        lines.append(
            "- 未找到 machine_runtime_operator_trace.json，跳过该章节。"
        )
        return lines
    control_ratio = (
        (trace_stats.get("total_time", 0.0) / timeline_length)
        if timeline_length > 0 else 0.0
    )
    lines.extend([
        f"数据源: `{trace_stats.get('source', 'N/A')}`",
        "",
        f"- AICPU-CTRL 总时长: {trace_stats.get('total_time', 0.0):.2f} (同源单位)",
        f"- 控制开销占比: {control_ratio * 100:.2f}%",
        f"- 数据来源类型: {trace_stats.get('source_type', 'unknown')}",
        "",
        "| Stage | 时长(us) | 占比 |",
        "|-------|----------|------|",
    ])
    sorted_stages = sorted(
        trace_stats.get("stages", {}).items(),
        key=lambda x: x[1], reverse=True,
    )
    for stage_name, stage_dur in sorted_stages[:TOP_STAGE_LIMIT]:
        ratio = trace_stats.get("stage_ratios", {}).get(stage_name, 0.0)
        lines.append(
            f"| `{stage_name}` | {stage_dur:.2f} | {ratio * 100:.2f}% |"
        )
    return lines


def _report_artifact_execute(execute_stats: Dict[str, Any]) -> List[str]:
    """Build execute.json subsection."""
    if not execute_stats:
        return []
    return [
        "### execute.json",
        f"- 数据源: `{execute_stats.get('source', 'N/A')}`",
        f"- 任务数: {execute_stats.get('task_count', 0)}",
        f"- 平均执行时长: {execute_stats.get('avg_exec_time', 0.0):.2f} us",
        f"- 最大执行时长: {execute_stats.get('max_exec_time', 0.0):.2f} us",
    ]


def _report_artifact_pipe(pipe_stats: Dict[str, Any]) -> List[str]:
    """Build pipe_usage.csv subsection."""
    if not pipe_stats:
        return []
    lines = [
        "", "### pipe_usage.csv",
        f"- 数据源: `{pipe_stats.get('source', 'N/A')}`",
        (
            f"- Core 数: {pipe_stats.get('total_core_num', 'N/A')} "
            f"(AIC={pipe_stats.get('aic_num', 'N/A')}, "
            f"AIV={pipe_stats.get('aiv_num', 'N/A')})"
        ),
        "- Total Pipe Usage:",
    ]
    for pn, pv in sorted(pipe_stats.get("total_pipe_usage", {}).items()):
        lines.append(
            f"  - {pn}: avg={pv.get('avg_time', 0.0):.2f}, "
            f"usage={pv.get('usage_percent', 0.0):.2f}%"
        )
    return lines


def _report_artifact_topo(topo_stats: Dict[str, Any]) -> List[str]:
    """Build topo.json subsection."""
    if not topo_stats:
        return []
    return [
        "", "### topo.json",
        f"- 数据源: `{topo_stats.get('source', 'N/A')}`",
        f"- 任务节点数(近似): {topo_stats.get('task_like_nodes', 0)}",
        f"- 依赖边数(近似): {topo_stats.get('edge_count', 0)}",
    ]


def _report_artifact_program(program_stats: Dict[str, Any]) -> List[str]:
    """Build program.json subsection."""
    if not program_stats:
        return []
    return [
        "", "### program.json",
        f"- 数据源: `{program_stats.get('source', 'N/A')}`",
        f"- functions 数量: {program_stats.get('function_count', 0)}",
        f"- tensors 数量: {program_stats.get('tensor_count', 0)}",
    ]


def _report_artifact_tilefwk(tilefwk_stats: Dict[str, Any]) -> List[str]:
    """Build tilefwk subsection."""
    if not tilefwk_stats:
        return []
    return [
        "", "### tilefwk_L1_prof_data.json",
        f"- 数据源: `{tilefwk_stats.get('source', 'N/A')}`",
        f"- Block 数: {tilefwk_stats.get('block_count', 0)}",
        f"- Task 数: {tilefwk_stats.get('task_count', 0)}",
        f"- 单 Task 最大 cycles: "
        f"{tilefwk_stats.get('max_task_cycles', 0.0):.2f}",
    ]


def _report_artifact_section(artifact_stats: Dict[str, Any]) -> List[str]:
    """Build the output artifacts section of the report."""
    lines: List[str] = ["", "## 其他产物分析", ""]
    has_data = any(bool(v) for v in artifact_stats.values())
    if not has_data:
        lines.append(
            "- 未检测到可解析的额外产物（execute.json / pipe_usage.csv / "
            "topo.json / program.json / tilefwk_L1_prof_data.json）。"
        )
        return lines
    lines.extend(_report_artifact_execute(artifact_stats.get("execute", {})))
    lines.extend(_report_artifact_pipe(artifact_stats.get("pipe_usage", {})))
    lines.extend(_report_artifact_topo(artifact_stats.get("topo", {})))
    lines.extend(_report_artifact_program(artifact_stats.get("program", {})))
    lines.extend(_report_artifact_tilefwk(artifact_stats.get("tilefwk", {})))
    return lines


# ---------------------------------------------------------------------------
# Report generation - orchestrator + section builders
# ---------------------------------------------------------------------------


@dataclass
class _AnalysisResults:
    """Container for all analysis results used in report generation."""

    timeline_length: int
    core_stats: Dict[int, Dict[str, Any]]
    aic_util: float
    aic_count: int
    aiv_util: float
    aiv_count: int
    execution_stats: Dict[str, Any]
    memory_stats: Dict[str, Any]
    bubble_stats: Dict[str, Any]
    trace_stats: Dict[str, Any]
    artifact_stats: Dict[str, Any]
    rating: Dict[str, Any]
    top_tasks: List[Dict[str, Any]]
    agg_tasks: List[Dict[str, Any]]
    top_gaps: List[Dict[str, Any]]
    recommendations: List[Dict[str, str]]


def _collect_analysis_results(
    analyzer: SwimlaneAnalyzer, output_dir: Optional[str]
) -> _AnalysisResults:
    """Run all analyses and return collected results."""
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
            aic_util=aic_util, aiv_util=aiv_util,
            core_stats=core_stats, bubble_result=bubble_stats,
            memory_result=memory_stats, trace_result=trace_stats,
            timeline_length=timeline_length,
        )
    )
    top_tasks = analyzer.get_top_hot_tasks(TOP_K_DEFAULT)
    agg_tasks = analyzer.get_aggregated_hot_tasks(TOP_K_DEFAULT)
    top_gaps = analyzer.get_top_gaps(TOP_K_DEFAULT)
    recommendations = generate_recommendations(
        RecommendationData(
            aic_util=aic_util, aiv_util=aiv_util,
            top_gaps=top_gaps, core_stats=core_stats,
            rating_result=rating, memory_result=memory_stats,
            trace_result=trace_stats, execution_result=execution_stats,
        )
    )
    return _AnalysisResults(
        timeline_length=timeline_length, core_stats=core_stats,
        aic_util=aic_util, aic_count=aic_count,
        aiv_util=aiv_util, aiv_count=aiv_count,
        execution_stats=execution_stats, memory_stats=memory_stats,
        bubble_stats=bubble_stats, trace_stats=trace_stats,
        artifact_stats=artifact_stats, rating=rating,
        top_tasks=top_tasks, agg_tasks=agg_tasks,
        top_gaps=top_gaps, recommendations=recommendations,
    )


def _build_report_header(
    res: _AnalysisResults, pypto_repo: str, output_dir: Optional[str]
) -> List[str]:
    """Build header, overview, and rating sections of the report."""
    bubble_ratio = res.rating["metrics"]["bubble_ratio"]
    memory_efficiency = res.rating["metrics"]["memory_efficiency"]
    control_overhead = res.rating["metrics"]["control_overhead"]
    bubble_suffix = "(降级:缺失)" if bubble_ratio is None else ""
    memory_suffix = "(降级:缺失)" if memory_efficiency is None else ""
    control_suffix = "(降级:缺失)" if control_overhead is None else ""

    return [
        "# PyPTO 性能分析报告",
        f"\n**生成时间**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"**PyPTO 仓库**: `{pypto_repo}`",
        f"**输出目录**: `{output_dir or 'N/A'}`",
        "",
        "## 整体性能指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| Timeline 长度 | {res.timeline_length:,} µs ({res.timeline_length / 1000:.2f} ms) |",
        f"| AICore 数量 | {res.aic_count} |",
        f"| AIVector 数量 | {res.aiv_count} |",
        f"| AICore 利用率 | {res.aic_util * 100:.1f}% |",
        f"| AIVector 利用率 | {res.aiv_util * 100:.1f}% |",
        "",
        "## 性能评级",
        "",
        f"**综合评级**: {res.rating['label']} ({res.rating['stars']}/5)",
        "",
        "| 维度 | 指标值 |",
        "|------|--------|",
        f"| 利用率 | {res.rating['metrics']['utilization'] * 100:.1f}% |",
        f"| 气泡率 | {(bubble_ratio * 100 if bubble_ratio is not None else 0):.1f}% {bubble_suffix} |",
        (
            f"| 内存效率 | {(memory_efficiency * 100 if memory_efficiency is not None else 0):.1f}% "
            f"{memory_suffix} |"
        ),
        f"| 控制开销 | {(control_overhead * 100 if control_overhead is not None else 0):.1f}% {control_suffix} |",
        f"| 星级上限(缺失修正) | {res.rating.get('max_star_cap', 5)}/5 |",
    ]


def _build_core_stats_section(
    res: _AnalysisResults, analyzer: SwimlaneAnalyzer
) -> List[str]:
    """Build core-level statistics table."""
    lines = [
        "",
        "## 核心级统计",
        "",
        "| 核心 | 类型 | 任务数 | 总耗时(µs) | Bubble | 利用率 |",
        "|------|------|--------|-----------|--------|--------|",
    ]
    for tid, stats in sorted(res.core_stats.items()):
        core_name = analyzer.thread_names.get(tid, f"tid_{tid}")
        lines.append(
            f"| {core_name} | {stats['core_type']} | {stats['task_count']} | "
            f"{stats['total_dur']:,} | {stats['bubble'] * 100:.1f}% | "
            f"{stats['utilization'] * 100:.1f}% |"
        )
    return lines


def _build_hot_tasks_section(res: _AnalysisResults, analyzer: SwimlaneAnalyzer) -> List[str]:
    """Build top hot tasks (single + aggregated) section."""
    lines = [
        "",
        "## Top 热点任务 (单事件)",
        "",
        "| 排名 | 任务名称 | 耗时(µs) | 核心类型 |",
        "|------|----------|----------|----------|",
    ]
    for i, task in enumerate(res.top_tasks[:TOP_LIST_LIMIT], 1):
        name = task.get("name", "Unknown")
        dur = task.get("dur", 0)
        tid = task.get("tid", 0)
        core_type = analyzer.get_core_type(tid)
        lines.append(f"| {i} | `{name}` | {dur:,} | {core_type} |")

    lines.extend([
        "",
        "## Top 热点任务 (聚合)",
        "",
        "| 任务名称 | 次数 | 总耗时(µs) | 中位数(µs) | P95(µs) |",
        "|----------|------|-----------|-----------|---------|",
    ])
    for task in res.agg_tasks[:TOP_LIST_LIMIT]:
        lines.append(
            f"| `{task['name']}` | {task['count']} | {task['sum_dur']:,} | "
            f"{task['median_dur']:,} | {task['p95_dur']:,} |"
        )
    return lines


def _build_gaps_section(res: _AnalysisResults) -> List[str]:
    """Build top idle gaps section."""
    lines = [
        "",
        "## Top 空闲间隔",
        "",
        "| 核心 | 间隔(µs) | 前任务 | 后任务 |",
        "|------|----------|--------|--------|",
    ]
    for gap in res.top_gaps[:TOP_LIST_LIMIT]:
        lines.append(
            f"| {gap['core_name']} | {gap['gap_us']:,} | "
            f"`{gap['prev_task']}` | `{gap['next_task']}` |"
        )
    return lines


def _build_recommendations_section(res: _AnalysisResults) -> List[str]:
    """Build tuning recommendations section."""
    lines = [
        "",
        "## 调优建议",
        "",
        "| 问题 | 相关 Knob | 建议 | 文档参考 |",
        "|------|-----------|------|----------|",
    ]
    for rec in res.recommendations:
        lines.append(
            f"| {rec['issue']} | `{rec['knob']}` | {rec['suggestion']} | `{rec['doc_ref']}` |"
        )
    return lines


def generate_report(
    analyzer: SwimlaneAnalyzer,
    output_path: str,
    pypto_repo: str,
    output_dir: Optional[str] = None
) -> str:
    """Generate markdown performance report."""
    res = _collect_analysis_results(analyzer, output_dir)

    report_lines: List[str] = []
    report_lines.extend(_build_report_header(res, pypto_repo, output_dir))
    report_lines.extend(_build_core_stats_section(res, analyzer))
    report_lines.extend(_report_bubble_section(res.bubble_stats))
    report_lines.extend(_report_execution_section(res.execution_stats))
    report_lines.extend(_report_memory_section(res.memory_stats))
    report_lines.extend(_report_trace_section(res.trace_stats, res.timeline_length))
    report_lines.extend(_report_artifact_section(res.artifact_stats))
    report_lines.extend(_build_hot_tasks_section(res, analyzer))
    report_lines.extend(_build_gaps_section(res))
    report_lines.extend(_build_recommendations_section(res))
    report_lines.extend(["", "---", "*报告由 pypto-performance-analyzer 生成*"])

    report_content = "\n".join(report_lines)

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")

    return report_content


# ---------------------------------------------------------------------------
# Summary and bottleneck helpers
# ---------------------------------------------------------------------------


def _compute_bottleneck_labels(
    summary_input: AnalysisSummaryInput, avg_bubble_ratio: Optional[float]
) -> List[str]:
    """Identify bottleneck labels from summary metrics."""
    mem_stats = summary_input.memory_stats or {}
    mem_eff = mem_stats.get("memory_efficiency")
    trace = summary_input.trace_stats or {}
    trace_ratio = (
        (trace.get("total_time", 0) / summary_input.timeline_length)
        if summary_input.timeline_length > 0
        else 0
    )
    bubble = summary_input.bubble_stats or {}
    wait_ratio = bubble.get("total_wait_time", 0) / bubble.get("total_span_time", 1)
    labels = (
        ["compute"] * int(summary_input.aic_util < UTIL_LOW_THRESHOLD)
        + ["compute"] * int(summary_input.aiv_util < UTIL_LOW_THRESHOLD)
        + ["scheduling"] * int(avg_bubble_ratio is not None and avg_bubble_ratio > WAIT_RATIO_LABEL_THRESHOLD)
        + ["memory"] * int(mem_eff is not None and mem_eff < MEMORY_EFFICIENCY_LOW_THRESHOLD)
        + ["control_overhead"] * int(trace_ratio > CONTROL_OVERHEAD_LABEL_THRESHOLD)
        + ["stitch"] * int(wait_ratio > WAIT_RATIO_LABEL_THRESHOLD)
    )
    return list(dict.fromkeys(labels))


def _extract_suggested_knobs(
    recommendations: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Extract suggested knobs from recommendation list."""
    knobs: List[Dict[str, str]] = []
    for rec in recommendations:
        if rec.get("knob") and rec.get("suggestion"):
            if rec.get("issue") == "通用优化建议":
                continue
            knobs.append({"knob": rec["knob"], "suggestion": rec["suggestion"]})
    return knobs


def write_analysis_summary(summary_input: AnalysisSummaryInput) -> str:
    """Write analysis summary JSON with key metrics and bottleneck labels."""
    bubble_ratios = [
        s["bubble"]
        for s in summary_input.core_stats.values()
        if s.get("bubble") is not None
    ]
    avg_bubble_ratio = (
        sum(bubble_ratios) / len(bubble_ratios) if bubble_ratios else None
    )
    bottleneck_labels = _compute_bottleneck_labels(
        summary_input, avg_bubble_ratio
    )
    suggested_knobs = _extract_suggested_knobs(summary_input.recommendations)

    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "key_metrics": {
            "aic_utilization": round(summary_input.aic_util, 4),
            "aiv_utilization": round(summary_input.aiv_util, 4),
            "timeline_length_us": summary_input.timeline_length,
            "bubble_ratio": (
                round(avg_bubble_ratio, 4)
                if avg_bubble_ratio is not None else None
            ),
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
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return str(summary_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="PyPTO Performance Analyzer - Swimlane & Graph Analysis"
    )
    parser.add_argument(
        "--pypto-repo", default=DEFAULT_PYPTO_REPO,
        help="Path to PyPTO repository"
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output directory (auto-detect latest if not specified)"
    )
    parser.add_argument(
        "--report-out", default=DEFAULT_REPORT_OUT,
        help="Path to output report file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use sample data for dry-run validation"
    )
    return parser


def _resolve_swimlane_path(args: argparse.Namespace) -> Tuple[Path, Optional[str]]:
    """Resolve swimlane file path and output directory from CLI args."""
    if args.dry_run:
        return REFERENCES_DIR / "sample_swimlane.json", None

    output_dir = args.output_dir or find_latest_output_dir(args.pypto_repo)
    if not output_dir:
        logger.error("No output directory found in %s", args.pypto_repo)
        logger.error("Hint: Run a PyPTO example first or specify --output-dir")
        raise RuntimeError("No output directory found")

    swimlane_path = Path(output_dir) / "merged_swimlane.json"
    if not swimlane_path.exists():
        swimlane_files = list(Path(output_dir).rglob("merged_swimlane.json"))
        if swimlane_files:
            swimlane_path = swimlane_files[0]
    return swimlane_path, output_dir


def _run_analysis_and_summary(
    analyzer: SwimlaneAnalyzer,
    output_dir: Optional[str],
    args: argparse.Namespace,
) -> None:
    """Run full analysis, generate report and summary, log results."""
    generate_report(analyzer, args.report_out, args.pypto_repo, output_dir)

    aic_util, aic_count = analyzer.calculate_aicore_utilization()
    aiv_util, aiv_count = analyzer.calculate_aivector_utilization()
    core_stats = analyzer.calculate_per_core_stats()
    bubble_stats = BubbleAnalyzer(output_dir).analyze()
    trace_stats = TraceAnalyzer(output_dir).analyze()
    memory_stats = analyzer.analyze_memory()
    timeline_length = analyzer.calculate_timeline_length()
    rating = calculate_performance_rating(
        PerformanceData(
            aic_util=aic_util, aiv_util=aiv_util,
            core_stats=core_stats, bubble_result=bubble_stats,
            memory_result=memory_stats, trace_result=trace_stats,
            timeline_length=timeline_length,
        )
    )
    recommendations = generate_recommendations(
        RecommendationData(
            aic_util=aic_util, aiv_util=aiv_util,
            top_gaps=analyzer.get_top_gaps(TOP_K_DEFAULT),
            core_stats=core_stats, rating_result=rating,
            memory_result=memory_stats, trace_result=trace_stats,
            execution_result=analyzer.analyze_execution_time_stats(),
        )
    )

    summary_path = Path(args.report_out).parent / "analysis_summary.json"
    write_analysis_summary(
        AnalysisSummaryInput(
            output_path=str(summary_path),
            aic_util=aic_util, aiv_util=aiv_util,
            timeline_length=timeline_length,
            core_stats=core_stats, rating=rating,
            recommendations=recommendations,
            bubble_stats=bubble_stats,
            memory_stats=memory_stats,
            trace_stats=trace_stats,
        )
    )
    _log_analysis_summary(
        summary_path, timeline_length,
        (aic_count, aic_util), (aiv_count, aiv_util), args.report_out,
    )


def _log_analysis_summary(
    summary_path: Path, timeline_length: int,
    aic_info: Tuple[int, float], aiv_info: Tuple[int, float],
    report_out: str,
) -> None:
    """Log the final analysis summary to console."""
    logger.info("Analysis summary saved to: %s", summary_path)
    logger.info("\n%s", "=" * 60)
    logger.info("PERFORMANCE ANALYSIS SUMMARY")
    logger.info("%s", "=" * 60)
    logger.info("Timeline: %s µs", f"{timeline_length:,}")
    logger.info("AICore: %d cores, %.1f%% utilization", aic_info[0], aic_info[1] * 100)
    logger.info("AIVector: %d cores, %.1f%% utilization", aiv_info[0], aiv_info[1] * 100)
    logger.info("\nReport saved to: %s", report_out)


def main():
    """CLI entry point for the performance analyzer."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    swimlane_path, output_dir = _resolve_swimlane_path(args)

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

    analyzer = SwimlaneAnalyzer(trace_data)
    _run_analysis_and_summary(analyzer, output_dir, args)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
