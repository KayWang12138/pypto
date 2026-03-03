#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_DIR = SCRIPT_DIR.parent
REFERENCES_DIR = SKILL_DIR / "references"

# Default paths
DEFAULT_PYPTO_REPO = "/workspace/code/pypto"
DEFAULT_REPORT_OUT = "/workspace/code/.sisyphus/evidence/pypto-performance-report.md"


class SwimlaneAnalyzer:
    """Analyzes Chrome Trace Format swimlane JSON for performance bottlenecks."""
    
    def __init__(self, trace_data: Dict[str, Any]):
        self.trace_data = trace_data
        self.events = trace_data.get("traceEvents", [])
        self.thread_names: Dict[int, str] = {}  # tid -> name
        self.task_events: List[Dict] = []  # ph == "X" events
        self.counter_events: List[Dict] = []  # ph == "C" events
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
    
    def get_core_type(self, tid: int) -> str:
        """Determine core type (AIC/AIV) from thread name."""
        name = self.thread_names.get(tid, "")
        if name.startswith("AIC"):
            return "AIC"
        elif name.startswith("AIV"):
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
        
        for tid, stats in core_stats.items():
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


def generate_recommendations(
    aic_util: float,
    aiv_util: float,
    top_gaps: List[Dict],
    core_stats: Dict[int, Dict]
) -> List[Dict[str, str]]:
    """Generate optimization recommendations based on analysis."""
    recommendations = []
    
    # Low AIC utilization recommendation
    if aic_util < 0.6:
        recommendations.append({
            "issue": f"低 AICore 利用率 ({aic_util*100:.1f}%)",
            "knob": "cube_l1_reuse_mode / cube_nbuffer_mode",
            "suggestion": "启用 cube_l1_reuse_mode=1 或 cube_nbuffer_mode=1 增加子图合并",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
        })
    
    # Low AIV utilization recommendation
    if aiv_util < 0.6:
        recommendations.append({
            "issue": f"低 AIVector 利用率 ({aiv_util*100:.1f}%)",
            "knob": "vec_nbuffer_mode / mg_vec_parallel_lb",
            "suggestion": "启用 vec_nbuffer_mode=1 或降低 mg_vec_parallel_lb 阈值",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
        })
    
    # Large gaps recommendation
    if top_gaps and top_gaps[0]["gap_us"] > 1000:
        recommendations.append({
            "issue": f"存在较大空闲间隔 ({top_gaps[0]['gap_us']}us)",
            "knob": "device_sched_mode",
            "suggestion": "尝试 device_sched_mode=1 (L2亲和调度) 或 device_sched_mode=2 (公平调度)",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_runtime_options.md"
        })
    
    # High bubble recommendation
    high_bubble_cores = [(tid, stats) for tid, stats in core_stats.items() if stats["bubble"] > 0.4]
    if high_bubble_cores:
        recommendations.append({
            "issue": f"存在 {len(high_bubble_cores)} 个核心 bubble > 40%",
            "knob": "pg_lower_bound / pg_parallel_lower_bound",
            "suggestion": "降低 pg_lower_bound 或 pg_parallel_lower_bound 增加并行度",
            "doc_ref": "/workspace/code/pypto/docs/api/config/pypto-set_pass_options.md"
        })
    
    # Default tile recommendation
    recommendations.append({
        "issue": "通用优化建议",
        "knob": "set_vec_tile_shapes / set_cube_tile_shapes",
        "suggestion": "Vector: pypto.set_vec_tile_shapes(64, 512); Cube: pypto.set_cube_tile_shapes([128,128], [128,128], [128,128])",
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
    top_tasks = analyzer.get_top_hot_tasks(10)
    agg_tasks = analyzer.get_aggregated_hot_tasks(10)
    top_gaps = analyzer.get_top_gaps(10)
    recommendations = generate_recommendations(aic_util, aiv_util, top_gaps, core_stats)
    
    report_lines = [
        "# PyPTO 性能分析报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**PyPTO 仓库**: `{pypto_repo}`",
        f"**输出目录**: `{output_dir or 'N/A'}`",
        "",
        "## 整体性能指标",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Timeline 长度 | {timeline_length:,} µs ({timeline_length/1000:.2f} ms) |",
        f"| AICore 数量 | {aic_count} |",
        f"| AIVector 数量 | {aiv_count} |",
        f"| AICore 利用率 | {aic_util*100:.1f}% |",
        f"| AIVector 利用率 | {aiv_util*100:.1f}% |",
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
            f"{stats['total_dur']:,} | {stats['bubble']*100:.1f}% | {stats['utilization']*100:.1f}% |"
        )
    
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
            print(f"ERROR: No output directory found in {args.pypto_repo}", file=sys.stderr)
            print("Hint: Run a PyPTO example first or specify --output-dir", file=sys.stderr)
            sys.exit(1)
        
        swimlane_path = Path(output_dir) / "merged_swimlane.json"
        
        # Also check subdirectories
        if not swimlane_path.exists():
            swimlane_files = list(Path(output_dir).rglob("merged_swimlane.json"))
            if swimlane_files:
                swimlane_path = swimlane_files[0]
    
    if not swimlane_path.exists():
        print(f"ERROR: Swimlane file not found: {swimlane_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading swimlane: {swimlane_path}")
    
    try:
        with open(swimlane_path, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in swimlane file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    analyzer = SwimlaneAnalyzer(trace_data)
    
    # Generate report
    report = generate_report(
        analyzer,
        args.report_out,
        args.pypto_repo,
        output_dir
    )
    
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS SUMMARY")
    print('='*60)
    print(f"Timeline: {analyzer.calculate_timeline_length():,} µs")
    aic_util, aic_count = analyzer.calculate_aicore_utilization()
    aiv_util, aiv_count = analyzer.calculate_aivector_utilization()
    print(f"AICore: {aic_count} cores, {aic_util*100:.1f}% utilization")
    print(f"AIVector: {aiv_count} cores, {aiv_util*100:.1f}% utilization")
    print(f"\nReport saved to: {args.report_out}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
