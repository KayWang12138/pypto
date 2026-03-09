#!/usr/bin/env python3
"""
PyPTO 算子性能自动分析脚本

分析 PyPTO 算子运行时生成的性能数据（runtime_debug_mode=1）。
当算子运行结束时，执行此脚本会自动分析最新的 output 目录。

使用方法:
    python3 auto_performance_analysis.py              # 分析最新的 output 目录
    python3 auto_performance_analysis.py --detailed    # 分析最新的 output 目录，显示详细分析
    python3 auto_performance_analysis.py output/output_xxx  # 分析指定目录
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.swimlane_file = self.output_dir / "merged_swimlane.json"
        self.trace_file = self.output_dir / "machine_runtime_operator_trace.json"
        self.bubble_file = self.output_dir / "bubble_analysis.log"
        
    def analyze_bubble_log(self) -> Dict[str, Any]:
        """分析气泡日志"""
        if not self.bubble_file.exists():
            return {}
        
        with open(self.bubble_file, 'r') as f:
            content = f.read()
        
        result = {
            'threads': [],
            'total_wait_time': 0.0,
            'total_work_time': 0.0
        }
        
        lines = content.split('\n')
        for line in lines:
            if 'Execute task num' in line:
                thread_name = line.split('[')[1].split(']')[0]
                result['threads'].append({
                    'name': thread_name,
                    'work_time': 0.0,
                    'wait_time': 0.0,
                    'wait_schedule': 0.0,
                    'wait_predecessor': 0.0
                })
            elif 'Core Total Work Time' in line:
                result['threads'][-1]['work_time'] = float(line.split(':')[1].strip())
                result['total_work_time'] += result['threads'][-1]['work_time']
            elif 'Total Wait Time' in line:
                result['threads'][-1]['wait_time'] = float(line.split(':')[1].strip())
                result['total_wait_time'] += result['threads'][-1]['wait_time']
            elif 'Wait Schedule Time' in line:
                result['threads'][-1]['wait_schedule'] = float(line.split(':')[1].strip())
            elif 'Wait Predecessor Time' in line:
                result['threads'][-1]['wait_predecessor'] = float(line.split(':')[1].strip())
        
        return result

    def analyze_swimlane(self) -> Dict[str, Any]:
        """分析泳道图数据"""
        if not self.swimlane_file.exists():
            return {}
        
        with open(self.swimlane_file, 'r') as f:
            data = json.load(f)
        
        result = {
            'tasks': [],
            'total_duration': 0.0,
            'peak_memory': 0,
            'memory_events': []
        }
        
        for event in data.get('traceEvents', []):
            if event.get('ph') == 'X' and 'dur' in event:
                dur = event.get('dur', 0)
                name = event.get('name', '')
                args = event.get('args', {})
                
                if 'execution-hint' in args:
                    exec_hint = args['execution-hint']
                    avg_time = self._extract_value(exec_hint, 'Average Execution Time')
                    max_time = self._extract_value(exec_hint, 'Max Execution Time')
                    min_time = self._extract_value(exec_hint, 'Min Execution Time')
                    
                    result['tasks'].append({
                        'name': name,
                        'duration': dur,
                        'avg_time': avg_time,
                        'max_time': max_time,
                        'min_time': min_time
                    })
                    result['total_duration'] += dur
                
                if 'OOO_Mem_Usage(UB)' in name:
                    mem_usage = args.get('/byte', 0)
                    result['peak_memory'] = max(result['peak_memory'], mem_usage)
                    result['memory_events'].append({
                        'name': name,
                        'memory': mem_usage
                    })
        
        return result
    
    def analyze_trace(self) -> Dict[str, Any]:
        """分析性能追踪数据"""
        if not self.trace_file.exists():
            return {}
        
        with open(self.trace_file, 'r') as f:
            data = json.load(f)
        
        result = {
            'ctrl_stages': {},
            'sched_stages': {},
            'ctrl_total_time': 0.0,
            'sched_total_time': 0.0
        }
        
        for event in data.get('traceEvents', []):
            if event.get('ph') == 'X' and 'dur' in event:
                name = event.get('name', '')
                dur = event.get('dur', 0)
                cat = event.get('cat', '')
                
                if 'AICPU-CTRL' in cat:
                    result['ctrl_stages'][name] = result['ctrl_stages'].get(name, 0) + dur
                    result['ctrl_total_time'] += dur
                elif 'AICPU-SCHED' in cat:
                    result['sched_stages'][name] = result['sched_stages'].get(name, 0) + dur
                    result['sched_total_time'] += dur
        
        return result
    
    def _extract_value(self, text: str, key: str) -> float:
        """从文本中提取数值"""
        for line in text.split('\n'):
            if key in line:
                try:
                    return float(line.split(':')[1].strip())
                except:
                    pass
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        bubble_data = self.analyze_bubble_log()
        swimlane_data = self.analyze_swimlane()
        trace_data = self.analyze_trace()
        
        summary = {
            'dir_name': self.output_dir.name,
            'task_count': len(swimlane_data.get('tasks', [])),
            'total_compute_time': swimlane_data.get('total_duration', 0),
            'avg_task_time': 0,
            'ctrl_total_time': trace_data.get('ctrl_total_time', 0),
            'sched_total_time': trace_data.get('sched_total_time', 0),
            'total_control_time': trace_data.get('ctrl_total_time', 0) + trace_data.get('sched_total_time', 0),
            'peak_memory': swimlane_data.get('peak_memory', 0),
            'thread_utilization': 0,
            'bubble_rate': 0,
            'control_ratio': 0
        }
        
        if summary['task_count'] > 0:
            summary['avg_task_time'] = summary['total_compute_time'] / summary['task_count']
        
        if bubble_data.get('threads'):
            total_work = sum(t['work_time'] for t in bubble_data['threads'])
            total_time = sum(t['work_time'] + t['wait_time'] for t in bubble_data['threads'])
            summary['thread_utilization'] = (total_work / total_time * 100) if total_time > 0 else 0
            summary['bubble_rate'] = (100 - summary['thread_utilization'])
        
        total_time = summary['total_compute_time'] + summary['total_control_time']
        summary['control_ratio'] = (summary['total_control_time'] / total_time * 100) if total_time > 0 else 0
        
        return summary


def get_output_dirs(base_dir: str = "output") -> List[Path]:
    """获取所有 output 目录，按时间排序"""
    output_base = Path(base_dir)
    if not output_base.exists():
        return []
    
    dirs = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith('output_')]
    dirs.sort(key=lambda x: x.name, reverse=True)
    return dirs


def print_table(data: List[Dict[str, Any]]):
    """打印表格"""
    if not data:
        print("没有找到性能数据")
        return
    
    print("\n" + "=" * 140)
    print("PyPTO 性能分析汇总表")
    print("=" * 140)
    print()
    
    print("| 序号 | 目录名称                      | 任务数 | 计算时间 | 平均任务时间 | CTRL时间 | SCHED时间 | 总控制时间 | 控制占比(%) | 线程利用率(%) | 气泡率(%) |")
    print("|------|------------------------------|--------|-------------|----------------|---------|---------|---------|-----------|-------------|---------|")
    
    for i, item in enumerate(data, 1):
        dir_name = item['dir_name'][:30]
        print(f"| {i:4d} | {dir_name:30s} | {item['task_count']:6d} | "
              f"{item['total_compute_time']:11.2f} | {item['avg_task_time']:14.2f} | "
              f"{item['ctrl_total_time']:9.2f} | {item['sched_total_time']:9.2f} | "
              f"{item['total_control_time']:11.2f} | {item['control_ratio']:9.2f} | "
              f"{item['thread_utilization']:11.2f} | {item['bubble_rate']:7.2f} |")
    
    print()
    print("=" * 140)


def print_detailed_analysis(analyzer: PerformanceAnalyzer):
    """打印详细分析"""
    bubble_data = analyzer.analyze_bubble_log()
    swimlane_data = analyzer.analyze_swimlane()
    trace_data = analyzer.analyze_trace()
    
    print(f"\n## 详细分析: {analyzer.output_dir.name}")
    print("-" * 80)
    
    if bubble_data.get('threads'):
        print("\n### 线程性能")
        print("| 线程                   | 工作时间 | 等待时间 | 等待调度 | 等待前驱 | 利用率 |")
        print("|------------------------|---------|---------|---------|---------|--------|")
        for thread in bubble_data['threads']:
            total_time = thread['work_time'] + thread['wait_time']
            utilization = (thread['work_time'] / total_time * 100) if total_time > 0 else 0
            print(f"| {thread['name']:22s} | {thread['work_time']:11.2f}us | "
                  f"{thread['wait_time']:11.2f}us | {thread['wait_schedule']:11.2f}us | "
                  f"{thread['wait_predecessor']:11.2f}us | {utilization:7.1f}% |")
    
    if swimlane_data.get('tasks'):
        print(f"\n### 任务执行性能")
        print(f"- 总任务数: {len(swimlane_data['tasks'])}")
        print(f"- 总执行时间: {swimlane_data['total_duration']:.2f}us")
        print(f"- 平均执行时间: {swimlane_data['total_duration'] / len(swimlane_data['tasks']):.2f}us")
        print(f"- 峰值内存使用: {swimlane_data['peak_memory']} bytes")
    
    if trace_data.get('ctrl_stages'):
        print(f"\n### AICPU-CTRL 控制开销")
        print("| 阶段                          | 时间 | 占比(%) |")
        print("|--------------------------------|---------|--------|")
        for stage, time in sorted(trace_data['ctrl_stages'].items(), key=lambda x: x[1], reverse=True):
            ratio = (time / trace_data['ctrl_total_time'] * 100) if trace_data['ctrl_total_time'] > 0 else 0
            print(f"| {stage:30s} | {time:7.2f} | {ratio:6.1f} |")
    
    if trace_data.get('sched_stages'):
        print(f"\n### AICPU-SCHED 调度开销")
        print("| 阶段                          | 时间 | 占比(%) |")
        print("|--------------------------------|---------|--------|")
        for stage, time in sorted(trace_data['sched_stages'].items(), key=lambda x: x[1], reverse=True):
            ratio = (time / trace_data['sched_total_time'] * 100) if trace_data['sched_total_time'] > 0 else 0
            print(f"| {stage:30s} | {time:7.2f} | {ratio:6.1f} |")
    
    print()


def analyze_directory(output_dir: Path, show_detailed: bool = False):
    """分析单个目录"""
    if not output_dir.exists():
        print(f"警告: 目录不存在: {output_dir}")
        return None
    
    analyzer = PerformanceAnalyzer(str(output_dir))
    summary = analyzer.get_summary()
    
    if show_detailed:
        print_detailed_analysis(analyzer)
    
    return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PyPTO 性能自动分析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
说明:
  此脚本用于分析 PyPTO 算子运行时生成的性能数据（runtime_debug_mode=1）。
  当算子运行结束时，执行此脚本会自动分析最新的 output 目录。

示例:
  python3 auto_performance_analysis.py              # 分析最新的 output 目录
  python3 auto_performance_analysis.py --detailed    # 分析最新的 output 目录，显示详细分析
  python3 auto_performance_analysis.py output/output_xxx  # 分析指定目录
        """
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        help="指定要分析的 output 目录，默认分析最新的目录"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="显示详细分析"
    )
    
    args = parser.parse_args()
    
    # 分析指定目录或最新目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 获取最新的 output 目录
        output_dirs = get_output_dirs()
        if not output_dirs:
            print("错误: 没有找到 output 目录")
            sys.exit(1)
        output_dir = output_dirs[0]
    
    summary = analyze_directory(output_dir, args.detailed)
    if summary:
        print_table([summary])


if __name__ == "__main__":
    main()