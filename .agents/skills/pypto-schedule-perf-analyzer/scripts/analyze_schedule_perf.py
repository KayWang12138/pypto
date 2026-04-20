#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You cannot use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
PyPTO 调度性能分析脚本

用于分析 PerfMtTrace 打点数据，定位调度性能瓶颈。
"""

import json
import argparse
import os
import sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import statistics


PERF_TRACE_THRESHOLDS = {
    'DEV_TASK_BUILD': 100,
    'DEV_TASK_RCV': 50,
    'DEV_TASK_ENTER_DISPATCH_TASK': 100,
    'DEV_TASK_DISPATCH_RESOL_TASK': 200,
    'DEV_TASK_DISPATCH_TASK': 100,
    'DEV_TASK_SEND_CALLOP_TASK': 50,
    'DEV_TASK_SYNC_CORE_STOP': 150,
    'DEV_TASK_SCHED_EXEC': 150,
}

PERF_TRACE_CRITICAL_PATHS = [
    ('DEV_TASK_BUILD', 'DEV_TASK_RCV', '任务接收延迟'),
    ('DEV_TASK_ENTER_DISPATCH_TASK', 'DEV_TASK_DISPATCH_RESOL_TASK', '依赖解析耗时'),
    ('DEV_TASK_DISPATCH_RESOL_TASK', 'DEV_TASK_DISPATCH_TASK', '任务分发耗时'),
    ('DEV_TASK_DISPATCH_TASK', 'DEV_TASK_SEND_CALLOP_TASK', '任务发送耗时'),
    ('DEV_TASK_RUN_CORE_TASK', 'DEV_TASK_DISPATCH_TASK', '任务分发总耗时'),
]


def load_perf_data(log_dir: str, turn: int = 0) -> Optional[List[Dict]]:
    perf_file = os.path.join(log_dir, f'machine_trace_perf_data_{turn}.json')
    
    if not os.path.exists(perf_file):
        print(f"错误：性能数据文件不存在: {perf_file}")
        return None
    
    with open(perf_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def cycles_to_us(cycles: int, freq: int) -> float:
    return cycles / freq


def parse_task_name(name: str) -> Tuple[str, Optional[int]]:
    import re
    pattern = r'([A-Z_]+(?:\([0-9]+\))?)'
    match = re.match(pattern, name)
    if match:
        base_name = match.group(1)
        task_num_pattern = r'\(([0-9]+)\)'
        task_match = re.search(task_num_pattern, name)
        task_num = int(task_match.group(1)) if task_match else None
        clean_name = base_name.replace('(', '').replace(')', '').replace(f'{task_num}', '').rstrip('_')
        if task_num is not None:
            clean_name = clean_name.replace(f'_{task_num}', '')
        return clean_name, task_num
    return name, None


def analyze_core_perf(core_data: Dict) -> Dict:
    tasks = core_data.get('tasks', [])
    freq = core_data.get('freq', 1000)
    block_idx = core_data.get('blockIdx', -1)
    core_type = core_data.get('coreType', 'UNKNOWN')
    
    parsed_tasks = defaultdict(list)
    
    for task in tasks:
        name = task.get('name', '')
        end_timestamp = task.get('end', 0)
        
        clean_name, task_num = parse_task_name(name)
        parsed_tasks[clean_name].append({
            'timestamp': end_timestamp,
            'task_num': task_num,
            'original_name': name
        })
    
    return {
        'block_idx': block_idx,
        'core_type': core_type,
        'freq': freq,
        'parsed_tasks': parsed_tasks,
        'total_tasks': len(tasks)
    }


def calculate_critical_path_time(parsed_tasks: Dict, path: Tuple[str, str, str], freq: int) -> Optional[Dict]:
    start_type, end_type, description = path
    
    start_times = parsed_tasks.get(start_type, [])
    end_times = parsed_tasks.get(end_type, [])
    
    if not start_times or not end_times:
        return None
    
    start_time = start_times[0]['timestamp']
    end_time = end_times[0]['timestamp']
    
    if end_time <= start_time:
        return None
    
    time_cycles = end_time - start_time
    time_us = cycles_to_us(time_cycles, freq)
    
    threshold = PERF_TRACE_THRESHOLDS.get(end_type, 100)
    is_bottleneck = time_us > threshold
    
    return {
        'path': f'{start_type} → {end_type}',
        'description': description,
        'time_cycles': time_cycles,
        'time_us': time_us,
        'threshold': threshold,
        'is_bottleneck': is_bottleneck,
        'status': '超标' if is_bottleneck else '正常'
    }


def generate_perf_report(perf_data: List[Dict], turn: int) -> str:
    report = []
    report.append(f"## 调度性能分析报告 (turn_{turn})\n")
    
    ctrl_cores = []
    sched_cores = []
    
    for core_data in perf_data:
        analysis = analyze_core_perf(core_data)
        if analysis['core_type'] == 'AICPU-CTRL':
            ctrl_cores.append(analysis)
        elif analysis['core_type'] == 'AICPU-SCHED':
            sched_cores.append(analysis)
    
    report.append("### 1. AICPU-CTRL 调度性能\n")
    
    for ctrl in ctrl_cores:
        report.append(f"\n**核心 {ctrl['block_idx']} (AICPU-CTRL)**\n")
        report.append(f"- 总任务数: {ctrl['total_tasks']}\n")
        report.append(f"- 频率: {ctrl['freq']} MHz\n")
        
        parsed_tasks = ctrl['parsed_tasks']
        freq = ctrl['freq']
        
        report.append("\n| 打点类型 | 时间戳 (cycles) | 转换时间 (us) |\n")
        report.append("|---------|----------------|--------------|\n")
        
        for trace_type in ['BEGIN', 'ALLOC_THREAD_ID', 'DEV_TASK_BUILD', 'EXIT']:
            if trace_type in parsed_tasks:
                task = parsed_tasks[trace_type][0]
                time_us = cycles_to_us(task['timestamp'], freq)
                report.append(f"| {trace_type} | {task['timestamp']} | {time_us:.2f} |\n")
    
    report.append("\n### 2. AICPU-SCHED 调度性能\n")
    
    for sched in sched_cores:
        report.append(f"\n**核心 {sched['block_idx']} (AICPU-SCHED)**\n")
        report.append(f"- 总任务数: {sched['total_tasks']}\n")
        report.append(f"- 频率: {sched['freq']} MHz\n")
        
        parsed_tasks = sched['parsed_tasks']
        freq = sched['freq']
        
        report.append("\n| 打点类型 | 任务编号 | 时间戳 (cycles) | 转换时间 (us) |\n")
        report.append("|---------|---------|----------------|--------------|\n")
        
        dev_task_types = [
            'DEV_TASK_RCV',
            'DEV_TASK_RUN_CORE_TASK',
            'DEV_TASK_ENTER_DISPATCH_TASK',
            'DEV_TASK_DISPATCH_RESOL_REG_TASK',
            'DEV_TASK_DISPATCH_RESOL_TASK',
            'DEV_TASK_DISPATCH_TASK',
            'DEV_TASK_SEND_CALLOP_TASK'
        ]
        
        for trace_type in dev_task_types:
            if trace_type in parsed_tasks:
                for task in parsed_tasks[trace_type][:3]:
                    task_num = task.get('task_num', 'N/A')
                    time_us = cycles_to_us(task['timestamp'], freq)
                    report.append(f"| {trace_type} | {task_num} | {task['timestamp']} | {time_us:.2f} |\n")
    
    report.append("\n### 3. 性能瓶颈分析\n")
    
    bottlenecks = []
    for sched in sched_cores:
        parsed_tasks = sched['parsed_tasks']
        freq = sched['freq']
        
        for path in PERF_TRACE_CRITICAL_PATHS:
            result = calculate_critical_path_time(parsed_tasks, path, freq)
            if result and result['is_bottleneck']:
                bottlenecks.append({
                    'core_idx': sched['block_idx'],
                    'bottleneck': result
                })
    
    if bottlenecks:
        report.append("\n**识别到的瓶颈：**\n\n")
        for i, b in enumerate(bottlenecks, 1):
            bn = b['bottleneck']
            report.append(f"{i}. **{bn['description']}**\n")
            report.append(f"   - 核心索引: {b['core_idx']}\n")
            report.append(f"   - 打点路径: {bn['path']}\n")
            report.append(f"   - 耗时: {bn['time_us']:.2f} us\n")
            report.append(f"   - 阈值: {bn['threshold']} us\n")
            report.append(f"   - 状态: {bn['status']}\n\n")
    else:
        report.append("\n**未发现明显性能瓶颈。**\n\n")
    
    report.append("### 4. 关键耗时统计\n\n")
    
    for sched in sched_cores:
        parsed_tasks = sched['parsed_tasks']
        freq = sched['freq']
        
        report.append(f"\n**核心 {sched['block_idx']} 关键耗时：**\n")
        
        for path in PERF_TRACE_CRITICAL_PATHS:
            result = calculate_critical_path_time(parsed_tasks, path, freq)
            if result:
                report.append(f"- {result['description']}: {result['time_us']:.2f} us ({result['status']})\n")
    
    report.append("\n### 5. 优化建议\n\n")
    
    optimization_suggestions = {
        '任务接收延迟': [
            "检查任务构建复杂度",
            "合并小任务减少构建开销",
            "使用 cube_nbuffer_setting 合并同构子图"
        ],
        '依赖解析耗时': [
            "减少依赖层级，使用 loop_unroll 展开循环",
            "优化任务调度模式 device_sched_mode",
            "添加细化打点定位具体瓶颈"
        ],
        '任务分发耗时': [
            "增大任务粒度（tile size）",
            "启用 stitch 合并任务",
            "优化核心负载均衡"
        ],
        '任务发送耗时': [
            "检查任务发送频率",
            "优化任务参数处理"
        ]
    }
    
    if bottlenecks:
        for b in bottlenecks:
            bn = b['bottleneck']
            desc = bn['description']
            if desc in optimization_suggestions:
                report.append(f"\n**针对 {desc} 的优化建议：**\n")
                for suggestion in optimization_suggestions[desc]:
                    report.append(f"  - {suggestion}\n")
    else:
        report.append("\n当前调度性能良好，无明显优化需求。\n")
    
    return ''.join(report)


def load_swimlane_data(log_dir: str) -> Optional[List[Dict]]:
    swimlane_file = os.path.join(log_dir, 'merged_swimlane.json')
    
    if not os.path.exists(swimlane_file):
        print(f"错误：泳道图文件不存在: {swimlane_file}")
        return None
    
    with open(swimlane_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def analyze_swimlane_details(swimlane_data: List[Dict]) -> str:
    report = []
    report.append("##泳道图时间点详细分析\n\n")
    
    for core in swimlane_data:
        block_idx = core.get('blockIdx', -1)
        core_type = core.get('coreType', 'UNKNOWN')
        freq = core.get('freq', 1000)
        tasks = core.get('tasks', [])
        
        report.append(f"\n### 核心 {block_idx} ({core_type})\n\n")
        report.append(f"- 频率: {freq} MHz\n")
        report.append(f"- 总打点数: {len(tasks)}\n\n")
        
        if len(tasks) < 2:
            report.append("打点数过少，无法计算时间差\n")
            continue
        
        report.append("| 打点名称 | 开始时间 (us) | 结束时间 (us) | 耗时 (us) | 状态 |\n")
        report.append("|---------|-------------|-------------|----------|------|\n")
        
        prev_task = None
        large_duration_tasks = []
        
        for task in tasks:
            name = task.get('name', '')
            start_time = task.get('start', 0)
            end_time = task.get('end', 0)
            
            start_us = cycles_to_us(start_time, freq) if start_time else 0
            end_us = cycles_to_us(end_time, freq)
            
            if prev_task:
                prev_end_us = cycles_to_us(prev_task['end'], freq)
                duration = end_us - prev_end_us
                
                threshold = 50
                if 'DEV_TASK_DISPATCH' in name or 'RESOLVE' in name:
                    threshold = 200
                
                status = '超标' if duration > threshold else '正常'
                
                report.append(f"| {prev_task['name']} → {name} | {prev_end_us:.2f} | {end_us:.2f} | {duration:.2f} | {status} |\n")
                
                if duration > threshold:
                    large_duration_tasks.append({
                        'path': f"{prev_task['name']} → {name}",
                        'duration': duration,
                        'threshold': threshold
                    })
            
            prev_task = task
        
        if large_duration_tasks:
            report.append(f"\n**耗时较大的打点路径（需细化分析）：**\n\n")
            for item in large_duration_tasks:
                report.append(f"- {item['path']}: {item['duration']:.2f} us (阈值: {item['threshold']} us)\n")
                report.append(f"  建议：在该路径内部添加更细化的打点，定位具体耗时环节\n")
    
    return ''.join(report)


def compare_threads(perf_data: List[Dict]) -> str:
    report = []
    report.append("## 多线程对比分析\n\n")
    
    thread_stats = defaultdict(dict)
    
    for core_data in perf_data:
        analysis = analyze_core_perf(core_data)
        block_idx = analysis['block_idx']
        core_type = analysis['core_type']
        freq = analysis['freq']
        parsed_tasks = analysis['parsed_tasks']
        
        for path in PERF_TRACE_CRITICAL_PATHS:
            result = calculate_critical_path_time(parsed_tasks, path, freq)
            if result:
                key = result['description']
                if key not in thread_stats:
                    thread_stats[key] = {}
                thread_stats[key][block_idx] = result['time_us']
    
    report.append("### 关键路径耗时对比\n\n")
    
    for metric_name, thread_values in thread_stats.items():
        if len(thread_values) < 2:
            continue
        
        values = list(thread_values.values())
        mean_val = statistics.mean(values)
        if len(values) >= 2:
            std_val = statistics.stdev(values)
            variance_rate = (std_val / mean_val) * 100 if mean_val > 0 else 0
        else:
            std_val = 0
            variance_rate = 0
        
        report.append(f"\n**{metric_name}**\n\n")
        report.append(f"- 平均值: {mean_val:.2f} us\n")
        report.append(f"- 标准差: {std_val:.2f} us\n")
        report.append(f"- 波动率: {variance_rate:.2f}%\n\n")
        
        report.append("| 线程索引 | 耗时 (us) | 与平均值差异 |\n")
        report.append("|---------|----------|------------|\n")
        
        for block_idx, time_us in sorted(thread_values.items()):
            diff = time_us - mean_val
            diff_pct = (diff / mean_val) * 100 if mean_val > 0 else 0
            report.append(f"| {block_idx} | {time_us:.2f} | {diff:.2f} ({diff_pct:+.2f}%) |\n")
        
        if variance_rate > 30:
            report.append(f"\n**⚠️ 波动率超过 30%，需要细化分析！**\n")
            report.append(f"建议：在 `{metric_name}` 相关打点路径内添加更细化的打点\n")
    
    ctrl_sched_diff = []
    for core_data in perf_data:
        analysis = analyze_core_perf(core_data)
        if analysis['core_type'] in ['AICPU-CTRL', 'AICPU-SCHED']:
            freq = analysis['freq']
            parsed_tasks = analysis['parsed_tasks']
            
            if 'BEGIN' in parsed_tasks and 'EXIT' in parsed_tasks:
                begin_time = parsed_tasks['BEGIN'][0]['timestamp']
                exit_time = parsed_tasks['EXIT'][0]['timestamp']
                total_time = cycles_to_us(exit_time - begin_time, freq)
                ctrl_sched_diff.append({
                    'block_idx': analysis['block_idx'],
                    'core_type': analysis['core_type'],
                    'total_time': total_time
                })
    
    if ctrl_sched_diff:
        report.append("\n### 线程总耗时对比\n\n")
        report.append("| 线程索引 | 类型 | 总耗时 (us) |\n")
        report.append("|---------|------|----------|\n")
        for item in ctrl_sched_diff:
            report.append(f"| {item['block_idx']} | {item['core_type']} | {item['total_time']:.2f} |\n")
    
    return ''.join(report)


def analyze_multi_run_fluctuation(log_dirs: List[str]) -> str:
    report = []
    report.append("## 多次运行波动分析\n\n")
    
    all_runs_metrics = defaultdict(list)
    
    for log_dir in log_dirs:
        perf_data = load_perf_data(log_dir, turn=0)
        if perf_data is None:
            continue
        
        for core_data in perf_data:
            analysis = analyze_core_perf(core_data)
            freq = analysis['freq']
            parsed_tasks = analysis['parsed_tasks']
            
            for path in PERF_TRACE_CRITICAL_PATHS:
                result = calculate_critical_path_time(parsed_tasks, path, freq)
                if result:
                    key = result['description']
                    all_runs_metrics[key].append(result['time_us'])
    
    report.append("### 各指标波动统计\n\n")
    
    fluctuation_threshold = 15
    
    high_fluctuation_metrics = []
    
    for metric_name, values in all_runs_metrics.items():
        if len(values) < 2:
            continue
        
        mean_val = statistics.mean(values)
        min_val = min(values)
        max_val = max(values)
        
        if len(values) >= 2:
            std_val = statistics.stdev(values)
            variance_rate = (std_val / mean_val) * 100 if mean_val > 0 else 0
        else:
            std_val = 0
            variance_rate = 0
        
        status = '波动过大' if variance_rate > fluctuation_threshold else '稳定'
        
        report.append(f"\n**{metric_name}**\n\n")
        report.append(f"| 运行次数 | 耗时 (us) |\n")
        report.append("|---------|----------|\n")
        for i, val in enumerate(values, 1):
            diff = val - mean_val
            report.append(f"| Run {i} | {val:.2f} ({diff:+.2f}) |\n")
        
        report.append(f"\n- 平均值: {mean_val:.2f} us\n")
        report.append(f"- 最小值: {min_val:.2f} us\n")
        report.append(f"- 最大值: {max_val:.2f} us\n")
        report.append(f"- 标准差: {std_val:.2f} us\n")
        report.append(f"- 波动率: {variance_rate:.2f}% ({status})\n")
        
        if variance_rate > fluctuation_threshold:
            high_fluctuation_metrics.append({
                'name': metric_name,
                'variance_rate': variance_rate,
                'mean': mean_val,
                'std': std_val
            })
    
    if high_fluctuation_metrics:
        report.append("\n### 波动较大的指标（需细化分析）\n\n")
        
        for item in high_fluctuation_metrics:
            report.append(f"\n**{item['name']}**\n")
            report.append(f"- 波动率: {item['variance_rate']:.2f}% (> 15%)\n")
            report.append(f"- 平均耗时: {item['mean']:.2f} us\n")
            report.append(f"- 标准差: {item['std']:.2f} us\n\n")
            report.append(f"**优化建议：**\n")
            report.append(f"1. 在 `{item['name']}` 打点路径内添加更细化的打点\n")
            report.append(f"2. 分析波动原因：\n")
            report.append(f"   - 检查任务数量是否波动\n")
            report.append(f"   - 分析核心负载波动\n")
            report.append(f"   - 检查依赖解析复杂度波动\n")
            report.append(f"3. 针对最大耗时的运行，单独分析其调度流程\n\n")
    
    return ''.join(report)


def main():
    parser = argparse.ArgumentParser(
        description='PyPTO 调度性能分析脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 analyze_schedule_perf.py /path/to/logs
  python3 analyze_schedule_perf.py /path/to/logs --turn 0
  python3 analyze_schedule_perf.py /path/to/logs --output report.md
  python3 analyze_schedule_perf.py /path/to/logs --swimlane-details
  python3 analyze_schedule_perf.py /path/to/logs --compare-threads
  python3 analyze_schedule_perf.py /path/to/logs --multi-run --run-count 5
        """
    )
    
    parser.add_argument('log_dir', help='性能数据日志目录')
    parser.add_argument('--turn', type=int, default=0, help='分析轮次编号（默认: 0）')
    parser.add_argument('--output', help='输出报告文件路径（默认: 打印到屏幕）')
    parser.add_argument('--swimlane-details', action='store_true', help='详细分析泳道图时间点')
    parser.add_argument('--compare-threads', action='store_true', help='对比不同线程性能')
    parser.add_argument('--multi-run', action='store_true', help='多次运行波动分析')
    parser.add_argument('--run-count', type=int, default=5, help='多次运行的次数（默认: 5）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_dir):
        print(f"错误：日志目录不存在: {args.log_dir}")
        sys.exit(1)
    
    reports = []
    
    perf_data = load_perf_data(args.log_dir, args.turn)
    
    if perf_data is None:
        sys.exit(1)
    
    print(f"成功加载性能数据，共 {len(perf_data)} 个核心")
    
    if not args.swimlane_details and not args.compare_threads and not args.multi_run:
        report = generate_perf_report(perf_data, args.turn)
        reports.append(report)
    
    if args.swimlane_details:
        swimlane_data = load_swimlane_data(args.log_dir)
        if swimlane_data:
            print(f"成功加载泳道图数据")
            report = analyze_swimlane_details(swimlane_data)
            reports.append(report)
    
    if args.compare_threads:
        report = compare_threads(perf_data)
        reports.append(report)
    
    if args.multi_run:
        log_dirs = [args.log_dir]
        for i in range(1, args.run_count):
            potential_dir = os.path.join(os.path.dirname(args.log_dir), f"output_{i}")
            if os.path.exists(potential_dir):
                log_dirs.append(potential_dir)
        
        if len(log_dirs) < 2:
            print(f"警告：只找到 {len(log_dirs)} 个日志目录，波动分析可能不准确")
            print(f"提示：请多次运行测试用例，或使用 run_multi_perf_analysis.py 脚本")
        
        report = analyze_multi_run_fluctuation(log_dirs)
        reports.append(report)
    
    full_report = '\n\n'.join(reports)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"报告已保存到: {args.output}")
    else:
        print(full_report)


if __name__ == '__main__':
    main()