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
PyPTO 多次运行性能分析脚本

自动多次运行测试用例，收集性能数据，并分析波动。
"""

import argparse
import os
import sys
import subprocess
import json
import time
from typing import List, Dict
import statistics


def run_test_case(test_case: str, run_id: int, output_base_dir: str) -> str:
    env = os.environ.copy()
    env['DUMP_DEVICE_PERF'] = 'true'
    
    output_dir = os.path.join(output_base_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = ['pytest', test_case, '-v', '--tb=short']
    
    print(f"\n{'='*60}")
    print(f"运行第 {run_id} 次: {test_case}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"警告：测试运行失败（第 {run_id} 次）")
        print(f"错误输出:\n{result.stderr}")
    
    time.sleep(2)
    
    return output_dir


def find_latest_log_dir(base_output_dir: str) -> str:
    import glob
    log_pattern = os.path.join(base_output_dir, '*', 'logs_*')
    log_dirs = glob.glob(log_pattern)
    
    if not log_dirs:
        return None
    
    latest_log = max(log_dirs, key=os.path.getmtime)
    return latest_log


def collect_perf_data(run_dirs: List[str]) -> Dict:
    metrics_data = {}
    
    perf_trace_paths = [
        ('DEV_TASK_BUILD', 'DEV_TASK_RCV', '任务接收延迟'),
        ('DEV_TASK_ENTER_DISPATCH_TASK', 'DEV_TASK_DISPATCH_RESOL_TASK', '依赖解析耗时'),
        ('DEV_TASK_DISPATCH_RESOL_TASK', 'DEV_TASK_DISPATCH_TASK', '任务分发耗时'),
        ('DEV_TASK_DISPATCH_TASK', 'DEV_TASK_SEND_CALLOP_TASK', '任务发送耗时'),
        ('DEV_TASK_RUN_CORE_TASK', 'DEV_TASK_DISPATCH_TASK', '任务分发总耗时'),
    ]
    
    for run_id, run_dir in enumerate(run_dirs, 1):
        perf_file = os.path.join(run_dir, 'machine_trace_perf_data_0.json')
        
        if not os.path.exists(perf_file):
            print(f"警告：第 {run_id} 次运行未找到性能数据: {perf_file}")
            continue
        
        with open(perf_file, 'r', encoding='utf-8') as f:
            perf_data = json.load(f)
        
        for core in perf_data:
            freq = core.get('freq', 1000)
            tasks = core.get('tasks', [])
            block_idx = core.get('blockIdx', -1)
            
            task_times = {}
            for task in tasks:
                name = task.get('name', '')
                end_time = task.get('end', 0)
                import re
                clean_name = re.sub(r'\([0-9]+\)', '', name).strip()
                task_times[clean_name] = end_time
            
            for start_type, end_type, metric_name in perf_trace_paths:
                if start_type in task_times and end_type in task_times:
                    start_time = task_times[start_type]
                    end_time = task_times[end_type]
                    
                    if end_time > start_time:
                        time_cycles = end_time - start_time
                        time_us = time_cycles / freq
                        
                        key = f"{metric_name}_thread_{block_idx}"
                        if key not in metrics_data:
                            metrics_data[key] = []
                        metrics_data[key].append({
                            'run_id': run_id,
                            'time_us': time_us
                        })
    
    return metrics_data


def analyze_fluctuation(metrics_data: Dict) -> str:
    report = []
    report.append("## 多次运行波动分析报告\n\n")
    
    fluctuation_threshold = 15
    
    high_fluctuation_metrics = []
    
    for metric_key, values in sorted(metrics_data.items()):
        times = [v['time_us'] for v in values]
        
        if len(times) < 2:
            continue
        
        mean_val = statistics.mean(times)
        min_val = min(times)
        max_val = max(times)
        
        if len(times) >= 2:
            std_val = statistics.stdev(times)
            variance_rate = (std_val / mean_val) * 100 if mean_val > 0 else 0
        else:
            std_val = 0
            variance_rate = 0
        
        status = '波动过大' if variance_rate > fluctuation_threshold else '稳定'
        
        metric_name = metric_key.replace('_thread_', ' (线程 ')
        report.append(f"\n### {metric_name})\n\n")
        report.append(f"| 运行次数 | 耗时 (us) | 与平均值差异 |\n")
        report.append("|---------|----------|------------|\n")
        
        for v in values:
            diff = v['time_us'] - mean_val
            diff_pct = (diff / mean_val) * 100 if mean_val > 0 else 0
            report.append(f"| Run {v['run_id']} | {v['time_us']:.2f} | {diff:+.2f} ({diff_pct:+.2f}%) |\n")
        
        report.append(f"\n**统计信息：**\n")
        report.append(f"- 平均值: {mean_val:.2f} us\n")
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
        report.append("\n## 波动较大的指标（需细化分析）\n\n")
        report.append(f"以下指标波动率超过 {fluctuation_threshold}%，需要细化打点分析：\n\n")
        
        for item in high_fluctuation_metrics:
            report.append(f"\n### {item['name']})\n\n")
            report.append(f"- 波动率: {item['variance_rate']:.2f}% (> {fluctuation_threshold}%)\n")
            report.append(f"- 平均耗时: {item['mean']:.2f} us\n")
            report.append(f"- 标准差: {item['std']:.2f} us\n\n")
            
            report.append(f"**优化建议：**\n\n")
            report.append(f"1. **添加细化打点**\n")
            report.append(f"   在相关打点路径内添加更细化的 PerfMtTrace 打点\n\n")
            report.append(f"2. **分析波动原因**\n")
            report.append(f"   - 检查任务数量是否波动（不同运行的任务数差异）\n")
            report.append(f"   - 分析核心负载波动（activeCoreIdx_ 数量变化）\n")
            report.append(f"   - 检查依赖解析复杂度波动（寄存器读取频率）\n")
            report.append(f"   - 分析任务队列长度波动\n\n")
            report.append(f"3. **针对性优化**\n")
            report.append(f"   - 针对 Run {max_val} 单独分析其调度流程\n")
            report.append(f"   - 对比 Run {min_val} 和 Run {max_val} 的打点数据\n")
            report.append(f"   - 找出导致最大耗时的具体环节\n\n")
    
    if not high_fluctuation_metrics:
        report.append("\n## 总结\n\n")
        report.append(f"所有指标波动率均小于 {fluctuation_threshold}%，调度性能稳定。\n")
    
    return ''.join(report)


def main():
    parser = argparse.ArgumentParser(
        description='PyPTO 多次运行性能分析脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 run_multi_perf_analysis.py --test-case python/tests/st/test_swim_line.py::test_swim
  python3 run_multi_perf_analysis.py --test-case python/tests/st/test_swim_line.py::test_swim --run-count 10
  python3 run_multi_perf_analysis.py --test-case test.py --output-dir ./multi_run_logs --run-count 5
        """
    )
    
    parser.add_argument('--test-case', required=True, help='测试用例路径（pytest 格式）')
    parser.add_argument('--run-count', type=int, default=5, help='运行次数（默认: 5）')
    parser.add_argument('--output-dir', default='./multi_run_output', help='输出目录')
    parser.add_argument('--report-output', help='报告输出文件路径')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n配置信息：")
    print(f"- 测试用例: {args.test_case}")
    print(f"- 运行次数: {args.run_count}")
    print(f"- 输出目录: {args.output_dir}")
    print(f"\n开始多次运行...\n")
    
    run_dirs = []
    
    for i in range(1, args.run_count + 1):
        run_dir = run_test_case(args.test_case, i, args.output_dir)
        run_dirs.append(run_dir)
    
    print(f"\n\n{'='*60}")
    print(f"完成 {args.run_count} 次运行")
    print(f"{'='*60}\n")
    
    print(f"\n开始收集性能数据...\n")
    
    metrics_data = collect_perf_data(run_dirs)
    
    print(f"收集到 {len(metrics_data)} 个性能指标")
    
    report = analyze_fluctuation(metrics_data)
    
    if args.report_output:
        with open(args.report_output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存到: {args.report_output}")
    else:
        print(report)
    
    summary_file = os.path.join(args.output_dir, 'run_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_case': args.test_case,
            'run_count': args.run_count,
            'run_dirs': run_dirs,
            'metrics_count': len(metrics_data)
        }, f, indent=2)
    
    print(f"\n运行摘要已保存到: {summary_file}")


if __name__ == '__main__':
    main()