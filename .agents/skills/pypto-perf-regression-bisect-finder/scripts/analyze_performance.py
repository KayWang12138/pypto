#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO 性能分析脚本
根据用户定义的分析方法提取性能指标
"""

import json
import os
import sys
import glob
import argparse


def extract_metric(swimlane_file, analysis_method):
    """
    根据用户定义的分析方法提取性能指标

    Args:
        swimlane_file: 泳道图文件路径
        analysis_method: 用户定义的分析方法

    Returns:
        提取的性能指标值
    """
    with open(swimlane_file, 'r') as f:
        data = json.load(f)

    trace = data.get('traceEvents', [])

    # 获取线程名称映射
    thread_names = {}
    for e in trace:
        if e.get('name') == 'thread_name':
            tid = e.get('tid')
            thread_names[tid] = e.get('args', {}).get('name', '')

    # 根据用户定义的分析方法提取指标

    # 示例 1：计算最后一个 AIC 完成时间和第一个 AIV 开始时间的差值
    if analysis_method == "aic_last_to_aiv_first":
        aic_events = []
        aiv_events = []

        for e in trace:
            if e.get('ph') in ['X', 'B', 'E']:
                tid = e.get('tid')
                thread_name = thread_names.get(tid, '')
                ts = e.get('ts', 0)
                dur = e.get('dur', 0)

                if thread_name.startswith('AIC'):
                    aic_events.append({
                        'start': ts,
                        'end': ts + dur,
                        'dur': dur
                    })
                elif thread_name.startswith('AIV'):
                    aiv_events.append({
                        'start': ts,
                        'end': ts + dur,
                        'dur': dur
                    })

        if aic_events and aiv_events:
            aic_latest_end = max([e['end'] for e in aic_events])
            aiv_first_start = min([e['start'] for e in aiv_events])
            gap = aiv_first_start - aic_latest_end
            return gap

    # 示例 2：计算第一个 AIC 完成时间和第一个 AIV 开始时间的差值
    elif analysis_method == "aic_first_to_aiv_first":
        aic_events = []
        aiv_events = []

        for e in trace:
            if e.get('ph') in ['X', 'B', 'E']:
                tid = e.get('tid')
                thread_name = thread_names.get(tid, '')
                ts = e.get('ts', 0)
                dur = e.get('dur', 0)

                if thread_name.startswith('AIC'):
                    aic_events.append({
                        'start': ts,
                        'end': ts + dur,
                        'dur': dur
                    })
                elif thread_name.startswith('AIV'):
                    aiv_events.append({
                        'start': ts,
                        'end': ts + dur,
                        'dur': dur
                    })

        if aic_events and aiv_events:
            aic_first_end = min([e['end'] for e in aic_events])
            aiv_first_start = min([e['start'] for e in aiv_events])
            gap = aiv_first_start - aic_first_end
            return gap

    # 示例 3：计算总执行时间
    elif analysis_method == "total_time":
        if trace:
            first_ts = min([e.get('ts', 0) for e in trace if e.get('ts')])
            last_end = max([e.get('ts', 0) + e.get('dur', 0) for e in trace if e.get('ts')])
            total_time = last_end - first_ts
            return total_time

    return None


def judge_performance(metric_value, good_threshold, bad_threshold):
    """
    根据性能指标值判断版本类型

    Args:
        metric_value: 性能指标值
        good_threshold: Good 版本阈值
        bad_threshold: Bad 版本阈值

    Returns:
        'good', 'bad', 或 'unknown'
    """
    if metric_value is None:
        return 'unknown'

    if metric_value < good_threshold:
        return 'good'
    elif metric_value >= bad_threshold:
        return 'bad'
    else:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='PyPTO 性能分析工具')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    parser.add_argument('--analysis-method', type=str, required=True,
                       help='分析方法（由用户提供）')
    parser.add_argument('--good-threshold', type=float, required=True,
                       help='Good 版本阈值')
    parser.add_argument('--bad-threshold', type=float, required=True,
                       help='Bad 版本阈值')

    args = parser.parse_args()

    # 查找最新的泳道图文件
    swimlane_files = glob.glob(f'{args.output_dir}/output_*/merged_swimlane.json')

    if not swimlane_files:
        sys.exit(1)

    # 分析最新的文件
    latest_file = max(swimlane_files, key=os.path.getmtime)

    # 提取性能指标
    metric_value = extract_metric(latest_file, args.analysis_method)

    if metric_value is None:
        sys.exit(1)

    # 判断性能
    result = judge_performance(metric_value, args.good_threshold, args.bad_threshold)

    if result == 'good':
        sys.exit(0)
    elif result == 'bad':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == '__main__':
    main()
