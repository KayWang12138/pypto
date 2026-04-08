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
批次间依赖间隙分析脚本
分析 AIC/AIV 任务批次转换点的依赖间隙，记录每一个独立间隙。
"""

import json
import logging
import os
import argparse
import sys
from collections import defaultdict


def setup_logging():
    """设置日志，替代 print"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def analyze_batch_gaps(data_path, topo_path, logger):
    """分析并记录每一个批次间的依赖间隙"""
    if not os.path.exists(data_path) or not os.path.exists(topo_path):
        logger.error(f"Missing required files: {data_path} or {topo_path}")
        return None

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            swimlane_data = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {data_path}")
        return None

    tasks = {}
    thread_names = {}

    for event in swimlane_data.get('traceEvents', []):
        if event.get('ph') == 'M' and event.get('name') == 'thread_name':
            thread_id = event.get('tid')
            thread_names[thread_id] = event.get('args', {}).get('name', '')

    for event in swimlane_data.get('traceEvents', []):
        if event.get('ph') == 'X':
            args = event.get('args', {})
            event_hint = args.get('event-hint', '')

            if 'TaskId:' in event_hint:
                task_id = int(event_hint.split('TaskId:')[1].split(',')[0])
                task_name = event.get('name', '')
                is_fake = '(fake)' in task_name
                start_time = event.get('ts', 0)
                duration = event.get('dur', 0)
                thread_id = event.get('tid', 0)

                tasks[task_id] = {
                    'task_id': task_id,
                    'start': start_time,
                    'end': start_time + duration,
                    'thread_name': thread_names.get(thread_id, f'thread_{thread_id}'),
                    'is_fake': is_fake
                }

    # 解析依赖关系
    deps = defaultdict(set)
    try:
        with open(topo_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    task_id = int(parts[1])
                    successors = [int(p.strip()) for p in parts[9:] if p.strip()]
                    if successors:
                        deps[task_id] = set(successors)
    except Exception as e:
        logger.error(f"Error parsing topo file: {e}")
        return None

    # 获取有效任务并排序
    real_tasks = {tid: t for tid, t in tasks.items() if not t['is_fake']}
    aic_tasks = [t for t in real_tasks.values() if t['thread_name'].startswith('AIC_')]
    aiv_tasks = [t for t in real_tasks.values() if t['thread_name'].startswith('AIV_')]

    if not aic_tasks or not aiv_tasks:
        logger.warning("No valid AIC or AIV tasks found.")
        return None

    all_tasks = sorted(aic_tasks + aiv_tasks, key=lambda t: t['start'])
    batch_gaps = []

    # 核心：找出批次转换点，记录每一次的独立间隙
    for i in range(len(all_tasks) - 1):
        current = all_tasks[i]
        next_task = all_tasks[i + 1]

        c_is_aic = current['thread_name'].startswith('AIC_')
        n_is_aiv = next_task['thread_name'].startswith('AIV_')
        c_is_aiv = current['thread_name'].startswith('AIV_')
        n_is_aic = next_task['thread_name'].startswith('AIC_')

        is_aic_to_aiv = c_is_aic and n_is_aiv
        is_aiv_to_aic = c_is_aiv and n_is_aic

        if is_aic_to_aiv or is_aiv_to_aic:
            has_dependency = (
                current['task_id'] in deps and
                next_task['task_id'] in deps[current['task_id']]
            )

            # 使用 max 避免时间戳重叠产生负间隙干扰
            gap_us = max(0.0, next_task['start'] - current['end'])

            batch_gaps.append({
                'transition_index': len(batch_gaps),
                'from_task_id': current['task_id'],
                'to_task_id': next_task['task_id'],
                'gap_us': gap_us,
                'type': f"{current['thread_name'][:3]} -> {next_task['thread_name'][:3]}",
                'has_dependency': has_dependency
            })

    return batch_gaps


def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(description="Extract batch gaps from Swimlane & Topo")
    parser.add_argument("--swimlane", type=str, required=True, help="Path to merged_swimlane.json")
    parser.add_argument("--topo", type=str, required=True, help="Path to dyn_topo.txt")
    parser.add_argument("--output", type=str, required=True, help="Path to save the JSON result")
    args = parser.parse_args()

    gaps = analyze_batch_gaps(args.swimlane, args.topo, logger)

    if not gaps:
        logger.error("Failed to extract any batch gaps.")
        sys.exit(125)

    result = {
        "status": "success",
        "total_transitions": len(gaps),
        "transitions": gaps
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

    logger.info(f"Successfully recorded {len(gaps)} batch transitions to {args.output}")


if __name__ == "__main__":
    main()