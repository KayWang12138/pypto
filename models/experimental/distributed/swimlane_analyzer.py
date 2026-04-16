#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---
"""
Performance use
"""

import os
import json
import csv
from typing import List, Optional, Tuple
from datetime import datetime

class SwimlaneAnalyzer:
    def __init__(self, output_dir: str, expected_total_time: Optional[float] = None):
        """
        初始化Swimlane分析器

        参数:
        output_dir: 输出目录路径
        expected_total_time: 预期总时间（微秒），在测试函数中设置
        """
        self.output_dir = output_dir
        self.performance_data = None
        self.expected_total_time = expected_total_time
        self._cached_min_file_info = None  # 缓存最小文件信息

    def find_all_rank_dirs(self) -> List[str]:
        """查找所有rank目录"""
        rank_dirs = []

        # 在输出目录中查找rank目录
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path):
                # 检查是否是rank目录（包含rank字样）
                if 'rank' in item.lower():
                    rank_dirs.append(item_path)

        # 如果没找到rank目录，则返回所有子目录
        if not rank_dirs:
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    rank_dirs.append(item_path)

        return rank_dirs

    def find_recent_rank_dirs(self, world_size: int) -> List[str]:
        """查找最近的world_size个rank目录"""
        rank_dirs = self.find_all_rank_dirs()

        if not rank_dirs:
            return []

        # 按修改时间排序，最新的在前面
        rank_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # 返回最近的world_size个目录
        return rank_dirs[:world_size]

    def find_min_time_file_in_ranks(self, world_size: int) -> Tuple[str, float]:
        """
        在最近的world_size个rank目录中找到耗时最少的文件

        返回:
        (文件路径, 最小耗时)
        """
        # 如果已经有缓存，直接返回缓存结果
        if self._cached_min_file_info is not None:
            return self._cached_min_file_info

        rank_dirs = self.find_recent_rank_dirs(world_size)

        if not rank_dirs:
            raise FileNotFoundError(f"No rank directories found in {self.output_dir}")

        min_time = float('inf')
        min_file = None
        all_times = []

        for rank_dir in rank_dirs:
            # 在每个rank目录中查找merged_swimlane.json
            swimlane_files = []
            for root, dirs, files in os.walk(rank_dir):
                for file in files:
                    if file == "merged_swimlane.json":
                        swimlane_files.append(os.path.join(root, file))

            for file_path in swimlane_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    total_time = self._calculate_total_time_from_data(data)
                    all_times.append((file_path, total_time))

                    if total_time < min_time:
                        min_time = total_time
                        min_file = file_path

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

        if min_file is None:
            raise ValueError("Could not find any valid swimlane file")

        # 缓存结果
        self._cached_min_file_info = (min_file, min_time)

        return min_file, min_time

    def _calculate_total_time_from_data(self, performance_data: dict) -> float:
        """从性能数据计算总体执行时间（第一个任务开始到最后一个任务结束的时间跨度）"""
        if 'traceEvents' not in performance_data:
            raise ValueError("Performance data does not contain traceEvents")

        # 收集所有真实任务的开始时间和结束时间
        start_times = []
        end_times = []

        for event in performance_data['traceEvents']:
            if event.get('ph') == 'X':  # 执行事件
                # 跳过fake事件
                if 'fake' in event.get('name', '').lower():
                    continue

                # 获取任务开始时间和持续时间
                start_time = event.get('ts', 0)
                duration = event.get('dur', 0)

                if duration <= 0:  # 跳过无效时间
                    continue

                end_time = start_time + duration
                start_times.append(start_time)
                end_times.append(end_time)

        if not start_times or not end_times:
            return 0.0

        # 计算总体时间跨度
        overall_start = min(start_times)
        overall_end = max(end_times)
        total_time = overall_end - overall_start

        return total_time

    def calculate_min_total_time(self, world_size: int) -> float:
        """计算所有rank文件中总体执行时间的最小值"""
        _, min_time = self.find_min_time_file_in_ranks(world_size)
        return min_time

    def generate_comparison_report(self, world_size: int, debug: bool = True) -> str:
        """生成实际最小总体执行时间与预期总时间的对比报告"""
        actual_min_time = self.calculate_min_total_time(world_size)
        expected_total_time = self.expected_total_time
        min_file_path, _ = self.find_min_time_file_in_ranks(world_size)

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("Swimlane总体执行时间对比报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        report_lines.append(f"分析目录: {self.output_dir}")
        report_lines.append(f"World Size: {world_size}")
        report_lines.append(f"最小耗时文件: {os.path.basename(os.path.dirname(min_file_path))}")
        report_lines.append(f"文件路径: {min_file_path}")
        report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        if expected_total_time is not None:
            is_within = actual_min_time <= expected_total_time
            diff = actual_min_time - expected_total_time

            if is_within:
                status = "✓"
                status_text = f"实际总体执行时间在预期范围内"
            else:
                status = "✗"
                status_text = f"实际总体执行时间超出预期"

            report_lines.append(f"实际总体执行时间: {actual_min_time:.3f} us")
            report_lines.append(f"预期总时间: {expected_total_time:.3f} us")
            report_lines.append(f"差值: {diff:+.3f} us ({diff/expected_total_time*100:.1f}%)")
            report_lines.append(f"状态: {status} {status_text}")
        else:
            report_lines.append(f"实际总体执行时间: {actual_min_time:.3f} us")
            report_lines.append(f"预期总时间: 未设置")

        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def check_within_expected(self, world_size: int) -> bool:
        """检查实际总体执行时间是否在预期总时间内"""
        if self.expected_total_time is None:
            return True  # 如果没有设置预期时间，默认通过

        actual_min_time = self.calculate_min_total_time(world_size)
        return actual_min_time <= self.expected_total_time

    def save_to_csv(self, world_size: int, csv_file: str = "performance_results.csv"):
        """将结果保存到CSV文件"""
        actual_min_time = self.calculate_min_total_time(world_size)
        min_file_path, _ = self.find_min_time_file_in_ranks(world_size)
        expected_total_time = self.expected_total_time
        is_within = self.check_within_expected(world_size)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 获取rank信息（从文件路径推断）
        rank = "unknown"
        rank_dir = os.path.dirname(min_file_path)
        for part in rank_dir.split(os.sep):
            if 'rank' in part.lower():
                rank = part
                break

        # CSV文件头
        fieldnames = [
            'timestamp',
            'world_size',
            'rank',
            'expected_time_us',
            'actual_total_time_us',
            'is_within_expected',
            'min_file_path',
            'status'
        ]

        # 确保CSV文件存在并写入数据
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # 确保预期时间被正确保存
            expected_value = expected_total_time if expected_total_time is not None else 'N/A'

            writer.writerow({
                'timestamp': timestamp,
                'world_size': world_size,
                'rank': rank,
                'expected_time_us': expected_value,
                'actual_total_time_us': round(actual_min_time, 3),
                'is_within_expected': is_within,
                'min_file_path': min_file_path,
                'status': 'PASS' if is_within else 'FAIL'
            })

        return csv_file

    def print_swimlane_files_info(self, world_size: int):
        """打印找到的swimlane文件信息"""
        if self._cached_min_file_info is None:
            # 如果没有缓存，先计算一次
            self.find_min_time_file_in_ranks(world_size)

        # 重新计算以获取所有文件的时间
        rank_dirs = self.find_recent_rank_dirs(world_size)

        if not rank_dirs:
            print(f"No rank directories found in {self.output_dir}")
            return

        all_times = []
        for rank_dir in rank_dirs:
            # 在每个rank目录中查找merged_swimlane.json
            swimlane_files = []
            for root, dirs, files in os.walk(rank_dir):
                for file in files:
                    if file == "merged_swimlane.json":
                        swimlane_files.append(os.path.join(root, file))

            for file_path in swimlane_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    total_time = self._calculate_total_time_from_data(data)
                    all_times.append((file_path, total_time))

                except Exception as e:
                    continue

        if all_times:
            print(f"Found {len(all_times)} swimlane files with times:")
            for file_path, time in sorted(all_times, key=lambda x: x[1]):
                rank_name = os.path.basename(os.path.dirname(file_path))
                print(f"  {rank_name}: {time:.3f} us")