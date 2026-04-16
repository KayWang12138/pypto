#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 MatMul AllReduce Add RmsNorm Module for performance

This module implements a fused matmul, all-reduce, add, and RMSNorm operation for large-scale distributed models.
It efficiently combines computation and communication, reducing memory overhead and accelerating training and inference.

Main Functions:
    - matmul_allreduce_add_rmsnorm: Main function for fused matmul, all-reduce, add, and RMSNorm computation
"""

import multiprocessing as mp
import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pytest
import torch

import pypto

from distributed_config import DistributedConfig


@pypto.frontend.jit(
    debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1},
    runtime_options={"stitch_function_max_num": 128,
                     "stitch_cfgcache_size": 100000000},
)
def matmul_allreduce_add_rmsnorm_kernel(
    in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    matmul_weight: pypto.Tensor(),
    residual: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    gamma: pypto.Tensor(),
    bias: pypto.Tensor(),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    residual_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    eps,
    group_name,
    world_size,
):
    batch_size = in_tensor.shape[0]
    hidden_size = matmul_weight.shape[0]

    in_tensor_mean_coff = 1.0 / hidden_size
    view_row_shape = 8
    bs_loop = (batch_size + view_row_shape - 1) // view_row_shape

    pypto.set_vec_tile_shapes(hidden_size)
    gamma_2d = pypto.reshape(gamma, [1, hidden_size], inplace=True)
    bias_2d = pypto.reshape(bias, [1, hidden_size], inplace=True)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_MM_ALLREDUCE_ADD_RMSNORM", idx_name="bs_idx"):
        shmem_shape = [view_row_shape, hidden_size]
        shmem_tensor = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_FP32, shmem_shape)
        shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for _ in pypto.loop(1, name="LOOP_MM_AR_ARMS_L0", idx_name="_"):
            in_tensor_tile = pypto.view(
                in_tensor, (view_row_shape, in_tensor.shape[1]), [bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])

            pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
            data_clear_out = pypto.distributed.shmem_clear_data(
                shmem_tensor, shmem_shape, [0, 0], pred=[in_tensor_tile])
            signal_clear_out = pypto.distributed.shmem_clear_signal(
                shmem_tensor, pred=[in_tensor_tile])
            barrier_out = pypto.distributed.shmem_barrier_all(
                shmem_barrier_signal, [data_clear_out, signal_clear_out])

            pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])
            matmul_result = pypto.matmul(in_tensor_tile, matmul_weight, pypto.DT_FP32, b_trans=True)

            pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
            for dyn_idx in range(world_size):
                put_out = pypto.distributed.shmem_put(matmul_result, [0, 0], shmem_tensor, dyn_idx,
                    put_op=pypto.AtomicType.ADD, pred=[barrier_out])
                pypto.distributed.shmem_signal(shmem_tensor, dyn_idx, 1, shmem_shape,
                    [0, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.ADD, pred=[put_out])
            wait_until_out = pypto.distributed.shmem_wait_until(shmem_tensor, my_pe, world_size,
                shmem_shape, [0, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[in_tensor_tile])
            pypto.set_vec_tile_shapes(1, hidden_size)
            all_reduce_out = pypto.experimental.shmem_load(
                shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out],
                valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), hidden_size]
            )

            residual_tile = pypto.view(
                residual, (view_row_shape, hidden_size), [bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - bs_idx * view_row_shape).min(view_row_shape), hidden_size])

            residual_tile_fp32 = pypto.cast(residual_tile, pypto.DT_FP32)
            add_out = pypto.add(all_reduce_out, residual_tile_fp32)

            square = pypto.mul(add_out, add_out)
            mean_res = pypto.mul(square, in_tensor_mean_coff)
            reduce_asum = pypto.sum(mean_res, -1, True)
            reduce_sum = pypto.add(reduce_asum, eps)
            reduce_sqrt = pypto.sqrt(reduce_sum)
            res_div = pypto.div(add_out, reduce_sqrt)

            hidden_bf16 = pypto.tensor([view_row_shape, hidden_size], pypto.DT_BF16, "hidden_bf16")
            residual_bf16_tmp = pypto.cast(add_out, in_tensor.dtype)
            for tmp_idx in range(view_row_shape):
                gamma_2d_fp32 = pypto.cast(gamma_2d, pypto.DT_FP32)
                bias_2d_fp32 = pypto.cast(bias_2d, pypto.DT_FP32)
                res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])
                res = pypto.mul(res_div_single, gamma_2d_fp32)
                res_add = pypto.add(res, bias_2d_fp32)
                in_tensor_norm = pypto.cast(res_add, in_tensor.dtype)
                hidden_bf16[tmp_idx:tmp_idx + 1] = in_tensor_norm

            residual_out[bs_idx * pypto.symbolic_scalar(view_row_shape):] = residual_bf16_tmp
            out_tensor[bs_idx * pypto.symbolic_scalar(view_row_shape):] = hidden_bf16


def generate_golden_data(world_size: int):
    batch_size = 8
    attn_dim_per_tp = 1536
    hidden_size = 5120
    torch.manual_seed(42)

    input_datas = []
    for _ in range(world_size):
        in_tensor = torch.randn((batch_size, attn_dim_per_tp), dtype=torch.bfloat16).share_memory_()
        matmul_weight = torch.randn((hidden_size, attn_dim_per_tp), dtype=torch.bfloat16).share_memory_()
        residual = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16).share_memory_()
        gamma = torch.randn((hidden_size), dtype=torch.bfloat16).share_memory_()
        bias = torch.randn((hidden_size), dtype=torch.bfloat16).share_memory_()
        eps = 1e-5
        input_data = [in_tensor, matmul_weight, residual, gamma, bias, eps]
        input_datas.append(input_data)
    output_datas = matmul_allreduce_add_rmsnorm_result_golden(batch_size, hidden_size, input_datas)
    return input_datas, output_datas


def matmul_allreduce_add_rmsnorm_result_golden(batch_size, num, input_datas):
    output_datas = []
    matmul_allreduce_result_fp32 = torch.zeros((batch_size, num), dtype=torch.float32)
    for input_data in input_datas:
        in_tensor, matmul_weight = input_data[:2]
        matmul_result = torch.matmul(in_tensor.to(torch.float32), matmul_weight.to(torch.float32).T)
        matmul_allreduce_result_fp32 += matmul_result

    for input_data in input_datas:
        residual, gamma, bias, eps = input_data[-4:]
        res_add = residual.to(torch.float32) + matmul_allreduce_result_fp32
        mean_coff = 1.0 / res_add.shape[-1]
        in_tensor_f32 = res_add
        square = in_tensor_f32 * in_tensor_f32
        square = square.sum(dim=-1, keepdim=True)
        mean_res = square * mean_coff
        reduce_sum = mean_res + eps
        reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = in_tensor_f32 / reduce_sqrt
        res = res_div * gamma.to(torch.float32)
        res = res + bias.to(res.dtype)
        output_data = [res.to(torch.bfloat16), in_tensor_f32.to(torch.bfloat16)]
        output_datas.append(output_data)
    return output_datas


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


def matmul_allreduce_add_rmsnorm_worker(
    config: DistributedConfig,
    input_data: list,
    output_data: list,
    logical_rank_id: int,
    output_dir: str,
):
    groups = config.init_hccl_comm(logical_rank_id)
    physical_device_id = config.get_physical_device_id(logical_rank_id)
    device = f'npu:{physical_device_id}'

    in_tensor, matmul_weight, residual, gamma, bias, eps = input_data
    golden_out_tensor, golden_residual = output_data

    out_tensor = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)
    residual_out = torch.empty(residual.shape, dtype=torch.bfloat16, device=device)

    inputs = [in_tensor.to(device), matmul_weight.to(device), residual.to(device), gamma.to(device),
        bias.to(device), out_tensor, residual_out]

    matmul_allreduce_add_rmsnorm_kernel(*inputs, eps, groups[0], config.world_size)

    np.testing.assert_allclose(
        np.array(out_tensor.cpu().flatten().tolist()),
        np.array(golden_out_tensor.cpu().flatten().tolist()),
        rtol=8e-3,
        atol=8e-3,
    )

    np.testing.assert_allclose(
        np.array(residual_out.cpu().flatten().tolist()),
        np.array(golden_residual.cpu().flatten().tolist()),
        rtol=8e-3,
        atol=8e-3,
    )


@pytest.mark.skip(reason="performance test case")
def test_matmul_allreduce_add_rmsnorm_performance():
    mp.set_start_method('spawn', force=True)
    config = DistributedConfig(world_size=2)

    # 创建输出目录
    output_name = f"output"
    output_dir = f"{Path.cwd()}/{output_name}"
    os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir

    ##################################################################
    # 在这里设置您的理想整体花费时间（预期总时间）
    # 单位：微秒 (us)
    ##################################################################
    expected_total_time = 70.0  # 设置为您期望的总时间
    ##################################################################

    # CSV文件名
    csv_file = "performance_results.csv"

    processes = []
    input_datas, output_datas = generate_golden_data(config.world_size)

    for i in range(config.world_size):
        p = mp.Process(target=matmul_allreduce_add_rmsnorm_worker,
                      args=(config, input_datas[i], output_datas[i], i, output_dir))
        p.start()
        processes.append(p)

    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise AssertionError(f"process {i} failed, return: {p.exitcode}")

    try:
        # 分析总体执行时间
        analyzer = SwimlaneAnalyzer(output_dir, expected_total_time=expected_total_time)

        # 打印swimlane文件信息
        analyzer.print_swimlane_files_info(config.world_size)

        # 生成对比报告
        report = analyzer.generate_comparison_report(config.world_size, debug=False)
        print(report)

        # 保存结果到CSV
        csv_path = analyzer.save_to_csv(config.world_size, csv_file)
        print(f"结果已保存到: {csv_path}")

        # 检查实际总体执行时间是否在预期总时间内
        is_within = analyzer.check_within_expected(config.world_size)

        if not is_within:
            actual_min_time = analyzer.calculate_min_total_time(config.world_size)
            pytest.fail(f"总体执行时间超出预期: 实际{actual_min_time:.3f}us > 预期{expected_total_time:.3f}us")
        else:
            print("✓ 总体执行时间在预期范围内")

    finally:
        print(f"\n性能数据保存在: {output_dir}")


def summarize_performance_results(csv_file: str = "performance_results.csv"):
    """汇总并显示所有性能测试结果"""
    if not os.path.exists(csv_file):
        print(f"CSV文件不存在: {csv_file}")
        return

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("CSV文件为空")
        return

    print("=" * 80)
    print("性能测试结果汇总")
    print("=" * 80)

    # 打印表头
    print(f"{'序号':<4} {'时间':<20} {'world_size':<10} {'rank':<8} {'预期时间(us)':<12} {'实际总体时间(us)':<16} {'状态':<8}")
    print("-" * 80)

    # 打印每一行
    for i, row in enumerate(rows, 1):
        timestamp = row.get('timestamp', 'N/A')
        world_size = row.get('world_size', 'N/A')
        rank = row.get('rank', 'N/A')
        expected_time = row.get('expected_time_us', 'N/A')
        actual_time = row.get('actual_total_time_us', 'N/A')
        status = row.get('status', 'N/A')

        print(f"{i:<4} {timestamp:<20} {world_size:<10} {rank:<8} {expected_time:<12} {actual_time:<16} {status:<8}")

    # 统计
    total_tests = len(rows)
    passed_tests = sum(1 for row in rows if row.get('status') == 'PASS')
    failed_tests = total_tests - passed_tests

    print("-" * 80)
    print(f"总计测试: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {failed_tests}")

    if total_tests > 0:
        # 按world_size分组统计
        ws_stats = {}
        for row in rows:
            ws = row.get('world_size', 'unknown')
            if ws not in ws_stats:
                ws_stats[ws] = {'count': 0, 'times': [], 'passed': 0}

            ws_stats[ws]['count'] += 1
            if row.get('actual_total_time_us', 'N/A') != 'N/A':
                try:
                    ws_stats[ws]['times'].append(float(row['actual_total_time_us']))
                except:
                    pass
            if row.get('status') == 'PASS':
                ws_stats[ws]['passed'] += 1

        if len(ws_stats) > 0:
            print(f"\n按world_size统计:")
            for ws, stats in sorted(ws_stats.items()):
                if stats['times']:
                    avg_time = sum(stats['times']) / len(stats['times'])
                    min_time = min(stats['times'])
                    max_time = max(stats['times'])
                    pass_rate = (stats['passed'] / stats['count'] * 100) if stats['count'] > 0 else 0

                    print(f"  world_size={ws}:")
                    print(f"    测试次数: {stats['count']}, 通过率: {pass_rate:.1f}%")
                    print(f"    平均时间: {avg_time:.3f} us, 最小: {min_time:.3f} us, 最大: {max_time:.3f} us")

    print("=" * 80)


def main():
    # 运行测试
    test_matmul_allreduce_add_rmsnorm_performance()

    # 显示汇总结果
    print("\n")
    summarize_performance_results()


if __name__ == '__main__':
    main()