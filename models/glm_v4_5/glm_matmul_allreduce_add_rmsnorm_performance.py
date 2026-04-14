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

import multiprocessing as mp
import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pytest
import torch

import pypto

from utils.distributed_config import DistributedConfig


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
                shmem_tensor, my_pe, shmem_shape, [0, 0], pred=[wait_until_out], valid_shape=shmem_shape
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


class PerformanceAnalyzer:
    def __init__(self, output_dir: str, time_thresholds: Optional[Dict[str, float]] = None):
        self.output_dir = output_dir
        self.latest_dir = self._find_latest_output_dir()
        self.performance_data = None
        self.time_thresholds = time_thresholds or {}

    def _find_latest_output_dir(self) -> str:
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")
        
        subdirs = [os.path.join(self.output_dir, d) for d in os.listdir(self.output_dir)
                  if os.path.isdir(os.path.join(self.output_dir, d))]
        
        if not subdirs:
            raise FileNotFoundError(f"No output subdirectories found in {self.output_dir}")
        
        return max(subdirs, key=os.path.getctime)

    def load_swimlane_data(self) -> dict:
        swimlane_path = os.path.join(self.latest_dir, "merged_swimlane.json")
        if not os.path.exists(swimlane_path):
            raise FileNotFoundError(f"Swimlane data file not found: {swimlane_path}")
        
        with open(swimlane_path, 'r') as f:
            self.performance_data = json.load(f)
        
        return self.performance_data

    def load_bubble_analysis(self) -> dict:
        bubble_path = os.path.join(self.latest_dir, "bubble_analysis.log")
        if not os.path.exists(bubble_path):
            return {}
        
        bubble_data = {}
        with open(bubble_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    bubble_data[key.strip()] = value.strip()
        
        return bubble_data

    def load_baseline_swimlanes(self, baseline_file: str = "performance_baseline.json") -> Dict[str, Dict]:
        baseline_path = os.path.join(os.path.dirname(self.output_dir), baseline_file)
        if not os.path.exists(baseline_path):
            return {}
        
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        return baseline_data.get('swimlanes', {})

    def save_baseline_swimlanes(self, swimlanes: Dict[str, Dict], 
                               baseline_file: str = "performance_baseline.json"):
        baseline_path = os.path.join(os.path.dirname(self.output_dir), baseline_file)
        
        baseline_data = {
            'swimlanes': swimlanes,
            'timestamp': self.latest_dir
        }
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)

    def analyze_swimlanes(self) -> Dict[str, Dict]:
        if not self.performance_data:
            self.load_swimlane_data()
        
        swimlane_data = {}
        
        if 'traceEvents' in self.performance_data:
            for event in self.performance_data['traceEvents']:
                if event.get('ph') == 'X':
                    swimlane_id = event.get('tid', 'unknown')
                    task_name = event.get('name', 'unknown')
                    duration = event.get('dur', 0)
                    timestamp = event.get('ts', 0)
                    
                    if swimlane_id not in swimlane_data:
                        swimlane_data[swimlane_id] = {
                            'total_duration': 0,
                            'task_count': 0,
                            'tasks': [],
                            'start_time': timestamp,
                            'end_time': timestamp + duration,
                            'task_types': {}
                        }
                    
                    swimlane_data[swimlane_id]['total_duration'] += duration
                    swimlane_data[swimlane_id]['task_count'] += 1
                    swimlane_data[swimlane_id]['tasks'].append({
                        'name': task_name,
                        'duration': duration,
                        'start': timestamp,
                        'end': timestamp + duration
                    })
                    swimlane_data[swimlane_id]['start_time'] = min(swimlane_data[swimlane_id]['start_time'], timestamp)
                    swimlane_data[swimlane_id]['end_time'] = max(swimlane_data[swimlane_id]['end_time'], timestamp + duration)
                    
                    if task_name not in swimlane_data[swimlane_id]['task_types']:
                        swimlane_data[swimlane_id]['task_types'][task_name] = {
                            'count': 0,
                            'total_duration': 0
                        }
                    swimlane_data[swimlane_id]['task_types'][task_name]['count'] += 1
                    swimlane_data[swimlane_id]['task_types'][task_name]['total_duration'] += duration
        
        return swimlane_data

    def analyze_core_metrics(self) -> Dict[str, Dict]:
        if not self.performance_data:
            self.load_swimlane_data()
        
        core_metrics = {}
        
        if 'traceEvents' in self.performance_data:
            for event in self.performance_data['traceEvents']:
                if 'name' in event and 'ph' in event and event['ph'] == 'X':
                    core_name = event.get('name', 'unknown')
                    duration = event.get('dur', 0)
                    
                    if core_name not in core_metrics:
                        core_metrics[core_name] = {
                            'total_work_time': 0,
                            'total_wait_time': 0,
                            'wait_schedule_time': 0,
                            'wait_predecessor_time': 0,
                            'task_count': 0
                        }
                    
                    core_metrics[core_name]['total_work_time'] += duration
                    core_metrics[core_name]['task_count'] += 1
        
        return core_metrics

    def calculate_utilization(self, core_metrics: Dict[str, Dict]) -> Dict[str, float]:
        utilization = {}
        
        for core_name, metrics in core_metrics.items():
            total_work_time = metrics['total_work_time']
            total_wait_time = metrics['total_wait_time']
            
            if total_work_time > 0:
                aicore_time = total_work_time - total_wait_time
                utilization[core_name] = (aicore_time / total_work_time) * 100
            else:
                utilization[core_name] = 0.0
        
        return utilization

    def calculate_bubble_rate(self, core_metrics: Dict[str, Dict]) -> Dict[str, float]:
        bubble_rates = {}
        
        for core_name, metrics in core_metrics.items():
            total_work_time = metrics['total_work_time']
            wait_schedule_time = metrics['wait_schedule_time']
            
            if total_work_time > 0:
                bubble_rates[core_name] = (wait_schedule_time / total_work_time) * 100
            else:
                bubble_rates[core_name] = 0.0
        
        return bubble_rates

    def get_performance_rating(self, utilization: float, bubble_rate: float) -> Tuple[int, str]:
        if utilization > 90 and bubble_rate < 2:
            return 5, "Excellent"
        elif utilization > 80 and bubble_rate < 5:
            return 4, "Good"
        elif utilization > 60 and bubble_rate < 10:
            return 3, "Fair"
        elif utilization > 50 and bubble_rate < 20:
            return 2, "Poor"
        else:
            return 1, "Very Poor"

    def generate_performance_report(self) -> str:
        core_metrics = self.analyze_core_metrics()
        utilization = self.calculate_utilization(core_metrics)
        bubble_rates = self.calculate_bubble_rate(core_metrics)
        
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        avg_bubble_rate = sum(bubble_rates.values()) / len(bubble_rates) if bubble_rates else 0
        
        rating, rating_text = self.get_performance_rating(avg_utilization, avg_bubble_rate)
        
        report = f"""
{'='*60}
Performance Analysis Report
{'='*60}

Output Directory: {self.latest_dir}

Core Performance Metrics:
{'-'*60}
"""
        
        for core_name in core_metrics:
            report += f"{core_name}:\n"
            report += f"  Total Work Time: {core_metrics[core_name]['total_work_time']:.2f} us\n"
            report += f"  Task Count: {core_metrics[core_name]['task_count']}\n"
            report += f"  Utilization: {utilization[core_name]:.2f}%\n"
            report += f"  Bubble Rate: {bubble_rates[core_name]:.2f}%\n"
            report += "\n"
        
        report += f"""
Summary Statistics:
{'-'*60}
Average Core Utilization: {avg_utilization:.2f}%
Average Bubble Rate: {avg_bubble_rate:.2f}%
Performance Rating: {'⭐' * rating} ({rating_text})

Performance Thresholds:
{'-'*60}
Target Utilization: >90%
Target Bubble Rate: <2%
Current Utilization: {avg_utilization:.2f}% {'✓' if avg_utilization > 90 else '✗'}
Current Bubble Rate: {avg_bubble_rate:.2f}% {'✓' if avg_bubble_rate < 2 else '✗'}

Optimization Suggestions:
{'-'*60}
"""
        
        if avg_bubble_rate > 10:
            report += "- High bubble rate detected. Consider increasing stitch_function_max_num\n"
            report += "- Try loop unrolling to reduce scheduling overhead\n"
        
        if avg_utilization < 50:
            report += "- Low core utilization detected. Consider:\n"
            report += "  * Adjusting tile shapes for better load balance\n"
            report += "  * Enabling L2 affinity scheduling\n"
            report += "  * Using cube_nbuffer_setting to merge isomorphic subgraphs\n"
        
        if avg_utilization > 50 and avg_utilization < 80:
            report += "-. Moderate utilization. Consider fine-tuning:\n"
            report += "  * Adjust cube_l1_reuse_setting for L1 data reuse\n"
            report += "  * Optimize tile shapes based on swimlane analysis\n"
        
        report += f"\n{'='*60}\n"
        
        return report

    def compare_swimlanes(self, baseline_swimlanes: Dict[str, Dict], 
                         current_swimlanes: Dict[str, Dict],
                         tolerance: float = 0.1) -> Dict[str, Dict]:
        comparison_result = {}
        
        all_swimlane_ids = set(baseline_swimlanes.keys()) | set(current_swimlanes.keys())
        
        for swimlane_id in all_swimlane_ids:
            baseline = baseline_swimlanes.get(swimlane_id, {})
            current = current_swimlanes.get(swimlane_id, {})
            
            baseline_duration = baseline.get('total_duration', 0)
            current_duration = current.get('total_duration', 0)
            
            baseline_task_count = baseline.get('task_count', 0)
            current_task_count = current.get('task_count', 0)
            
            if baseline_duration > 0:
                duration_change = ((current_duration - baseline_duration) / baseline_duration) * 100
            else:
                duration_change = 0.0
            
            if baseline_task_count > 0:
                task_count_change = current_task_count - baseline_task_count
            else:
                task_count_change = 0
            
            comparison_result[swimlane_id] = {
                'baseline_duration': baseline_duration,
                'current_duration': current_duration,
                'duration_change': duration_change,
                'baseline_task_count': baseline_task_count,
                'current_task_count': current_task_count,
                'task_count_change': task_count_change,
                'status': 'regression' if duration_change > tolerance * 100 else 'improvement' if duration_change < -tolerance * 100 else 'stable',
                'task_comparison': self._compare_task_types(
                    baseline.get('task_types', {}),
                    current.get('task_types', {})
                )
            }
        
        return comparison_result

    def _compare_task_types(self, baseline_tasks: Dict, current_tasks: Dict) -> Dict:
        comparison = {}
        
        all_task_names = set(baseline_tasks.keys()) | set(current_tasks.keys())
        
        for task_name in all_task_names:
            baseline = baseline_tasks.get(task_name, {'count': 0, 'total_duration': 0})
            current = current_tasks.get(task_name, {'count': 0, 'total_duration': 0})
            
            comparison[task_name] = {
                'baseline_count': baseline['count'],
                'current_count': current['count'],
                'count_count': current['count'] - baseline['count'],
                'baseline_duration': baseline['total_duration'],
                'current_duration': current['total_duration'],
                'duration_change': current['total_duration'] - baseline['total_duration']
            }
        
        return comparison

    def generate_swimlane_comparison_report(self, baseline_swimlanes: Dict[str, Dict],
                                          current_swimlanes: Dict[str, Dict]) -> str:
        comparison = self.compare_swimlanes(baseline_swimlanes, current_swimlanes)
        
        report = f"""
{'='*80}
Swimlane Comparison Report
{'='*80}

Swimlane Performance Comparison:
{'-'*80}
"""
        
        for swimlane_id, comp in sorted(comparison.items()):
            status_icon = {
                'improvement': '✓',
                'stable': '~',
                'regression': '✗'
            }.get(comp['status'], '?')
            
            report += f"\n{status_icon} Swimlane {swimlane_id}:\n"
            report += f"  Duration: {comp['baseline_duration']:.2f} us → {comp['current_duration']:.2f} us "
            report += f"({comp['duration_change']:+.2f}%)\n"
            report += f"  Task Count: {comp['baseline_task_count']} → {comp['current_task_count']} "
            report += f"({comp['task_count_change']:+d})\n"
            
            if comp['task_comparison']:
                report += f"  Task Types:\n"
                for task_name, task_comp in sorted(comp['task_comparison'].items()):
                    if task_comp['count_count'] != 0 or abs(task_comp['duration_change']) > 0.01:
                        report += f"    - {task_name}:\n"
                        report += f"      Count: {task_comp['baseline_count']} → {task_comp['current_count']} "
                        report += f"({task_comp['count_count']:+d})\n"
                        report += f"      Duration: {task_comp['baseline_duration']:.2f} us → {task_comp['current_duration']:.2f} us "
                        report += f"({task_comp['duration_change']:+.2f} us)\n"
        
        report += f"\n{'='*80}\n"
        
        return report

    def check_performance_thresholds(self, 
                                   min_utilization: float = 50.0,
                                   max_bubble_rate: float = 20.0) -> Tuple[bool, str]:
        core_metrics = self.analyze_core_metrics()
        utilization = self.calculate_utilization(core_metrics)
        bubble_rates = self.calculate_bubble_rate(core_metrics)
        
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        avg_bubble_rate = sum(bubble_rates.values()) / len(bubble_rates) if bubble_rates else 0
        
        passed = True
        messages = []
        
        if avg_utilization < min_utilization:
            passed = False
            messages.append(f"Average utilization {avg_utilization:.2f}% below threshold {min_utilization}%")
        
        if avg_bubble_rate > max_bubble_rate:
            passed = False
            messages.append(f"Average bubble rate {avg_bubble_rate:.2f}% exceeds threshold {max_bubble_rate}%")
        
        return passed, "; ".join(messages)

    def check_time_thresholds(self, swimlanes: Dict[str, Dict], 
                             total_time_threshold: Optional[float] = None) -> Tuple[bool, str]:
        passed = True
        messages = []
        
        for swimlane_id, swimlane_data in swimlanes.items():
            total_duration = swimlane_data.get('total_duration', 0)
            
            if total_time_threshold is not None:
                if total_duration > total_time_threshold:
                    passed = False
                    messages.append(f"Swimlane {swimlane_id} total duration {total_duration:.2f} us exceeds threshold {total_time_threshold:.2f} us")
            
            for task_name, task_info in swimlane_data.get('task_types', {}).items():
                task_duration = task_info.get('total_duration', 0)
                
                threshold_key = f"{swimlane_id}.{task_name}"
                if threshold_key in self.time_thresholds:
                    threshold = self.time_thresholds[threshold_key]
                    if task_duration > threshold:
                        passed = False
                        messages.append(f"Task {task_name} in swimlane {swimlane_id} duration {task_duration:.2f} us exceeds threshold {threshold:.2f} us")
        
        return passed, "; ".join(messages)

    def generate_time_threshold_report(self, swimlanes: Dict[str, Dict],
                                       total_time_threshold: Optional[float] = None) -> str:
        report = f"""
{'='*80}
Time Threshold Check Report
{'='*80}

"""
        
        total_durations = {}
        for swimlane_id, swimlane_data in swimlanes.items():
            total_duration = swimlane_data.get('total_duration', 0)
            total_durations[swimlane_id] = total_duration
            
            report += f"\nSwimlane {swimlane_id}:\n"
            report += f"  Total Duration: {total_duration:.2f} us"
            
            if total_time_threshold is not None:
                status = "✓" if total_duration <= total_time_threshold else "✗"
                report += f" {status} (threshold: {total_time_threshold:.2f} us)\n"
            else:
                report += "\n"
            
            report += f"  Task Breakdown:\n"
            
            for task_name, task_info in swimlane_data.get('task_types', {}).items():
                task_duration = task_info.get('total_duration', 0)
                task_count = task_info.get('count', 0)
                avg_duration = task_duration / task_count if task_count > 0 else 0
                
                report += f"    - {task_name}:\n"
                report += f"      Total: {task_duration:.2f} us, Count: {task_count}, Avg: {avg_duration:.2f} us\n"
                
                threshold_key = f"{swimlane_id}.{task_name}"
                if threshold_key in self.time_thresholds:
                    threshold = self.time_thresholds[threshold_key]
                    status = "✓" if task_duration <= threshold else "✗"
                    report += f"      Threshold: {threshold:.2f} us {status}\n"
        
        if total_time_threshold is not None:
            report += f"\nOverall Total Duration Check:\n"
            report += f"{'-'*80}\n"
            
            overall_total = sum(total_durations.values())
            status = "✓" if overall_total <= total_time_threshold else "✗"
            report += f"Total Duration: {overall_total:.2f} us {status} (threshold: {total_time_threshold:.2f} us)\n"
        
        report += f"\n{'='*80}\n"
        
        return report


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


@pytest.mark.world_size(4)
def test_matmul_allreduce_add_rmsnorm_performance():
    mp.set_start_method('spawn', force=True)
    config = DistributedConfig(world_size=4)
    
    output_name = "output"
    output_dir = f"{Path.cwd()}/{output_name}"
    os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir
    
    save_baseline = os.environ.get('SAVE_PERFORMANCE_BASELINE', 'false').lower() == 'true'
    
    time_thresholds = {
        '0.matmul': 5000.0,
        '0.allreduce': 3000.0,
        '0.rmsnorm': 2000.0,
    }
    
    total_time_threshold = 10000.0
    
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
        analyzer = PerformanceAnalyzer(output_dir, time_thresholds=time_thresholds)
        
        swimlane_data = analyzer.load_swimlane_data()
        assert swimlane_data, "Failed to load swimlane data"
        
        current_swimlanes = analyzer.analyze_swimlanes()
        
        if save_baseline:
            analyzer.save_baseline_swimlanes(current_swimlanes)
            print("Performance baseline saved successfully")
        else:
            baseline_swimlanes = analyzer.load_baseline_swimlanes()
            
            if baseline_swimlanes:
                comparison_report = analyzer.generate_swimlane_comparison_report(
                    baseline_swimlanes, current_swimlanes
                )
                print(comparison_report)
            else:
                print("No baseline found. Set SAVE_PERFORMANCE_BASELINE=true to create one.")
        
        report = analyzer.generate_performance_report()
        print(report)
        
        time_threshold_report = analyzer.generate_time_threshold_report(
            current_swimlanes, total_time_threshold=total_time_threshold
        )
        print(time_threshold_report)
        
        passed, message = analyzer.check_performance_thresholds(
            min_utilization=50.0,
            max_bubble_rate=20.0
        )
        
        if not passed:
            pytest.fail(f"Performance threshold check failed: {message}")
        
        time_passed, time_message = analyzer.check_time_thresholds(
            current_swimlanes, total_time_threshold=total_time_threshold
        )
        
        if not time_passed:
            pytest.fail(f"Time threshold check failed: {time_message}")
        
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def main():
    test_matmul_allreduce_add_rmsnorm_performance()


if __name__ == '__main__':
    main()
