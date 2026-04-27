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
GLM-4.5 MatMul AllReduce Add RmsNorm Performance Test Module

This module implements performance benchmarking for fused matmul, all-reduce, add, and RMSNorm operation.
It measures execution time across multiple runs and provides statistical analysis.

Main Functions:
    - perf_matmul_allreduce_add_rmsnorm_kernel: Kernel function for performance testing
    - test_matmul_allreduce_perf: Performance benchmark test
"""

import logging
import multiprocessing as mp
import os
import csv
from pathlib import Path
import time
import statistics

import numpy as np
import pytest
import torch

import pypto

from distributed_config import DistributedConfig
from swimlane_analyzer import SwimlaneAnalyzer

logger = logging.getLogger(__name__)


def _get_soc_version():
    """获取 soc version"""
    try:
        import torch_npu
        return torch_npu.npu.get_soc_version()
    except Exception:
        return None


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    runtime_options={"stitch_function_max_num": 128},
)
def perf_matmul_allreduce_add_rmsnorm_kernel(
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

    for perf_bs_idx in pypto.loop(bs_loop, name="LOOP_MM_AR_RMSNORM_PERF", idx_name="perf_bs_idx"):
        # 1. create shmem tesnor
        shmem_shape = [view_row_shape, hidden_size]
        shmem_tensor = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_FP32, shmem_shape)
        shmem_barrier_signal = pypto.distributed.create_shmem_signal(group_name, world_size)
        my_pe = pypto.distributed.my_symbolic_pe(group_name)
        for perf_inner_idx in pypto.loop(1, name="LOOP_MM_AR_RMS_PERF_L0", idx_name="perf_inner_idx"):
            in_tensor_tile = pypto.view(
                in_tensor, (view_row_shape, in_tensor.shape[1]), [perf_bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - perf_bs_idx * view_row_shape).min(view_row_shape), in_tensor.shape[1]])

            # 2. clear data
            pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
            data_clear_out = pypto.distributed.shmem_clear_data(
                shmem_tensor, shmem_shape, [0, 0], pred=[in_tensor_tile])
            signal_clear_out = pypto.distributed.shmem_clear_signal(
                shmem_tensor, pred=[in_tensor_tile])
            barrier_out = pypto.distributed.shmem_barrier_all(
                shmem_barrier_signal, [data_clear_out, signal_clear_out])

            # 3. matmul
            pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512])
            matmul_result = pypto.matmul(in_tensor_tile, matmul_weight, pypto.DT_FP32, b_trans=True)

            # 4. allreduce
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
                valid_shape=[(batch_size - perf_bs_idx * view_row_shape).min(view_row_shape), hidden_size]
            )

            # 5. Add RmsNorm
            residual_tile = pypto.view(
                residual, (view_row_shape, hidden_size), [perf_bs_idx * view_row_shape, 0],
                valid_shape=[(batch_size - perf_bs_idx * view_row_shape).min(view_row_shape), hidden_size])

            # add
            residual_tile_fp32 = pypto.cast(residual_tile, pypto.DT_FP32)
            add_out = pypto.add(all_reduce_out, residual_tile_fp32)

            # rms norm
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

            residual_out[perf_bs_idx * pypto.symbolic_scalar(view_row_shape):] = residual_bf16_tmp
            out_tensor[perf_bs_idx * pypto.symbolic_scalar(view_row_shape):] = hidden_bf16


def perf_generate_golden_data(world_size: int):
    # 设置参数
    perf_batch_size = 8
    perf_attn_dim = 1536
    perf_hidden_dim = 5120
    torch.manual_seed(42)

    #构造每张卡上需要的数据
    perf_input_datas = []
    for _ in range(world_size):
        perf_in_tensor = torch.randn((perf_batch_size, perf_attn_dim), dtype=torch.bfloat16).share_memory_()
        perf_matmul_w = torch.randn((perf_hidden_dim, perf_attn_dim), dtype=torch.bfloat16).share_memory_()
        perf_residual_tensor = torch.randn((perf_batch_size, perf_hidden_dim), dtype=torch.bfloat16).share_memory_()
        perf_gamma_tensor = torch.randn((perf_hidden_dim), dtype=torch.bfloat16).share_memory_()
        perf_bias_tensor = torch.randn((perf_hidden_dim), dtype=torch.bfloat16).share_memory_()
        perf_eps_val = 1e-5
        perf_input_item = [perf_in_tensor, perf_matmul_w, perf_residual_tensor, perf_gamma_tensor, perf_bias_tensor, perf_eps_val]
        perf_input_datas.append(perf_input_item)
    perf_output_datas = perf_compute_golden_result(perf_batch_size, perf_hidden_dim, perf_input_datas)
    return perf_input_datas, perf_output_datas


def perf_compute_golden_result(perf_bs, perf_dim, perf_input_datas):
    perf_output_datas = []
    # 计算 matmul & allreduce 结果， 该结果所有卡上一致
    perf_mm_ar_result_fp32 = torch.zeros((perf_bs, perf_dim), dtype=torch.float32)
    for perf_input_item in perf_input_datas:
        perf_in_tensor, perf_matmul_w = perf_input_item[:2]
        perf_mm_result = torch.matmul(perf_in_tensor.to(torch.float32), perf_matmul_w.to(torch.float32).T)
        perf_mm_ar_result_fp32 += perf_mm_result

    # 计算各卡上add_rmsnorm之后的结果
    for perf_input_item in perf_input_datas:
        perf_residual_tensor, perf_gamma_tensor, perf_bias_tensor, perf_eps_val = perf_input_item[-4:]
        perf_add_result = perf_residual_tensor.to(torch.float32) + perf_mm_ar_result_fp32
        perf_mean_coef = 1.0 / perf_add_result.shape[-1]
        perf_f32_tensor = perf_add_result
        perf_square_val = perf_f32_tensor * perf_f32_tensor
        perf_square_val = perf_square_val.sum(dim=-1, keepdim=True)
        perf_mean_val = perf_square_val * perf_mean_coef
        perf_sum_val = perf_mean_val + perf_eps_val
        perf_sqrt_val = torch.sqrt(perf_sum_val)
        perf_div_val = perf_f32_tensor / perf_sqrt_val
        perf_res_val = perf_div_val * perf_gamma_tensor.to(torch.float32)
        perf_res_val = perf_res_val + perf_bias_tensor.to(perf_res_val.dtype)
        perf_output_item = [perf_res_val.to(torch.bfloat16), perf_f32_tensor.to(torch.bfloat16)]
        perf_output_datas.append(perf_output_item)
    return perf_output_datas


def perf_matmul_ar_rmsnorm_worker(
    config: DistributedConfig,
    perf_input_item: list,
    perf_output_item: list,
    perf_rank_id: int,
):
    perf_hccl_groups = config.init_hccl_comm(perf_rank_id)
    perf_device_id = config.get_physical_device_id(perf_rank_id)
    perf_device_str = f'npu:{perf_device_id}'

    perf_in_tensor, perf_matmul_w, perf_residual_tensor, perf_gamma_tensor, perf_bias_tensor, perf_eps_val = perf_input_item
    perf_golden_out, perf_golden_res = perf_output_item

    perf_out_tensor = torch.empty(perf_residual_tensor.shape, dtype=torch.bfloat16, device=perf_device_str)
    perf_res_out_tensor = torch.empty(perf_residual_tensor.shape, dtype=torch.bfloat16, device=perf_device_str)

    perf_kernel_inputs = [perf_in_tensor.to(perf_device_str), perf_matmul_w.to(perf_device_str),
        perf_residual_tensor.to(perf_device_str), perf_gamma_tensor.to(perf_device_str),
        perf_bias_tensor.to(perf_device_str), perf_out_tensor, perf_res_out_tensor]

    perf_matmul_allreduce_add_rmsnorm_kernel(*perf_kernel_inputs, perf_eps_val, perf_hccl_groups[0], config.world_size)

    np.testing.assert_allclose(
        np.array(perf_out_tensor.cpu().flatten().tolist()),
        np.array(perf_golden_out.cpu().flatten().tolist()),
        rtol=8e-3,
        atol=8e-3,
    )

    np.testing.assert_allclose(
        np.array(perf_res_out_tensor.cpu().flatten().tolist()),
        np.array(perf_golden_res.cpu().flatten().tolist()),
        rtol=8e-3,
        atol=8e-3,
    )


@pytest.mark.skip(reason="performance test case")
@pytest.mark.world_size(4)
def test_matmul_allreduce_perf_benchmark():
    logger.info("=" * 60)
    logger.info("开始运行matmul_allreduce_add_rmsnorm性能基准测试")
    logger.info("=" * 60)

    mp.set_start_method('spawn', force=True)
    soc_version = _get_soc_version()
    world_size = 2 if soc_version == 260 else 8
    config = DistributedConfig(world_size=world_size)
    logger.info(f"检测到 soc_version={soc_version}, 使用 world_size={world_size}")

    expected_total_time = 80 if soc_version == 260 else 60
    all_min_times = []

    for perf_run_num in range(1, 11):
        perf_min_time = _perf_run_single_iteration(perf_run_num, config, expected_total_time)
        if perf_min_time is not None:
            all_min_times.append(perf_min_time)
        if perf_run_num < 10:
            time.sleep(1)

    if all_min_times:
        perf_std_val = statistics.stdev(all_min_times) if len(all_min_times) > 1 else 0.0
        _perf_print_statistics(all_min_times, expected_total_time, perf_std_val)
        _perf_save_stats_csv(config, all_min_times, expected_total_time, perf_std_val)


def _perf_run_single_iteration(perf_run_num, config, perf_expected_time):
    logger.info(f"第 {perf_run_num}/10 次运行开始")

    perf_timestamp_dir = time.strftime('%Y-%m-%d_%H-%M-%S')
    perf_output_name = f"output_run_{perf_run_num}"
    perf_output_dir = f"{Path.cwd()}/output/{perf_timestamp_dir}/{perf_output_name}"
    os.environ["TILE_FWK_OUTPUT_DIR"] = perf_output_dir
    os.makedirs(perf_output_dir, exist_ok=True)

    perf_processes = []
    perf_input_datas, perf_output_datas = perf_generate_golden_data(config.world_size)

    for perf_idx in range(config.world_size):
        perf_process = mp.Process(target=perf_matmul_ar_rmsnorm_worker,
                      args=(config, perf_input_datas[perf_idx], perf_output_datas[perf_idx], perf_idx))
        perf_process.start()
        perf_processes.append(perf_process)

    for perf_join_idx, perf_join_proc in enumerate(perf_processes):
        perf_join_proc.join()
        if perf_join_proc.exitcode != 0:
            raise AssertionError(f"process {perf_join_idx} failed, return: {perf_join_proc.exitcode}")

    try:
        perf_analyzer = SwimlaneAnalyzer(perf_output_dir, expected_total_time=perf_expected_time)
        perf_stats = perf_analyzer.calculate_stats(config.world_size)
        perf_min_time = perf_stats['min_time']
        perf_is_pass = perf_analyzer.check_within_expected(config.world_size)

        if not perf_is_pass:
            logger.warning(f"第{perf_run_num}次运行执行时间超出预期: "
                f"实际{perf_min_time:.3f}us > 预期{perf_expected_time:.3f}us")
        else:
            logger.info(f"第{perf_run_num}次运行执行时间在预期范围内: "
                f"实际{perf_min_time:.3f}us <= 预期{perf_expected_time:.3f}us")

        return perf_min_time

    except Exception as perf_exc:
        logger.error(f"第{perf_run_num}次运行分析失败: {perf_exc}")
        return None

    finally:
        logger.info(f"第{perf_run_num}次运行性能数据保存在: {perf_output_dir}")


def _perf_print_statistics(perf_time_list, perf_expected_time, perf_std_val):
    logger.info("=" * 60)
    logger.info("性能基准测试统计结果")
    logger.info("=" * 60)

    perf_avg_time = statistics.mean(perf_time_list) if len(perf_time_list) > 1 else perf_time_list[0]
    perf_min_time = min(perf_time_list)
    perf_max_time = max(perf_time_list)

    logger.info(f"运行次数: {len(perf_time_list)}")
    logger.info(f"平均值: {perf_avg_time:.3f} us")
    logger.info(f"最小值: {perf_min_time:.3f} us")
    logger.info(f"最大值: {perf_max_time:.3f} us")

    if len(perf_time_list) > 1:
        logger.info(f"标准差: {perf_std_val:.3f} us")
        logger.info(f"波动范围: {perf_max_time - perf_min_time:.3f} us")

    if perf_expected_time is not None:
        logger.info(f"预期总时间: {perf_expected_time:.3f} us")
        perf_all_pass = all(perf_time <= perf_expected_time for perf_time in perf_time_list)
        if perf_all_pass:
            logger.info("[PASS] 所有运行的最小值都在预期时间内")
        else:
            perf_failed_runs = [perf_idx + 1 for perf_idx, perf_time in enumerate(perf_time_list)
                          if perf_time > perf_expected_time]
            logger.warning(f"[FAIL] 有{len(perf_failed_runs)}次运行的最小值超出预期: "
                f"第{', '.join(map(str, perf_failed_runs))}次")

    logger.info("=" * 60)


def _perf_save_stats_csv(config, perf_time_list, perf_expected_time, perf_std_val):
    perf_csv_file = "perf_benchmark_matmul_ar_rmsnorm.csv"
    perf_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    perf_avg_time = statistics.mean(perf_time_list) if len(perf_time_list) > 1 else perf_time_list[0]
    perf_min_time = min(perf_time_list)
    perf_max_time = max(perf_time_list)

    perf_stats_record = {
        'timestamp': perf_timestamp,
        'world_size': config.world_size,
        'expected_time_us': perf_expected_time,
        'min_of_mins_us': round(perf_min_time, 3),
        'avg_of_mins_us': round(perf_avg_time, 3),
        'max_of_mins_us': round(perf_max_time, 3),
        'std_dev_us': round(perf_std_val, 3) if len(perf_time_list) > 1 else 0.0,
        'num_runs': len(perf_time_list)
    }

    with open(perf_csv_file, 'a', newline='', encoding='utf-8') as perf_csv_f:
        perf_csv_writer = csv.DictWriter(perf_csv_f, fieldnames=[
            'timestamp', 'world_size', 'expected_time_us',
            'min_of_mins_us', 'avg_of_mins_us', 'max_of_mins_us',
            'std_dev_us', 'num_runs'
        ])
        if perf_csv_f.tell() == 0:
            perf_csv_writer.writeheader()
        perf_csv_writer.writerow(perf_stats_record)

    logger.info(f"汇总统计已追加保存到: {perf_csv_file}")


def perf_summarize_benchmark_results(
    perf_csv_file: str = "perf_benchmark_matmul_ar_rmsnorm.csv"
):
    """汇总并显示性能基准统计结果"""
    if not os.path.exists(perf_csv_file):
        logger.error(f"统计CSV文件不存在: {perf_csv_file}")
        return

    with open(perf_csv_file, 'r', encoding='utf-8') as perf_csv_f:
        perf_csv_reader = csv.DictReader(perf_csv_f)
        perf_rows = list(perf_csv_reader)

    if not perf_rows:
        logger.error("统计CSV文件为空")
        return

    perf_recent_rows = perf_rows[-10:] if len(perf_rows) > 10 else perf_rows
    _perf_print_summary_table(perf_recent_rows)
    _perf_print_overall_stats(perf_recent_rows)


def _perf_print_summary_table(perf_rows_list):
    logger.info("=" * 120)
    logger.info(f"性能基准统计结果汇总 (最近{len(perf_rows_list)}次)")
    logger.info("=" * 120)

    perf_header_fmt = "{:<5} {:<20} {:<10} {:<12} {:<12} {:<12} {:<12} {:<12} {:<8}"
    logger.info(perf_header_fmt.format('序号', '时间', 'world_size', '预期时间(us)',
                               '最小值(us)', '平均值(us)', '最大值(us)',
                               '标准差(us)', '运行次数'))
    logger.info("-" * 120)

    for perf_row_idx, perf_row in enumerate(perf_rows_list, 1):
        perf_ts = perf_row.get('timestamp', 'N/A')
        perf_ws = perf_row.get('world_size', 'N/A')
        perf_exp_time = perf_row.get('expected_time_us', 'N/A')
        perf_min_val = perf_row.get('min_of_mins_us', 'N/A')
        perf_avg_val = perf_row.get('avg_of_mins_us', 'N/A')
        perf_max_val = perf_row.get('max_of_mins_us', 'N/A')
        perf_std_val = perf_row.get('std_dev_us', 'N/A')
        perf_runs = perf_row.get('num_runs', 'N/A')

        logger.info(perf_header_fmt.format(
            perf_row_idx, perf_ts, perf_ws, perf_exp_time,
            perf_min_val, perf_avg_val, perf_max_val, perf_std_val, perf_runs
        ))

    logger.info("-" * 120)


def _perf_print_overall_stats(perf_rows_list):
    if len(perf_rows_list) <= 1:
        logger.info("=" * 120)
        return

    perf_all_min_vals = []
    perf_all_avg_vals = []
    perf_all_max_vals = []

    for perf_row in perf_rows_list:
        perf_min_str = perf_row.get('min_of_mins_us', '')
        perf_avg_str = perf_row.get('avg_of_mins_us', '')
        perf_max_str = perf_row.get('max_of_mins_us', '')

        if perf_min_str:
            perf_all_min_vals.append(float(perf_min_str))
        if perf_avg_str:
            perf_all_avg_vals.append(float(perf_avg_str))
        if perf_max_str:
            perf_all_max_vals.append(float(perf_max_str))

    if perf_all_min_vals and perf_all_avg_vals and perf_all_max_vals:
        logger.info(f"\n总体统计 (基于{len(perf_rows_list)}次统计记录):")
        _perf_print_stat_category("最小值统计", perf_all_min_vals)
        _perf_print_stat_category("平均值统计", perf_all_avg_vals)
        _perf_print_stat_category("最大值统计", perf_all_max_vals)

    logger.info("=" * 120)


def _perf_print_stat_category(perf_category, perf_vals):
    perf_category_min = min(perf_vals)
    perf_category_avg = statistics.mean(perf_vals) if len(perf_vals) > 1 else perf_vals[0]
    perf_category_max = max(perf_vals)
    logger.info(f"  {perf_category}:")
    logger.info(f"    平均: {perf_category_avg:.3f} us, 最小: {perf_category_min:.3f} us, 最大: {perf_category_max:.3f} us")


def main():
    test_matmul_allreduce_perf_benchmark()

    logger.info("\n")
    perf_summarize_benchmark_results()


if __name__ == '__main__':
    main()