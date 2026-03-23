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
PyPTO 二分调试通用对比脚本
按照技能文档自动检测检查点并对比 jit 和 golden 的中间结果
"""
import os
import sys
import glob
import argparse
import logging
import re
from pathlib import Path
import numpy as np


# 初始化日志（先只输出到控制台）
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def find_latest_output_dir(work_dir="."):
    """找到最新的 output 目录"""
    output_base = os.path.join(work_dir, "output")
    if not os.path.exists(output_base):
        return None

    output_dirs = [d for d in os.listdir(output_base) if d.startswith("output_")]
    if not output_dirs:
        return None

    # 按修改时间排序，取最新的
    output_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(output_base, x)), reverse=True)
    return output_dirs[0]


def scan_checkpoints_from_dir(tensor_dir):
    """从指定目录扫描所有检查点文件"""
    if not os.path.exists(tensor_dir):
        return []

    # 查找所有 .data 文件
    data_files = glob.glob(os.path.join(tensor_dir, "*.data"))

    # 提取检查点名称（去掉后缀的数字和 .data）
    checkpoints = set()
    for data_file in data_files:
        basename = os.path.basename(data_file)
        # 格式: checkpoint_name_number.data
        # 提取 checkpoint_name
        parts = basename.rsplit('_', 1)
        if len(parts) == 2 and parts[1].replace('.data', '').isdigit():
            checkpoint_name = parts[0]
            checkpoints.add(checkpoint_name)

    # 按开头的数字排序
    def get_sort_key(name):
        # 提取开头的数字
        match = re.match(r'^(\d+)', name)
        if match:
            return int(match.group(1))
        return float('inf')  # 如果没有数字，排在最后

    return sorted(list(checkpoints), key=get_sort_key)


def get_tolerance_by_dtype(dtype, custom_rtol=None, custom_atol=None):
    """根据数据类型返回对应的容差标准
    
    Args:
        dtype: 数据类型
        custom_rtol: 自定义相对容差（用于量化场景）
        custom_atol: 自定义绝对容差（用于量化场景）
    
    Returns:
        (rtol, atol): 相对容差和绝对容差
    """
    # 如果指定了自定义容差，优先使用
    if custom_rtol is not None and custom_atol is not None:
        return custom_rtol, custom_atol
    
    # 容差标准映射表（常用数据类型）
    tolerance_map = {
        1: (1e-3, 1e-3),   # DT_INT8
        2: (1e-4, 1e-4),   # DT_INT16
        3: (1e-5, 1e-5),   # DT_INT32
        4: (1e-5, 1e-5),   # DT_INT64
        5: (1e-1, 1e-2),   # DT_FP8
        6: (1e-2, 1e-3),   # DT_FP16
        7: (1e-4, 1e-5),   # DT_FP32
        8: (5e-2, 5e-3),   # DT_BF16
    }

    return tolerance_map.get(dtype, (1e-3, 1e-3))  # 默认容差


def read_jit_data(filename):
    """读取 jit 生成的数据文件，自动检测数据类型

    返回: (data, dtype)
    """
    # 读取对应的 CSV 文件获取 dtype
    csv_file = filename.replace('.data', '.csv')

    dtype = None
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            for line in f:
                if line.startswith('dtype,'):
                    dtype = int(line.split(',')[1].strip())
                    break

    # 读取原始字节数据
    data_bytes = np.fromfile(filename, dtype=np.uint8)

    # 根据 dtype 选择正确的读取方式
    if dtype is None:
        # 如果没有 CSV 文件，尝试自动检测
        file_size = len(data_bytes)
        # 假设可能是 BF16 (2字节) 或 FP32 (4字节)
        if file_size % 2 == 0:
            # 优先尝试 BF16
            data_bf16 = np.frombuffer(data_bytes.tobytes(), dtype=np.uint16)
            data_fp32 = data_bf16.astype(np.uint32) << 16
            data = data_fp32.view(np.float32)
            logger.warning(f"  警告: 未找到 CSV 文件，假设为 BF16 格式")
            return data, 8  # 返回返回 BF16 的 dtype
        else:
            return np.fromfile(filename, dtype=np.float32), 7  # 返回 FP32 的 dtype

    # PyPTO 常用数据类型映射
    # DT_INT8 = 1, DT_INT16 = 2, DT_INT32 = 3, DT_INT64 = 4
    # DT_FP8 = 5, DT_FP16 = 6, DT_FP32 = 7, DT_BF16 = 8

    dtype_map = {
        1: ('int8', 1),
        2: ('int16', 2),
        3: ('int32', 4),
        4: ('int64', 8),
        5: ('fp8', 1),       # FP8 特殊处理
        6: ('fp16', 2),
        7: ('fp32', 4),
        8: ('bf16', 2),      # BF16
    }

    if dtype not in dtype_map:
        logger.warning(f"  警告: 未知的数据类型 {dtype}, 尝试作为 FP32 读取")
        return np.fromfile(filename, dtype=np.float32), dtype

    type_name, bytes_per_element = dtype_map[dtype]

    # 处理不同的数据类型
    if type_name == 'bf16':
        # BF16 转 FP32
        data_bf16 = np.frombuffer(data_bytes.tobytes(), dtype=np.uint16)
        data_fp32 = data_bf16.astype(np.uint32) << 16
        data = data_fp32.view(np.float32)
        return data, dtype
    elif type_name == 'fp16':
        # FP16 转 FP32
        data_fp16 = np.frombuffer(data_bytes.tobytes(), dtype=np.float16)
        return data_fp16.astype(np.float32), dtype
    elif type_name == 'fp32':
        return np.fromfile(filename, dtype=np.float32), dtype
    elif type_name == 'int32':
        data_int32 = np.fromfile(filename, dtype=np.int32)
        return data_int32.astype(np.float32), dtype
    elif type_name == 'int64':
        data_int64 = np.fromfile(filename, dtype=np.int64)
        return data_int64.astype(np.float32), dtype
    elif type_name in ['int8', 'int16']:
        # 整数类型转浮点
        dtype_numpy = {
            'int8': np.int8,
            'int16': np.int16,
        }
        data_int = np.fromfile(filename, dtype=dtype_numpy[type_name])
        return data_int.astype(np.float32), dtype
    else:
        # 其他特殊类型 (fp8 等)
        logger.warning(f"  警告: 数据类型 {type_name} (dtype={dtype}) 暂不支持，尝试作为 FP32 读取")
        return np.fromfile(filename, dtype=np.float32), dtype


def compare_with_golden(jit_data, golden_data, name, dtype=None, verbose=True, custom_rtol=None, custom_atol=None):
    """对比 jit 结果与 golden 结果
    
    Args:
        jit_data: jit 数据
        golden_data: golden 数据
        name: 检查点名称
        dtype: 数据类型
        verbose: (bool, optional): 是否显示详细对比
        custom_rtol: 自定义相对容差（用于量化场景）
        custom_atol: 自定义绝对容差（用于量化场景）
    
    Returns:
        bool: 是否匹配
    """
    min_size = min(jit_data.shape[0], golden_data.shape[0])
    jit_data_to_compare = jit_data[:min_size]
    golden_data_to_compare = golden_data[:min_size]

    # 根据 dtype 获取对应的容差
    if dtype is not None:
        rtol, atol = get_tolerance_by_dtype(dtype, custom_rtol, custom_atol)
    else:
        # 默认容差（向后兼容）
        rtol, atol = 1e-3, 1e-3

    if verbose:
        logger.info(f"  对比范围: 前 {min_size} 个元素 (jit={jit_data.shape[0]}, golden={golden_data.shape[0]})")

    # 使用 np.isclose 统计不匹配个数
    close_mask = np.isclose(jit_data_to_compare, golden_data_to_compare, rtol=rtol, atol=atol)
    mismatch_count = (~close_mask).sum()
    total_count = min_size

    # 计算统计信息（用于显示）
    diff = np.abs(jit_data_to_compare - golden_data_to_compare)
    max_diff = np.max(diff)
    max_val = np.max(np.abs(golden_data_to_compare))
    relative_error = max_diff / (max_val + 1e-10)

    # 计算实际得到的rtol和atol
    actual_rtol = relative_error
    actual_atol = max_diff

    # 判断条件：不匹配个数 < 总数 * max(rtol, atol)
    threshold = total_count * max(rtol, atol)
    match = mismatch_count < threshold

    status = "✓ PASS" if match else "✗ FAIL"
    logger.info(f"\n{name}: {status}")
    logger.info(f"  Max diff: {max_diff:.6f}")
    logger.info(f"  Max val: {max_val:.6f}")
    logger.info(f"  Relative error: {relative_error:.6f}")
    logger.info(f"  Mismatch count: {mismatch_count}/{total_count} ({mismatch_count/total_count*100:.2f}%)")
    logger.info(f"  Tolerance: rtol={rtol}, atol={atol} (dtype={dtype})")
    logger.info(f"  Actual: rtol={actual_rtol:.6f}, atol={actual_atol:.6f}")

    if verbose and not match:
        logger.info(f"  前10个元素对比:")
        for i in range(min(10, min_size)):
            jit_val = jit_data_to_compare[i]
            golden_val = golden_data_to_compare[i]
            diff_val = abs(jit_val - golden_val)
            logger.info(f"    [{i}] jit={jit_val:.6f}, golden={golden_val:.6f}, diff={diff_val:.6f}")

    return match


def analyze_results(results):
    """分析对比结果，给出二分建议"""
    logger.info("\n" + "=" * 80)
    logger.info("步骤 5：根据结果继续二分")
    logger.info("=" * 80)

    first_fail_idx = -1
    for idx, (name, match) in enumerate(results):
        if not match:
            first_fail_idx = idx
            break

    if first_fail_idx == -1:
        logger.info("✓ 所有检查点都匹配")
        logger.info("→ 问题可能在：检查点之后的操作")
        return

    fail_name = results[first_fail_idx][0]

    if first_fail_idx == 0:
        logger.info(f"✗ 第一个检查点 ({fail_name}) 就不匹配")
        logger.info(f"→ 问题可能在：输入数据或第一个计算步骤")
        logger.info(f"→ 建议：检查输入数据是否正确，或在更早的位置插入检查点")
    else:
        prev_name = results[first_fail_idx - 1][0]
        logger.info(f"✗ 检查点 {prev_name} 匹配，但 {fail_name} 不匹配")
        logger.info(f"→ 问题位置：{prev_name} 和 {fail_name} 之间的操作")
        logger.info(f"→ 建议：在这两个检查点之间插入新的检查点，进一步定位问题")


def main():
    parser = argparse.ArgumentParser(description='PyPTO 二分调试对比工具')
    parser.add_argument('--work-dir', '-w', default='.',
                        help='工作目录（默认为当前目录）')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='指定 output 目录名（不指定则自动检测最新的）')
    parser.add_argument('--golden-dir', '-g', default=None,
                        help='指定 golden 文件所在目录（默认从工作目录查找）')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细的元素级对比')
    parser.add_argument('--list', '-l', action='store_true',
                        help='只列出检查点，不进行对比')
    parser.add_argument('--rtol', type=float, default=None,
                        help='自定义相对容差（用于量化场景，推荐值：0.0078125）')
    parser.add_argument('--atol', type=float, default=None,
                        help='自定义绝对容差（用于量化场景，推荐值：0.0001）')

    args = parser.parse_args()

    # 配置日志输出到文件（保存到工作目录下）
    log_formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # 日志文件保存到工作目录
    log_file = os.path.join(os.path.abspath(args.work_dir), 'verify_result.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)

    # 重新配置日志处理器
    logger.handlers = [console_handler, file_handler]

    logger.info("=" * 80)
    logger.info("PyPTO 二分调试对比工具")
    logger.info("=" * 80)

    if args.output_dir:
        latest_dir = args.output_dir
    else:
        latest_dir = find_latest_output_dir(args.work_dir)

    if not latest_dir:
        logger.error("✗ 未找到 output 目录")
        sys.exit(1)

    latest_dir_full = latest_dir

    # 确定 golden 文件所在目录
    if args.golden_dir:
        golden_base_dir = args.golden_dir
    else:
        golden_base_dir = args.work_dir

    # 扫描指定目录
    tensor_dir = os.path.join(args.work_dir, "output", latest_dir_full, "tensor")
    checkpoints = scan_checkpoints_from_dir(tensor_dir)

    if not checkpoints:
        logger.error(f"✗ 未在 {tensor_dir} 找到检查点文件")
        sys.exit(1)

    logger.info(f"✓ 找到 output 目录: {latest_dir_full}")
    logger.info(f"✓ 找到 {len(checkpoints)} 个检查点: {checkpoints}")

    if args.list:
        logger.info("\n检查点列表:")
        for idx, ckpt in enumerate(checkpoints, 1):
            logger.info(f"  {idx}. {ckpt}")
        sys.exit(0)

    logger.info("\n" + "=" * 80)
    logger.info("步骤 4：对比 jit 和 golden 数据")
    logger.info("=" * 80)

    results = []

    for checkpoint_name in checkpoints:
        # 查找 jit 文件
        jit_pattern = os.path.join(tensor_dir, f"{checkpoint_name}_*.data")
        jit_files = sorted(glob.glob(jit_pattern))

        if not jit_files:
            logger.warning(f"\n{checkpoint_name}: ✗ 未找到 jit 文件")
            results.append((checkpoint_name, False))
            continue

        golden_pattern = os.path.join(golden_base_dir, f"golden_{checkpoint_name}.bin")
        golden_files = glob.glob(golden_pattern)

        if not golden_files:
            logger.warning(f"\n{checkpoint_name}: ✗ 未找到 golden 文件 ({golden_pattern})")
            results.append((checkpoint_name, False))
            continue

        jit_file = jit_files[0]
        golden_file = golden_files[0]

        logger.info(f"\n使用文件:")
        logger.info(f"  jit: {os.path.basename(jit_file)}")
        logger.info(f"  golden: {os.path.basename(golden_file)}")

        # 读取数据
        jit_data, dtype = read_jit_data(jit_file)
        golden_data = np.fromfile(golden_file, dtype=np.float32)

        # 对比
        match = compare_with_golden(jit_data, golden_data, checkpoint_name,
                                   dtype=dtype, verbose=args.verbose,
                                   custom_rtol=args.rtol, custom_atol=args.atol)
        results.append((checkpoint_name, match))

    # 分析结果
    analyze_results(results)

    # 返回码
    all_match = all(match for _, match in results)
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
