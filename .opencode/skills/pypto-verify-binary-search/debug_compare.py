#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import subprocess
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def read_jit_data(filename):
    """读取 jit 生成的数据文件"""
    if not os.path.exists(filename):
        logging.error(f"文件不存在: {filename}")
        return None
    data = np.fromfile(filename, dtype=np.float32)
    logging.info(f"Read {filename}: shape={data.shape}")
    return data

def compare_with_golden(jit_data, golden_data, name, rtol=1e-3, atol=1e-3):
    """对比 jit 结果与 golden 结果"""
    if jit_data is None or golden_data is None:
        logging.error(f"{name}: ✗ FAIL (文件不存在)")
        return False

    min_size = min(jit_data.shape[0], golden_data.shape[0])
    jit_data = jit_data[:min_size]
    golden_data = golden_data[:min_size]

    diff = np.max(np.abs(jit_data - golden_data))
    max_val = np.max(np.abs(golden_data))
    relative_error = diff / (max_val + 1e-10)

    match = relative_error < rtol and diff < atol

    status = "✓ PASS" if match else "✗ FAIL"
    logging.info(f"{name}: {status}")
    logging.info(f"  Max diff: {diff:.6f}")
    logging.info(f"  Max val: {max_val:.6f}")
    logging.info(f"  Relative error: {relative_error:.6f}")
    logging.info(f"  Tolerance: rtol={rtol}, atol={atol}")

    return match

def find_latest_jit_files():
    """查找最新生成的 jit 数据文件"""
    # 查找最新的 output 目录
    result = subprocess.run(['ls', '-lt', 'output/'], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("未找到 output 目录")
        return {}

    lines = result.stdout.strip().split('\n')
    if len(lines) < 2:
        logging.error("output 目录为空")
        return {}

    # 获取最新目录名
    latest_dir = lines[1].split()[-1]
    tensor_dir = f"output/{latest_dir}/tensor/kernel/"

    if not os.path.exists(tensor_dir):
        logging.error(f"tensor 目录不存在: {tensor_dir}")
        return {}

    # 查找所有 .data 文件
    result = subprocess.run(['find', tensor_dir, '-name', '*.data'], capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("未找到数据文件")
        return {}

    files = result.stdout.strip().split('\n')
    return {f: os.path.basename(f) for f in files}

def main():
    logging.info("=" * 80)
    logging.info("PyPTO 算子二分查找调试 - 中间结果对比")
    logging.info("=" * 80)

    # 读取 golden 数据
    logging.info("=" * 80)
    logging.info("步骤 1：读取 Golden 数据")
    logging.info("=" * 80)

    golden_files = {
        "rms_norm": "golden_checkpoint_rms_norm.bin",
        "qkv_matmul": "golden_checkpoint_qkv_matmul.bin",
        "q_rope": "golden_checkpoint_q_rope.bin",
        "k_rope": "golden_checkpoint_k_rope.bin"
    }

    golden_data = {}
    for key, filename in golden_files.items():
        if os.path.exists(filename):
            golden_data[key] = np.fromfile(filename, dtype=np.float32)
            logging.info(f"✓ 读取 {filename}: shape={golden_data[key].shape}")
        else:
            logging.warning(f"✗ 文件不存在: {filename}")

    # 读取 jit 数据
    logging.info("=" * 80)
    logging.info("步骤 2：读取 JIT 数据")
    logging.info("=" * 80)

    jit_files_map = find_latest_jit_files()
    jit_data = {}

    for file_path, file_name in jit_files_map.items():
        jit_data[file_name] = read_jit_data(file_path)

    # 对比数据
    logging.info("=" * 80)
    logging.info("步骤 3：对比 JIT 和 Golden 数据")
    logging.info("=" * 80)

    # 由于 jit 文件名是自动生成的，我们需要根据文件内容或位置来匹配
    # 这里简化处理：列出所有文件并让用户手动检查
    logging.info("Golden 数据文件:")
    for key, filename in golden_files.items():
        if key in golden_data:
            logging.info(f"  {key}: {filename} - shape={golden_data[key].shape}")

    logging.info("JIT 数据文件:")
    for file_name, data in jit_data.items():
        logging.info(f"  {file_name}: shape={data.shape}")

    # 尝试对比
    logging.info("=" * 80)
    logging.info("步骤 4：尝试对比关键检查点")
    logging.info("=" * 80)

    # 假设 jit 生成的文件按顺序对应 golden 的检查点
    # 这里需要根据实际情况调整
    jit_file_list = sorted(jit_data.keys())

    if len(jit_file_list) >= 4 and len(golden_data) >= 4:
        # 对应 residual_bf16 (rms_norm后)
        compare_with_golden(jit_data[jit_file_list[0]], golden_data["rms_norm"],
                           "Checkpoint 1: RMS Norm", rtol=1e-3, atol=1e-3)

        # 对应 mm_bf16 (QKV matmul后)
        compare_with_golden(jit_data[jit_file_list[1]], golden_data["qkv_matmul"],
                           "Checkpoint 2: QKV Matmul", rtol=1e-3, atol=1e-3)

        # 对应 q_res (RoPE后)
        compare_with_golden(jit_data[jit_file_list[2]], golden_data["q_rope"],
                           "Checkpoint 3: Q after RoPE", rtol=1e-3, atol=1e-3)

        # 对应 k_res (RoPE后)
        compare_with_golden(jit_data[jit_file_list[3]], golden_data["k_rope"],
                           "Checkpoint 4: K after RoPE", rtol=1e-3, atol=1e-3)
    else:
        logging.warning("数据文件数量不足，无法自动对比")
        logging.warning(f"  JIT 文件数量: {len(jit_file_list)}")
        logging.warning(f"  Golden 文件数量: {len(golden_data)}")

    logging.info("=" * 80)
    logging.info("调试建议")
    logging.info("=" * 80)
    logging.info("1. 查看每个检查点的对比结果")
    logging.info("2. 找到第一个结果不匹配的检查点")
    logging.info("3. 问题位于该检查点之前或此处")
    logging.info("4. 在该区间继续二分查找，定位具体问题 op")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()
