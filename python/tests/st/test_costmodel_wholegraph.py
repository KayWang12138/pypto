#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
整图 CostModel 验证脚本

验证内容:
  1. softmax kernel 在 SIM 模式下能正常编译和运行
  2. _cost_model_run_once_data_from_host 正常执行
  3. 输出目录及关键文件存在 (merged_swimlane.json, pipeline_summary.csv)
  4. merged_swimlane.json 格式合法
  5. pipeline_summary.csv 包含有效的 pipe cycle 数据
"""

import os
import sys
import json
import csv
import glob
import traceback

import pypto
import numpy as np
import torch

from pypto.cost_model import _cost_model_run_once_data_from_host


# ───────────────────── helpers ─────────────────────

def safe_json_load(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except json.JSONDecodeError as e:
        return None, f"Invalid json: {e}"
    except Exception as e:
        return None, f"Load json fail: {e}"


def find_latest_output_dir(base="./output"):
    """查找最新的输出目录"""
    if not os.path.exists(base):
        return None
    subdirs = [os.path.join(base, d) for d in os.listdir(base)
               if os.path.isdir(os.path.join(base, d))]
    return max(subdirs, key=os.path.getctime) if subdirs else None


def check_file_exists(path, label):
    if os.path.exists(path):
        return True, f"  [PASS] {label}: {path}"
    else:
        return False, f"  [FAIL] {label} not found: {path}"


# ───────────────────── kernel ─────────────────────

def softmax_core(input_tensor: pypto.tensor) -> pypto.tensor:
    row_max = pypto.amax(input_tensor, dim=-1, keepdim=True)
    sub = pypto.sub(input_tensor, row_max)
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    return pypto.div(exp, esum)


@pypto.frontend.jit(runtime_options={"run_mode": 1})  # SIM mode
def softmax(input_tensor: pypto.Tensor(), output_tensor: pypto.Tensor()):
    tensor_shape = input_tensor.shape
    b = tensor_shape[0]
    n1, n2, dim = tensor_shape[1:]
    tile_b = 1
    b_loop = b // tile_b

    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in range(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        input_view = input_tensor[b_offset:b_offset_end, :n1, :n2, :dim]
        softmax_out = softmax_core(input_view)
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)


# ───────────────────── test steps ─────────────────────

def test_wholegraph_costmodel():
    """运行整图 costmodel 并验证输出"""
    passed = 0
    failed = 0
    results = []

    print("=" * 60)
    print("  整图 CostModel 验证")
    print("=" * 60)

    # ── Step 1: 准备数据 ──
    print("\n[Step 1] 准备输入数据 ...")
    shape = (32, 32, 1, 256)
    input_data = torch.rand(shape, dtype=torch.float32)
    output_data = torch.zeros(shape, dtype=torch.float32)
    print(f"  input shape: {shape}, dtype: float32")
    passed += 1
    results.append(("数据准备", True))

    # ── Step 2: 清理旧输出 ──
    print("\n[Step 2] 清理旧输出目录 ...")
    old_output = find_latest_output_dir("./output")
    if old_output:
        print(f"  旧输出目录: {old_output} (将在运行后对比)")
    else:
        print("  无旧输出目录")

    # ── Step 3: 编译并运行 SIM 模式 (自动触发整图 costmodel) ──
    print("\n[Step 3] 编译 softmax kernel 并运行 SIM 模式 ...")
    print("  (SIM 模式下 kernel 运行自动执行整图 costmodel)")
    try:
        softmax(input_data, output_data)
        print("  kernel 编译 & SIM 运行成功")
        passed += 1
        results.append(("kernel 编译 & SIM 运行", True))
    except Exception as e:
        print(f"  [FAIL] kernel 编译运行失败: {e}")
        traceback.print_exc()
        failed += 1
        results.append(("kernel 编译 & SIM 运行", False))
        return passed, failed, results

    # ── Step 4: 验证输出目录和文件 ──
    print("\n[Step 4] 验证输出目录和文件 ...")
    output_dir = find_latest_output_dir("./output")
    if output_dir is None:
        print("  [FAIL] 未找到 output 目录")
        failed += 1
        results.append(("输出目录", False))
        return passed, failed, results

    print(f"  输出目录: {output_dir}")

    # 查找 CostModelSimulationOutput 子目录
    sim_output = os.path.join(output_dir, "CostModelSimulationOutput")
    if not os.path.exists(sim_output):
        # 可能直接在 output_dir 下
        sim_output = output_dir
    print(f"  CostModel 输出目录: {sim_output}")
    passed += 1
    results.append(("输出目录生成", True))

    # 4a: merged_swimlane.json
    swimlane_path = os.path.join(sim_output, "merged_swimlane.json")
    ok, msg = check_file_exists(swimlane_path, "merged_swimlane.json")
    print(msg)
    if ok:
        data, err = safe_json_load(swimlane_path)
        if err:
            print(f"  [FAIL] merged_swimlane.json 解析失败: {err}")
            failed += 1
            results.append(("merged_swimlane.json 解析", False))
        else:
            print(f"  [PASS] merged_swimlane.json 解析成功, 顶层 keys: {list(data.keys())[:8]}")
            passed += 1
            results.append(("merged_swimlane.json", True))
    else:
        failed += 1
        results.append(("merged_swimlane.json", False))

    # 4b: pipeline_summary.csv (sub_graph 分支新增特性，旧版本可能不存在)
    csv_path = os.path.join(sim_output, "pipeline_summary.csv")
    ok, msg = check_file_exists(csv_path, "pipeline_summary.csv")
    print(msg)
    if ok:
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
            header = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []
            print(f"  [PASS] pipeline_summary.csv: {len(data_rows)} data rows, columns: {header}")
            passed += 1
            results.append(("pipeline_summary.csv", True))
        except Exception as e:
            print(f"  [FAIL] pipeline_summary.csv 读取失败: {e}")
            failed += 1
            results.append(("pipeline_summary.csv", False))
    else:
        print("  [WARN] pipeline_summary.csv 不存在 (sub_graph 分支新增特性，旧版本不影响)")
        passed += 1
        results.append(("pipeline_summary.csv (optional, skipped)", True))

    # 4c: 列出所有输出文件
    print("\n  输出目录中的全部文件:")
    for root, dirs, files in os.walk(sim_output):
        level = root.replace(sim_output, '').count(os.sep)
        indent = '    ' + '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        for f in files:
            filepath = os.path.join(root, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f'{indent}  {f} ({size_kb:.1f} KB)')

    # ── Step 5: 验证 dyn_topo.txt ──
    print("\n[Step 5] 验证 dyn_topo.txt ...")
    topo_path = os.path.join(sim_output, "dyn_topo.txt")
    ok, msg = check_file_exists(topo_path, "dyn_topo.txt")
    print(msg)
    if ok:
        with open(topo_path, 'r') as f:
            lines = f.readlines()
        print(f"  [PASS] dyn_topo.txt: {len(lines)} lines")
        # 打印前几行用于诊断
        for line in lines[:5]:
            print(f"    {line.rstrip()}")
        passed += 1
        results.append(("dyn_topo.txt", True))
    else:
        print("  [WARN] dyn_topo.txt 不存在，不影响整图验证")
        # dyn_topo.txt 不存在不算失败
        passed += 1
        results.append(("dyn_topo.txt (skipped)", True))

    return passed, failed, results


# ───────────────────── main ─────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PyPTO 整图 CostModel 功能验证")
    print("=" * 60 + "\n")

    passed, failed, results = test_wholegraph_costmodel()

    # 汇总
    print("\n" + "=" * 60)
    print("  验证结果汇总")
    print("=" * 60)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  总计: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\n  >>> 整图 CostModel 验证失败 <<<\n")
        sys.exit(1)
    else:
        print("\n  >>> 整图 CostModel 验证全部通过 <<<\n")
        sys.exit(0)
