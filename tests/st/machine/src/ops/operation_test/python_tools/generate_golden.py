#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import json
import argparse
import re
from pathlib import Path
import numbers
from typing import Dict, List, Any, Optional

import numpy as np
from bfloat16 import bfloat16

dtype_map = {
        'FP16': np.float16,
        'FP32': np.float32,
        'BF16': bfloat16,
        'INT8': np.int8,
        'INT16': np.int16,
        'INT32': np.int32,
        'INT64': np.int64,
        'UINT8': np.uint8,
        'UINT16': np.uint16,
        'UINT32': np.uint32,
        'UINT64': np.uint64,
    }


def add_operation(inputs: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
    """golden 实例写法"""
    return [inputs[0] + inputs[1]]


def adds_operation(inputs: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
    """内置golden adds"""
    scalar = params.get('scalar', 0)
    return [inputs[0] + scalar]


def subs_operation(inputs: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
    """内置golden subs"""
    scalar = params.get('scalar', 0)
    return [inputs[0] - scalar]


def divs_operation(inputs: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
    """内置golden divs"""
    scalar = params.get('scalar', 0)
    return [inputs[0] / scalar]


def muls_operation(inputs: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
    """内置golden muls"""
    scalar = params.get('scalar', 0)
    return [inputs[0] * scalar]


def cast_operation(inputs: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
    """内置golden cast"""
    cast_mode = params.get('mode', 'CAST_NONE')
    dtype_out = dtype_map[params.get('newDataType', 'FP32')]
    x = inputs[0]
    if cast_mode == "CAST_NONE":
        x = x.astype(dtype_out)
    elif cast_mode == "CAST_RINT":
        x = np.rint(x).astype(dtype_out)
    elif cast_mode == "CAST_ROUND":
        x = np.round(x).astype(dtype_out)
    elif cast_mode == "CAST_FLOOR":
        x = np.floor(x).astype(dtype_out)
    elif cast_mode == "CAST_CEIL":
        x = np.ceil(x).astype(dtype_out)
    elif cast_mode == "CAST_TRUNC":
        x = np.trunc(x).astype(dtype_out)
    return [x]


def generate_test_case(test_case: Dict[str, Any], golden_root: str) -> Dict[str, Any]:
    """生成单个测试用例的输入/输出bin（含多输出支持）"""
    # 创建测试用例专属文件夹（golden_root/test_case_name/）
    case_name = test_case["name"]
    case_dir = os.path.join(golden_root, case_name)
    os.makedirs(case_dir, exist_ok=True)
    
    # 生成输入bin
    input_tensors = test_case["input_tensors"]
    input_arrays = []
    updated_inputs = []  # 记录更新后的输入信息（含路径）
    for tensor in input_tensors:
        dtype = dtype_map[tensor["dtype"]]
        shape = tensor["shape"]
        name = tensor["name"]
        
        # 处理数据范围控制
        data_range = tensor.get("data_range", {})
        min_val = data_range.get("min", -100)
        max_val = data_range.get("max", 100)
        
        # 特殊处理：min/max为inf、-inf或nan
        if isinstance(min_val, str):
            min_val = min_val.lower()
            if min_val == 'inf':
                arr = np.full(shape, np.inf, dtype=dtype)
            elif min_val == '-inf':
                arr = np.full(shape, -np.inf, dtype=dtype)
            elif min_val == 'nan':
                arr = np.full(shape, np.nan, dtype=dtype)
            else:
                raise ValueError(f"不支持的数据范围值: {min_val}")
            # 直接跳出，不再处理max_val
        elif isinstance(max_val, str):
            max_val = max_val.lower()
            if max_val == 'inf':
                arr = np.full(shape, np.inf, dtype=dtype)
            elif max_val == '-inf':
                arr = np.full(shape, -np.inf, dtype=dtype)
            elif max_val == 'nan':
                arr = np.full(shape, np.nan, dtype=dtype)
            else:
                raise ValueError(f"不支持的数据范围值: {max_val}")
            # 直接跳出，不再处理后续逻辑
        else:
            if min_val > max_val:
                raise ValueError(f"对于整型数据，min必须小于max。当前配置: min={min_val}, max={max_val}")
            
            if isinstance(dtype, numbers.Integral):
                arr = np.random.randint(min_val, max_val, size=shape, dtype=dtype)
            else:
                arr = np.random.uniform(min_val, max_val, size=shape).astype(dtype)
        
        # 保存到专属文件夹
        bin_path = os.path.join(case_dir, f"{name}.bin")
        arr.tofile(bin_path)
        # 记录更新后的信息
        updated_input = tensor.copy()
        updated_input["path"] = bin_path
        updated_inputs.append(updated_input)
        input_arrays.append(arr)
    
    # 确定操作函数
    if "golden_function" in test_case:
        op_name = test_case["golden_function"]
    else:
        op_name = resolve_operation_function(test_case["operation"])
    
    # 处理特殊的numpy包装函数
    if op_name.startswith("numpy_operation_wrapper"):
        op_func = eval(op_name)
    else:
        op_func = globals().get(op_name)
    
    if not op_func:
        raise ValueError(f"未找到操作函数: {op_name}")
    
    # 执行操作获取输出（返回列表，支持多输出）
    outputs = op_func(input_arrays, test_case.get("params", {}))
    # 检查输出数量与配置匹配
    output_tensors = test_case["output_tensors"]
    if len(outputs) != len(output_tensors):
        raise ValueError(f"操作{op_name}输出数量({len(outputs)})与配置({len(output_tensors)})不匹配")
    
    updated_outputs = []  # 记录更新后的输出信息（含路径）
    for _, (tensor, out_arr) in enumerate(zip(output_tensors, outputs)):
        name = tensor["name"]
        dtype = dtype_map[tensor["dtype"]]
        # 确保输出数据类型正确
        out_arr = out_arr.astype(dtype)
        # 保存到专属文件夹
        bin_path = os.path.join(case_dir, f"{name}.bin")
        out_arr.tofile(bin_path)
        # 记录更新后的信息
        updated_output = tensor.copy()
        updated_output["path"] = bin_path
        updated_outputs.append(updated_output)
    
    # 返回更新后的测试用例信息（含bin路径）
    return {
        **test_case,
        "input_tensors": updated_inputs,
        "output_tensors": updated_outputs
    }


def resolve_operation_function(operation: str) -> str:
    """根据operation字段自动解析对应的函数名"""
    custom_operations = {
        'adds': 'adds_operation',
        'subs': 'subs_operation',
        'muls': 'muls_operation',
        'divs': 'divs_operation',
        'cast': 'cast_operation',
    }
    if operation in custom_operations:
        return custom_operations[operation]
    
    # 尝试映射到numpy函数
    numpy_operations = {
        'add': 'np.add',
        'sub': 'np.subtract',
        'mul': 'np.multiply',
        'div': 'np.divide',
        'reduce_sum': 'np.sum',
        'mean': 'np.mean',
        'reduce_max': 'np.max',
        'reduce_min': 'np.min',
        'cast': 'np.ndarray.astype',
        'transpose': 'np.transpose',
        'reshape': 'np.reshape'
    }
    
    if operation in numpy_operations:
        return f"numpy_operation_wrapper('{numpy_operations[operation]}')"
    
    # 支持通用的ufunc操作（如add, multiply等）
    if hasattr(np, operation) and callable(getattr(np, operation)):
        return f"numpy_operation_wrapper('np.{operation}')"
    
    # 无法解析
    raise ValueError(f"自动解析失败：无法将operation '{operation}' 映射到对应函数。请手动指定golden_function字段。")


def numpy_operation_wrapper(np_func_path: str):
    """包装numpy函数，使其符合操作函数的接口"""
    def wrapper(inputs: List[np.ndarray], params: Dict[str, Any]) -> List[np.ndarray]:
        # 解析numpy函数路径
        parts = np_func_path.split('.')
        func = getattr(np, parts[-1])
        
        # 处理特殊情况
        if parts[-1] == 'astype':
            target_dtype = params.get('target_dtype', 'FP32')
            result = func(inputs[0], dtype=dtype_map[target_dtype])
            return [result]
        
        # 通用情况：直接调用numpy函数
        result = func(*inputs, **params)
        # 确保结果是列表（支持多输出）
        return [result] if not isinstance(result, tuple) else list(result)
    
    return wrapper


def main():
    parser = argparse.ArgumentParser(description='Generate golden files for test cases.')
    parser.add_argument('--path', type=str, default='test_suite.json', help='Path to the test suite JSON file.')
    parser.add_argument('--name', type=str, default=None, help='Name of the specific test case to generate.')
    args = parser.parse_args()

    # 读取原始配置
    with open(args.path, "r") as f:
        test_suite = json.load(f)
    golden_root = test_suite["golden_dir"]
    
    # 生成所有测试用例并更新路径
    updated_test_cases = []
    for case in test_suite["test_cases"]:
        if args.name is None or case["name"] == args.name:
            updated_case = generate_test_case(case, golden_root)
            updated_test_cases.append(updated_case)
        else:
            updated_test_cases.append(case)
    
    # 生成result.json（含所有bin路径）
    result_suite = {
        "golden_dir": golden_root,
        "test_cases": updated_test_cases
    }

    file_path = f"{Path(args.path).stem}_result.json"
    output_dir = os.path.dirname(file_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(result_suite, f, indent=4)
    print(f"生成{file_path}完成")

if __name__ == "__main__":
    main()