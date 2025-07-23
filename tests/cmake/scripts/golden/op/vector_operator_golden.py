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
""" vector op 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
"""
import sys
import logging
from pathlib import Path
from typing import List
import math

import numpy as np
from bfloat16 import bfloat16

if __name__ == "__main__":
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister
else:
    from golden_register import GoldenRegister


def dump_file(data_pool, data_path, type_str):
    if type_str.lower() == 'fp16':
        np.array(data_pool).astype(np.float16).tofile(data_path)
    elif type_str.lower() == 'fp32':
        np.array(data_pool).astype(np.float32).tofile(data_path)
    elif type_str.lower() == 'fp64':
        np.array(data_pool).astype(np.float64).tofile(data_path)
    elif type_str.lower() == 'int8':
        np.array(data_pool).astype(np.int8).tofile(data_path)
    elif type_str.lower() == 'int16':
        np.array(data_pool).astype(np.int16).tofile(data_path)
    elif type_str.lower() == 'int32':
        np.array(data_pool).astype(np.int32).tofile(data_path)
    elif type_str.lower() == 'int64':
        np.array(data_pool).astype(np.int64).tofile(data_path)
    elif type_str.lower() == 'uint8':
        np.array(data_pool).astype(np.uint8).tofile(data_path)
    elif type_str.lower() == 'uint16':
        np.array(data_pool).astype(np.uint16).tofile(data_path)
    elif type_str.lower() == 'uint32':
        np.array(data_pool).astype(np.uint32).tofile(data_path)
    elif type_str.lower() == 'uint64':
        np.array(data_pool).astype(np.uint64).tofile(data_path)
    elif type_str.lower() == 'complex64':
        np.array(data_pool).astype(np.complex64).tofile(data_path)
    elif type_str.lower() == 'complex128':
        np.array(data_pool).astype(np.complex128).tofile(data_path)
    elif type_str.lower() == 'bool':
        np.array(data_pool).astype(np.bool_).tofile(data_path)
    elif type_str.lower() == 'bf16':
        np.array(data_pool).astype(bfloat16).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value,
                             size=data_shape).astype(dtype)


def process_test_cases(
    case_name: str,
    base_output: Path,
    test_configs: list,
    generate_golden: callable,
    case_index: int = None
) -> bool:
    if case_index is None:
        for index, config in enumerate(test_configs):
            output_path = Path(base_output, str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            generate_golden_files(case_name, output_path, config)
    else:
        config = test_configs[case_index]
        generate_golden_files(case_name, output, config)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAdd/AddOperationTest.test_add",
    ]
)
def gen_add_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者徐根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def generate_golden_files(case_name: str, output_path: Path, shape: list, dtype) -> bool:
        x_path = Path(output_path, 'x.bin')
        y_path = Path(output_path, 'y.bin')
        o_path = Path(output_path, 'res.bin')

        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True

        x = np.random.uniform(0, 1, shape).astype(dtype)
        y = np.random.uniform(0, 1, shape).astype(dtype)
        x.tofile(x_path)
        y.tofile(y_path)
        (x + y).tofile(o_path)
        return False

    # 测试数据集，根据具体测试需求修改，可增加字段
    test_configs = [
        ([512, 128], np.float32),
        ([1024, 256], np.float32),
        ([512, 256], np.float32),
    ]

    # 1.跑测试套还是单个用力，生成不同场景的文件夹，一般不用改
    # 2.涉及test_configs数据结构变更，generate_golden_files的接口形式和调用传参可能需联动修改
    if case_index is None:
        for index, (shape, dtype) in enumerate(test_configs):
            output_path = Path(str(output) + '/' + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            generate_golden_files(case_name, output_path, shape, dtype)
    else:
        shape, dtype = test_configs[case_index]
        generate_golden_files(case_name, output, shape, dtype)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExp/ExpOperationTest.TestExp",
    ]
)
def gen_exp_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者徐根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def generate_golden_files(case_name: str, output_path: Path, shape: list, dtype, index: int) -> bool:
        x_path = Path(output_path, 'x.bin')
        o_path = Path(output_path, 'res.bin')

        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True

        x = np.random.uniform(0, 1, shape).astype(dtype)
        if index == 6:
            x[0][0] = np.inf
        elif index == 7:
            x[0][0] = np.nan
        elif index == 9:
            x[0][0] = np.finfo(np.float32).max
        elif index == 10:
            x[0][0] = np.finfo(np.float32).min

        x.tofile(x_path)
        (np.exp(x)).tofile(o_path)
        return False

    # 测试数据集，根据具体测试需求修改，可增加字段
    test_configs = [
        ([64 * 48, 1], np.float32),       # 0 1dims
        ([64 * 48, 128 * 3], np.float32), # 1 2dims
        ([64 * 48, 16 * 3], np.float32),  # 2 2dims tileshape 不对齐
        ([64 * 48, 128 * 3], np.float32), # 3 2dims viewshape 不对齐
        ([48 * 128, 64, 64], np.float32), # 5 3dims
        ([64 * 48, 128 * 3], np.float16), # 7 fp16
        ([64 * 48, 128 * 3], np.float32), # 8 inf
        ([64 * 48, 128 * 3], np.float32), # 9 nan
        ([64 * 48, 0], np.float32),       # 10 空tensor
        ([64 * 48, 128 * 3], np.float32), # 11 max
        ([64 * 48, 128 * 3], np.float32), # 12 min
        ([1, 1], np.float32),             # 14 originalshape < viewshape
        ([32, 32], np.float32),           # 15 originalshape == viewshape
        ([128, 128], np.float32),         # 16 viewshape < tileshape
        ([1, 64 * 192], np.float32),      # 19 k_dim 192 b min
        ([96, 64 * 192], np.float32),     # 20 k_dim 192 b max
        ([1, 64 * 128], np.float32),      # 23 v_dim 128 b min
        ([96, 64 * 128], np.float32),     # 24 v_dim 128 b max
    ]

    # 1.跑测试套还是单个用例，生成不同场景的文件夹，一般不用改
    # 2.涉及test_configs数据结构变更，generate_golden_files的接口形式和调用传参可能需联动修改
    if case_index is None:
        for index, (shape, dtype) in enumerate(test_configs):
            output_path = Path(str(output) + '/' + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            generate_golden_files(case_name, output_path, shape, dtype, case_index)
    else:
        shape, dtype = test_configs[case_index]
        generate_golden_files(case_name, output, shape, dtype, case_index)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDiv/DivOperationTest.TestDiv",
    ]
)
def gen_div_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def generate_golden_files(case_name: str, output_path: Path, config: list) -> bool:
        x_path = Path(output_path, "x.bin")
        y_path = Path(output_path, "y.bin")
        o_path = Path(output_path, "res.bin")

        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True

        default_values = {
            1: [None, np.inf],
            2: [np.inf, None],
            3: [np.inf, np.inf],
            4: [None, np.nan],
            5: [np.nan, None],
            6: [np.nan, np.nan],
            7: [np.finfo(np.float32).max, np.finfo(np.float32).min],
            8: [np.finfo(np.float32).min, np.finfo(np.float32).max],
            9: [None, 0.0],
        }
        [x_default_value, y_default_value] = default_values.get(
            case_index, [None, None]
        )
        if x_default_value is None:
            x = np.random.uniform(0, 1, config[0][0]).astype(config[0][1])
        else:
            x = np.full(config[0][0], x_default_value, dtype=config[0][1])
        if y_default_value is None:
            y = np.random.uniform(0, 1, config[1][0]).astype(config[1][1])
        else:
            y = np.full(config[1][0], y_default_value, dtype=config[1][1])

        x.tofile(x_path)
        y.tofile(y_path)
        (x / y).tofile(o_path)
        return False

    # 测试数据集，根据具体测试需求修改，可增加字段
    test_configs = [
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([1024, 128], np.float32), ([1024, 128], np.float32)],
        [([496, 128, 64], np.float32), ([496, 128, 64], np.float32)],
        [([128 + 17, 128], np.float32), ([128 + 17, 128], np.float32)],
        [([90 + 17, 128], np.float32), ([90 + 17, 128], np.float32)],
    ]

    # 1.跑测试套还是单个用例，生成不同场景的文件夹，一般不用改
    # 2.涉及test_configs数据结构变更，generate_golden_files的接口形式和调用传参可能需联动修改
    if case_index is None:
        for index, config in enumerate(test_configs):
            output_path = Path(str(output) + "/" + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            generate_golden_files(case_name, output_path, config)
    else:
        config = test_configs[case_index]
        generate_golden_files(case_name, output, config)
    return True


@GoldenRegister.reg_golden_func(case_names=[
    "TestSub/SubOperationTest.test_sub",
])
def gen_sub_op_golden(case_name: str,
                      output: Path,
                      case_index: int = None) -> bool:
    def generate_golden_files(case_name: str, output_path: Path, shape0: list,
                              shape1: list, dtype) -> bool:
        x_path = Path(output_path, 'x.bin')
        y_path = Path(output_path, 'y.bin')
        o_path = Path(output_path, 'res.bin')

        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True

        x = np.random.uniform(0, 1, shape0).astype(dtype)
        y = np.random.uniform(0, 1, shape1).astype(dtype)
        x.tofile(x_path)
        y.tofile(y_path)
        (x - y).tofile(o_path)
        return False
 
    # 测试数据集
    test_configs = [
        ([128 + 17, 128], [128 + 17, 128], np.float32),
        ([90 + 17, 128], [90 + 17, 128], np.float32),
    ]

    if case_index is None:
        for index, (shape0, shape1, dtype) in enumerate(test_configs):
            output_path = Path(str(output) + '/' + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            generate_golden_files(case_name, output_path, shape0, shape1,
                                  dtype)
    else:
        shape0, shape1, dtype = test_configs[case_index]
        generate_golden_files(case_name, output, shape0, shape1, dtype)
    return True


@GoldenRegister.reg_golden_func(case_names=[
    "TestMuls/MulsOperationTest.test_muls",
])
def gen_muls_op_golden(case_name: str,
                       output: Path,
                       case_index: int = None) -> bool:
    def generate_golden_files(case_name: str, output_path: Path,
                              datainfo: list) -> bool:

        shape, dtype, ele, default_value = datainfo
        x_path = Path(output_path, 'x.bin')
        y_path = Path(output_path, 'y.bin')
        o_path = Path(output_path, 'res.bin')

        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True

        if default_value is None:
            x = np.random.uniform(0, 1, shape).astype(dtype)
        else:
            x = np.full(shape, default_value, dtype=dtype)
        y = np.array([ele], dtype=np.float32)

        x.tofile(x_path)
        y.tofile(y_path)
        (x * np.float32(ele)).tofile(o_path)
        return False

    test_configs = [
        ([64 * 48, 512], np.float32, 1.0 / 512.0, None),  # 0
        ([64 * 48, 128 * 3], np.float32, -1.0, None),  # 1
        ([64 * 32, 16 * 3], np.float32, -1.0, None),  # 2
        ([64 * 48 + 3, 128 * 3], np.float32, -1.0, None),  # 3
        ([64 * 48, 128 * 3], np.float32, -1.0, None),  # 4
        ([64 * 48, 128 * 3], np.float32, -1.0, None),  # 5
        ([48 * 128, 64, 64], np.float32, 1.0 / math.sqrt(576.0), None),  # 6
        ([16, 16, 128, int(64 / 2)], np.float32, -1.0, None),  # 7
        ([64 * 48, 128 * 3], np.float32, -1.0, np.inf),  # 8
        ([64 * 48, 128 * 3], np.float32, -1.0, np.nan),  # 9
        ([64 * 48, 0], np.float32, -1.0, None),  # 10
        ([1, 1], np.float32, -1.0, None),  # 12
        ([32, 32], np.float32, -1.0, None),  # 13
        ([128, 128], np.float32, -1.0, None),  # 14
    ]

    if case_index is None:
        for index, (shape, dtype, ele,
                    default_value) in enumerate(test_configs):
            output_path = Path(str(output) + '/' + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            generate_golden_files(case_name, output_path,
                                  [shape, dtype, ele, default_value])
    else:
        shape, dtype, ele, default_value = test_configs[case_index]
        generate_golden_files(case_name, output,
                              [shape, dtype, ele, default_value])
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDivs/DivsOperationTest.test_divs",
    ]
)
def gen_divs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def generate_golden_files(case_name: str, output_path: Path, params) -> bool:
        x_path = Path(output_path, 'x.bin')
        y_path = Path(output_path, 'y.bin')
        o_path = Path(output_path, 'res.bin')

        default_val = None
        shape, dtype, ele = params[0], params[1], params[2]
        if len(params) > 3:
            default_val = params[3]

        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True

        if default_val is None:
            x = np.random.uniform(0, 1, shape).astype(dtype)
        else:
            x = np.full(shape, default_val, dtype=dtype)
        x.tofile(x_path)

        y = np.array([ele], dtype=dtype)
        y.tofile(y_path)

        (x / np.float32(ele)).tofile(o_path)
        return False

    # 测试数据集，根据具体测试需求修改，可增加字段
    test_configs = [
        ([64 * 48, 512], np.float32, 512.0),
        ([64 * 48, 128 * 3], np.float32, -1.0),
        ([64 * 32, 16 * 3], np.float32, -1.0),
        ([64 * 32 + 3, 16 * 3], np.float32, -1.0),
        ([64 * 48, 128 * 3], np.float32, -1.0),
        ([48 * 128, 64, 64], np.float32, math.sqrt(576)),
        ([16, 16, 128, 32], np.float32, -1.0),
        ([64 * 48, 128 * 3], np.float32, -1.0, np.inf),
        ([64 * 48, 128 * 3], np.float32, -1.0, np.nan),
        ([64 * 48, 0], np.float32, -1.0),
        ([64 * 48, 128 * 3], np.float32, 0.0),
        ([64 * 48, 128 * 3], np.float32, np.finfo(np.float32).max),
        ([64 * 48, 128 * 3], np.float32, np.finfo(np.float32).min),
        ([64 * 48, 128 * 3], np.float32, -1.0, np.finfo(np.float32).max),
        ([64 * 48, 128 * 3], np.float32, -1.0, np.finfo(np.float32).min),
        ([1, 1], np.float32, -1.0),
        ([32, 32], np.float32, -1.0),
        ([128, 128], np.float32, -1.0),
        ([16, 1536], np.float32, 1 / 1536),
        ([16, 512], np.float32, 1 / 512),
        ([16, 128 * 3], np.float32, -1),
    ]

    # 1.跑测试套还是单个用例，生成不同场景的文件夹，一般不用改
    # 2.涉及test_configs数据结构变更，generate_golden_files的接口形式和调用传参可能需联动修改
    if case_index is None:
        for index, config in enumerate(test_configs):
            output_path = Path(str(output) + '/' + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            generate_golden_files(case_name, output_path, config)
    else:
        config = test_configs[case_index]
        generate_golden_files(case_name, output, config)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAdds/AddsOperationTest.test_adds",
    ]
)
def gen_adds_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def generate_golden_files(case_name: str, output_path: Path, shape: list, dtype, ele) -> bool:
        x_path = Path(output_path, 'x.bin')
        y_path = Path(output_path, 'y.bin')
        o_path = Path(output_path, 'res.bin')

        complete = x_path.exists() and y_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        x = np.random.uniform(0, 1, shape).astype(dtype)
        x.tofile(x_path)
        y = np.array([ele], dtype=np.float32)
        y.tofile(y_path)
        (x + np.float32(ele)).tofile(o_path)
        return False

    # 测试数据集，根据具体测试需求修改，可增加字段
    test_configs = [
        ([3072, 1], np.float32, 1e-5),
        ([64 * 48, 128 * 3], np.float32, 1.0),
        ([64 * 32, 16 * 3], np.float32, 1.0),
        ([64 * 32 + 3, 16 * 3], np.float32, 1.0),
        ([64 * 48, 128 * 3], np.float32, 1.0),
        ([48 * 128, 64, 64], np.float32, 1.0 / 24),
        ([1, 1], np.float32, 1.0),
        ([32, 32], np.float32, 1.0),
        ([128, 128], np.float32, 1.0),
        ([48, 64, 128, 32], np.float32, 1.0),
        ([64 * 48, 0], np.float32, 2.0),
        ([0, 128 * 3], np.float32, 2.0),
        ([64 * 48, 128 * 3], np.float32, np.finfo(np.float32).max),
        ([96 * 1024, 128 * 3], np.float32, 1.0),
        ([64, 64 * 576], np.float32, 1.0),            # [b, 64*k_dim] + 1 (fp32)
        ([64, 64 * 512], np.float32, 1.0),            # [b, 64*v_dim] + 1 (fp32)
        ([64 * 32, 7168 * 4], np.float32, 1.0),         # [b*s, h*4] + 1 (fp32)
        ([96, 64 * 576], np.float32, 1.0),
        ([96, 64 * 512], np.float32, 1.0),
        ([768 * 1024, 128 * 3], np.float32, 1.0),
    ]

    return process_test_cases(case_name, output, test_configs, generate_golden_files, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestVectorDup/VectorDupOperationTest.test_vector_dup",
    ]
)
def gen_vector_dup_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def generate_golden_files(case_name: str, output_path: Path, shape: list, dtype, ele) -> bool:
        x_path = Path(output_path, 'x.bin')
        o_path = Path(output_path, 'res.bin')
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True

        x = np.full([1, 1], ele, dtype=dtype)
        x.tofile(x_path)
        tensor = np.full(shape, ele, dtype=dtype)
        tensor.tofile(o_path)
        return False
    # 测试数据集，根据具体测试需求修改，可增加字段
    test_configs = [
        ([3072, 1], np.float32, 1e-5),
        ([64 * 48, 128 * 3], np.float32, 2.0),
        ([64 * 32, 16 * 3], np.float32, 2.0),
        ([64 * 32 + 3, 16 * 3], np.float32, 2.0),
        ([64 * 48, 128 * 3], np.float32, 2.0),
        ([48 * 128, 64, 64], np.float32, 2.0),
        ([1, 1], np.float32, 2.0),
        ([32, 32], np.float32, 2.0),
        ([128, 128], np.float32, 2.0),
        ([48, 64, 128, 32], np.float32, 2.0),
        ([64 * 48, 0], np.float32, 2.0),
        ([0, 128 * 3], np.float32, 2.0),
        ([64 * 48, 1536], np.float32, 2.0),
        ([64 * 48, 512], np.float32, 2.0),
        ([64 * 48, 7168 * 4], np.float32, 2.0),
    ]

    return process_test_cases(case_name, output, test_configs, generate_golden_files, case_index)


def main() -> bool:
    # 用例名称
    case_name_list: List[str] = [
        "TestAdd/AddOperationTest.test_add",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_add_op_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
