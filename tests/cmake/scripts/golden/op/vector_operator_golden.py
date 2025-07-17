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
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


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
