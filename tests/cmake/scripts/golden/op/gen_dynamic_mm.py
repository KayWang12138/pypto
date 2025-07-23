
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
"""

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
"""
import math
import sys
import logging
from pathlib import Path

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

fp32 = np.float32
fp16 = np.float16
bf16 = bfloat16
int32 = np.int32
int8 = np.int8


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def ceil_div(a, b):
    return (a + b - 1) // b


def nd_to_fractal_nz(data: np.ndarray):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    if data.dtype == "int8":
        m0, n0 = 16, 32
    else:
        m0, n0 = 16, 16
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = np.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


class ShapeConfig:
    def __init__(self, m: int, k: int, n: int, dtype: str, out_dtype: str):
        self.m = m
        self.k = k
        self.n = n
        self.dtype = dtype
        self.out_dtype = out_dtype


def gen_mm_data(input_config: ShapeConfig, output_dir: Path, is_b_trans=False, is_b_nz=False):
    shape_a = [input_config.m, input_config.k]
    shape_b = [input_config.k, input_config.n]
    shape_c = [input_config.m, input_config.n]

    a_path = Path(output_dir, 'mat_a.bin')
    b_path = Path(output_dir, 'mat_b.bin')
    c_path = Path(output_dir, 'mat_c.bin')

    if input_config.dtype == 'int8':
        a = np.random.randint(-4, 5, shape_a).astype(int8)
        b = np.random.randint(-4, 5, shape_b).astype(int8)
        c = np.matmul(a.astype(int32), b.astype(int32)).astype(int32)
    elif input_config.dtype == 'fp16':
        a = np.random.uniform(-1, 1, shape_a).astype(fp16)
        b = np.random.uniform(-1, 1, shape_b).astype(fp16)
        c = np.matmul(a.astype(fp32), b.astype(fp32))
    else:
        a = np.random.uniform(-1, 1, shape_a).astype(bf16)
        b = np.random.uniform(-1, 1, shape_b).astype(bf16)
        c = np.matmul(a.astype(fp32), b.astype(fp32))
    a.tofile(a_path)
    if is_b_trans:
        b = b.transpose(1, 0)
    if is_b_nz:
        b = nd_to_fractal_nz(b)
    b.tofile(b_path)
    c.tofile(c_path)


@GoldenRegister.reg_golden_func(
    case_names=[
        #matmul
        "DynamicMatmulTest.mm_A_B_ND_bf16",
        "DynamicMatmulTest.mm_A_B_NZ_bf16",
        "DynamicMatmulTest.mm_A_Bt_ND_fp16",
        "DynamicMatmulTest.mm_A_Bt_NZ_fp16",
        "DynamicMatmulTest.mm_A_B_NZ_int8",
        "DynamicMatmulTest.mm_A_Bt_NZ_int8",
        "DynamicMatmulTest.mm_A_B_ND_bf16_tile1",
        "DynamicMatmulTest.mm_A_Bt_ND_fp16_tile2",
        "DynamicMatmulTest.mm_A_B_NZ_int8_tile3",
        "DynamicMatmulTest.mm_A_Bt_NZ_int8_tile4",
    ]
)
def gen_dynamic_mm_golden(case_name: str, output: Path) -> bool:
    if case_name == "DynamicMatmulTest.mm_A_B_ND_bf16":
        input_config = ShapeConfig(128, 256, 512, 'bf16', 'fp32')
        gen_mm_data(input_config, output, False, False)
        return True
    if case_name == "DynamicMatmulTest.mm_A_B_NZ_bf16":
        input_config = ShapeConfig(16, 32, 512, 'bf16', 'fp32')
        gen_mm_data(input_config, output, False, True)
        return True
    if case_name == "DynamicMatmulTest.mm_A_Bt_ND_fp16":
        input_config = ShapeConfig(128, 257, 511, 'fp16', 'fp32')
        gen_mm_data(input_config, output, True, False)
        return True
    if case_name == "DynamicMatmulTest.mm_A_Bt_NZ_fp16":
        input_config = ShapeConfig(1, 512, 256, 'fp16', 'fp32')
        gen_mm_data(input_config, output, True, True)
        return True
    if case_name == "DynamicMatmulTest.mm_A_B_NZ_int8":
        input_config = ShapeConfig(16, 32, 512, 'int8', 'int32')
        gen_mm_data(input_config, output, False, True)
        return True
    if case_name == "DynamicMatmulTest.mm_A_Bt_NZ_int8":
        input_config = ShapeConfig(1, 512, 256, 'int8', 'int32')
        gen_mm_data(input_config, output, True, True)
        return True
    if case_name == "DynamicMatmulTest.mm_A_B_ND_bf16_tile1":
        input_config = ShapeConfig(128, 256, 512, 'bf16', 'fp32')
        gen_mm_data(input_config, output, False, False)
        return True
    if case_name == "DynamicMatmulTest.mm_A_Bt_ND_fp16_tile2":
        input_config = ShapeConfig(16, 512, 512, 'fp16', 'fp32')
        gen_mm_data(input_config, output, True, False)
        return True
    if case_name == "DynamicMatmulTest.mm_A_B_NZ_int8_tile3":
        input_config = ShapeConfig(16, 32, 512, 'int8', 'int32')
        gen_mm_data(input_config, output, False, True)
        return True
    if case_name == "DynamicMatmulTest.mm_A_Bt_NZ_int8_tile4":
        input_config = ShapeConfig(1, 512, 256, 'int8', 'int32')
        gen_mm_data(input_config, output, True, True)
        return True
    else:
        logging.error("Can't get func to gen golden, case(%s)", case_name)
        return False

