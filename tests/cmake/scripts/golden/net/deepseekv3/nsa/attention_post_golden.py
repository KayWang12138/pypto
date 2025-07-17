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
""" AttentionPost 子图 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import sys
import math
import logging
from pathlib import Path
from typing import List

import numpy as np
from bfloat16 import bfloat16
np.random.seed(0)


if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


fp32 = np.float32


def quant(input_t, is_pertoken: bool = True, has_smooth=False, smooth_cq=None):
    input_fp32 = input_t.astype(fp32)
    if has_smooth:
        input_fp32 = input_fp32 * smooth_cq
    abs_res = np.abs(input_fp32)
    reduce_idx = -1
    if not is_pertoken:
        reduce_idx = -2
        logging.debug("This PerChannel Quant!!")

    max_value = np.max(abs_res, axis=reduce_idx, keepdims=True)
    scale_quant = 127 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.rint(out_fp32).astype(np.int32)
    out_fp16 = out_int32.astype(np.float16)
    out_int8 = np.trunc(out_fp16).astype(np.int8)
    scale_dequant = 1 / scale_quant

    return out_int8, scale_dequant


def gen_post_test_data(output_dir: Path, params, dtype, is_quant=True, is_nz=True):
    b, n, s, h, kv_lora_rank, v_head_dim = params
    x_shape = [b, s, n, kv_lora_rank]
    w_uv_shape = [n, kv_lora_rank, v_head_dim]
    w_o_shape = [n * v_head_dim, h]
    w_o_scale_shape = [1, h]
    logging.debug("x shape is %s", x_shape)
    logging.debug("w_uv shape is %s", w_uv_shape)
    logging.debug("w_0 shape is %s", w_o_shape)
    logging.debug("w_o_scale shape is %s", w_o_scale_shape)

    input_path = Path(output_dir, 'x.bin')
    w_uv_path = Path(output_dir, 'w_uv.bin')
    w_o_path = Path(output_dir, 'w_o.bin')
    w_o_scale_path = Path(output_dir, 'w_o_scale.bin')
    # output
    output_path = Path(output_dir, 'golden_output.bin')

    input = np.random.uniform(-1, 1, x_shape).astype(dtype)
    input.tofile(input_path)
    w_uv = np.random.uniform(-0.1, 0.1, w_uv_shape).astype(dtype)
    w_uv.tofile(w_uv_path)
    w_o = np.random.uniform(-0.1, 0.1, w_o_shape).astype(dtype)
    if is_quant:
        # per_channel, w_o_scale: [1, h]
        w_o_quant, w_o_scale = quant(w_o, False)
        if is_nz:
            w_o_quant.reshape(w_o_shape[0], w_o_shape[1] // 32, 32).transpose(1,0,2).tofile(w_o_path)
        else:
            w_o_quant.tofile(w_o_path)
        w_o_scale.tofile(w_o_scale_path)
        logging.debug("w_o_scale shape is %s", w_o_scale.shape)
    else:
        if is_nz:
            w_o.reshape(w_o_shape[0], w_o_shape[1] // 16, 16).transpose(1,0,2).tofile(w_o_path)
        else:
            w_o.tofile(w_o_path)

    logging.debug("================ calculate ================")
    x_reshape = input.reshape(b * s, n, kv_lora_rank)
    x_trans = np.transpose(x_reshape, (1, 0, 2))  # [n, b*s, kv_lora_rank]
    # [n, b*s, kv_lora_rank] @ [n, kv_lora_rank, v_head_dim] -> [n, b*s, v_head_dim]
    bmm = np.matmul(x_trans.astype(np.float32), w_uv.astype(np.float32))
    bmm = bmm.astype(dtype)

    bmm_trans = np.transpose(bmm, (1, 0, 2))  # [b*s, n, v_head_dim]
    bmm_reshape = bmm_trans.reshape(b * s, n * v_head_dim)  # [b*s, n*v_head_dim]
    if is_quant:
        # quant, per_token
        # scale_dequant: [b*s, 1]
        bmm_reshape, scale_dequant = quant(bmm_reshape, True)  # int8, fp32
        mm = np.matmul(bmm_reshape.astype(np.int32), w_o_quant.astype(np.int32))

        # dequant
        mm_fp32 = mm.astype(fp32)  # [b*s, h]
        mm_fp32_dequant = mm_fp32 * scale_dequant
        mm = mm_fp32_dequant * w_o_scale
    else:
        mm = np.matmul(bmm_reshape.astype(fp32), w_o.astype(fp32))
    mm = mm.astype(dtype)

    output = mm.reshape(b, s, h)
    output.tofile(output_path)

    return output


@GoldenRegister.reg_golden_func(
    case_names = [
        # fp16
        "AttentionPostSTest.b16_s1_nz_fp16_quant",
        "AttentionPostSTest.b16_s2_nz_fp16_quant",
        "AttentionPostSTest.b32_s1_nz_fp16_quant",
        "AttentionPostSTest.b32_s2_nz_fp16_quant",
        "AttentionPostSTest.b64_s1_nz_fp16_quant",
        "AttentionPostSTest.b64_s2_nz_fp16_quant",
        "AttentionPostSTest.b24_s1_nz_fp16_quant",
        "AttentionPostSTest.b24_s2_nz_fp16_quant",
        "AttentionPostSTest.b48_s1_nz_fp16_quant",
        "AttentionPostSTest.b48_s2_nz_fp16_quant",
        "AttentionPostSTest.b96_s1_nz_fp16_quant",
        "AttentionPostSTest.b96_s2_nz_fp16_quant",
        # bf16
        "AttentionPostSTest.b16_s1_nz_bf16_quant",
        "AttentionPostSTest.b16_s2_nz_bf16_quant",
        "AttentionPostSTest.b32_s1_nz_bf16_quant",
        "AttentionPostSTest.b32_s2_nz_bf16_quant",
        "AttentionPostSTest.b64_s1_nz_bf16_quant",
        "AttentionPostSTest.b64_s2_nz_bf16_quant",
        "AttentionPostSTest.b24_s1_nz_bf16_quant",
        "AttentionPostSTest.b24_s2_nz_bf16_quant",
        "AttentionPostSTest.b48_s1_nz_bf16_quant",
        "AttentionPostSTest.b48_s2_nz_bf16_quant",
        "AttentionPostSTest.b96_s1_nz_bf16_quant",
        "AttentionPostSTest.b96_s2_nz_bf16_quant",
        # fp16, nd, quant
        "AttentionPostSTest.b32_s1_nd_fp16_quant",
        "AttentionPostSTest.b32_s2_nd_fp16_quant",
        # fp16, nz, no quant
        "AttentionPostSTest.b32_s1_nz_fp16",
        "AttentionPostSTest.b32_s2_nz_fp16",
        # fp16, nd, no quant
        "AttentionPostSTest.b32_s1_nd_fp16",
        "AttentionPostSTest.b32_s2_nd_fp16",
    ]
)


def gen_post_date(case_name: str, output: Path) -> bool:
    # b, n, s, h, kv_lora_rank, v_head_dim
    # fp16, nz, quant
    if case_name == "AttentionPostSTest.b16_s1_nz_fp16_quant":
        gen_post_test_data(output, (16, 128, 1, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b16_s2_nz_fp16_quant":
        gen_post_test_data(output, (16, 128, 2, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b32_s1_nz_fp16_quant":
        gen_post_test_data(output, (32, 128, 1, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b32_s2_nz_fp16_quant":
        gen_post_test_data(output, (32, 128, 2, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b64_s1_nz_fp16_quant":
        gen_post_test_data(output, (64, 128, 1, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b64_s2_nz_fp16_quant":
        gen_post_test_data(output, (64, 128, 2, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b24_s1_nz_fp16_quant":
        gen_post_test_data(output, (24, 128, 1, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b24_s2_nz_fp16_quant":
        gen_post_test_data(output, (24, 128, 2, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b48_s1_nz_fp16_quant":
        gen_post_test_data(output, (48, 128, 1, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b48_s2_nz_fp16_quant":
        gen_post_test_data(output, (48, 128, 2, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b96_s1_nz_fp16_quant":
        gen_post_test_data(output, (96, 128, 1, 7168, 512, 128), np.float16, True, True)
    elif case_name == "AttentionPostSTest.b96_s2_nz_fp16_quant":
        gen_post_test_data(output, (96, 128, 2, 7168, 512, 128), np.float16, True, True)
    # bf16, nz, quant
    elif case_name == "AttentionPostSTest.b16_s1_nz_bf16_quant":
        gen_post_test_data(output, (16, 128, 1, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b16_s2_nz_bf16_quant":
        gen_post_test_data(output, (16, 128, 2, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b32_s1_nz_bf16_quant":
        gen_post_test_data(output, (32, 128, 1, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b32_s2_nz_bf16_quant":
        gen_post_test_data(output, (32, 128, 2, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b64_s1_nz_bf16_quant":
        gen_post_test_data(output, (64, 128, 1, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b64_s2_nz_bf16_quant":
        gen_post_test_data(output, (64, 128, 2, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b24_s1_nz_bf16_quant":
        gen_post_test_data(output, (24, 128, 1, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b24_s2_nz_bf16_quant":
        gen_post_test_data(output, (24, 128, 2, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b48_s1_nz_bf16_quant":
        gen_post_test_data(output, (48, 128, 1, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b48_s2_nz_bf16_quant":
        gen_post_test_data(output, (48, 128, 2, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b96_s1_nz_bf16_quant":
        gen_post_test_data(output, (96, 128, 1, 7168, 512, 128), bfloat16, True, True)
    elif case_name == "AttentionPostSTest.b96_s2_nz_bf16_quant":
        gen_post_test_data(output, (96, 128, 2, 7168, 512, 128), bfloat16, True, True)
    # fp16, nd, quant
    elif case_name == "AttentionPostSTest.b32_s1_nd_fp16_quant":
        gen_post_test_data(output, (32, 128, 1, 7168, 512, 128), np.float16, True, False)
    elif case_name == "AttentionPostSTest.b32_s2_nd_fp16_quant":
        gen_post_test_data(output, (32, 128, 2, 7168, 512, 128), np.float16, True, False)
    # fp16, nz, no quant
    elif case_name == "AttentionPostSTest.b32_s1_nz_fp16":
        gen_post_test_data(output, (32, 128, 1, 7168, 512, 128), np.float16, False, True)
    elif case_name == "AttentionPostSTest.b32_s2_nz_fp16":
        gen_post_test_data(output, (32, 128, 2, 7168, 512, 128), np.float16, False, True)
    # fp16, nd, no quant
    elif case_name == "AttentionPostSTest.b32_s1_nd_fp16":
        gen_post_test_data(output, (32, 128, 1, 7168, 512, 128), np.float16, False, False)
    elif case_name == "AttentionPostSTest.b32_s2_nd_fp16":
        gen_post_test_data(output, (32, 128, 2, 7168, 512, 128), np.float16, False, False)

    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "AttentionPostSTest.b16_s1_nz_fp16_quant",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_post_date(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    # 只有当脚本作为主程序执行时，才会调用 main()
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    exit(0 if main() else 1)
