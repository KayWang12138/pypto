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
import sys
import math
import logging
from pathlib import Path
from typing import List

import torch
import numpy as np
from bfloat16 import bfloat16
np.random.seed(0)
if __name__ == "__main__":
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister
else:
    from golden_register import GoldenRegister

fp32 = np.float32


def gen_gen_atten_golden_data(params, dtype, output_dir: Path):
    b = params.get("b")
    n = params.get("n")
    s = params.get("s")
    d = params.get("d")

    cmp_atten_shape = [b, s, n, d]
    sel_atten_shape = [b, s, n, d]
    win_atten_shape = [b, s, n, d]
    gating_score_shape = [b, s, n, 3]

    cmp_atten_path = Path(output_dir, 'cmp_atten.bin')
    sel_atten_path = Path(output_dir, 'sel_atten.bin')
    win_atten_path = Path(output_dir, 'win_atten.bin')
    gating_score_path = Path(output_dir, 'gating_score.bin')
    # output
    attention_out_path = Path(output_dir, 'attention_out.bin')

    # gen input
    cmp_atten = np.random.uniform(-1, 1, cmp_atten_shape).astype(dtype)
    cmp_atten.tofile(cmp_atten_path)
    sel_atten = np.random.uniform(-1, 1, sel_atten_shape).astype(dtype)
    sel_atten.tofile(sel_atten_path)
    win_atten = np.random.uniform(-1, 1, win_atten_shape).astype(dtype)
    win_atten.tofile(win_atten_path)
    gating_score = np.random.uniform(-1, 1, gating_score_shape).astype(dtype)
    gating_score.tofile(gating_score_path)

    cmp_atten_fp32 = cmp_atten.astype(fp32)
    sel_atten_fp32 = sel_atten.astype(fp32)
    win_atten_fp32 = win_atten.astype(fp32)
    gating_score_fp32 = gating_score.astype(fp32)
    w_cmp, w_slc, w_win = np.split(gating_score_fp32, 3, axis = -1)
    attention_out_fp32 = (w_cmp * cmp_atten_fp32 + w_slc * sel_atten_fp32 + w_win * win_atten_fp32)
    attention_out = attention_out_fp32.astype(dtype)
    attention_out.tofile(attention_out_path)


def gen_gen_atten_test_s1(dtypes, output_dir: Path):
    params = {
        "b": 16,
        "s": 1,
        "n": 128,
        "d": 512,
    }
    gen_gen_atten_golden_data(params, dtypes, output_dir)


def gen_gen_atten_test_s2(dtypes, output_dir: Path):
    params = {
        "b": 16,
        "s": 2,
        "n": 128,
        "d": 512,
    }
    gen_gen_atten_golden_data(params, dtypes, output_dir)


@GoldenRegister.reg_golden_func(
    case_names=[
        # MLA_prolog v2
        "TestGenAtten.TestOnboardGenAttenTest_FP16_S1",
        "TestGenAtten.TestOnboardGenAttenTest_FP32_S1",
        "TestGenAtten.TestOnboardGenAttenTest_BF16_S1",
        "TestGenAtten.TestOnboardGenAttenTest_FP16_S2",
        "TestGenAtten.TestOnboardGenAttenTest_FP32_S2",
        "TestGenAtten.TestOnboardGenAttenTest_BF16_S2",
    ]
)

def gen_gen_atten_data(case_name: str, output: Path) -> bool:
    if case_name == "TestGenAtten.TestOnboardGenAttenTest_FP16_S1":
        gen_gen_atten_test_s1(np.float16, output)
    elif case_name == "TestGenAtten.TestOnboardGenAttenTest_FP32_S1":
        gen_gen_atten_test_s1(np.float32, output)
    elif case_name == "TestGenAtten.TestOnboardGenAttenTest_BF16_S1":
        gen_gen_atten_test_s1(bfloat16, output)
    elif case_name == "TestGenAtten.TestOnboardGenAttenTest_FP16_S2":
        gen_gen_atten_test_s2(np.float16, output)
    elif case_name == "TestGenAtten.TestOnboardGenAttenTest_FP32_S2":
        gen_gen_atten_test_s2(np.float32, output)
    elif case_name == "TestGenAtten.TestOnboardGenAttenTest_BF16_S2":
        gen_gen_atten_test_s2(bfloat16, output)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


def main() -> bool:
    # 用例名称
    case_name_list: List[str] = [
        "TestGenAtten.TestOnboardGenAttenTest_FP16_S1",
        "TestGenAtten.TestOnboardGenAttenTest_FP32_S1",
        "TestGenAtten.TestOnboardGenAttenTest_BF16_S1",
        "TestGenAtten.TestOnboardGenAttenTest_FP16_S2",
        "TestGenAtten.TestOnboardGenAttenTest_FP32_S2",
        "TestGenAtten.TestOnboardGenAttenTest_BF16_S2",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_gen_atten_data(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
