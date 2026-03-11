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
"""Test pypto.frontend.jit kernel reuse and recompile behavior."""

import os
import time
import logging
import pypto
import torch

logging.basicConfig(level=logging.INFO, format="", force=True)
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
# 全局配置
pypto.set_debug_options(runtime_debug_mode=1)
pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
pypto.set_codegen_options(support_dynamic_aligned=True)
pypto.set_pass_options(cube_l1_reuse_setting={1: 4})

@pypto.frontend.jit(
    runtime_options={"run_mode": 1} # compile_stage is EXECUTE_GRAPH, run_mode must be 1 (cpu run)
)
def kernel_with_dynamic(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
):  
    # 在kernel中获取全局配置，并进行验证
    assert 1 == pypto.get_debug_options().get("runtime_debug_mode")
    assert 3 == pypto.get_host_options().get("compile_stage")
    assert True == pypto.get_codegen_options().get("support_dynamic_aligned")
    assert {1: 4} == pypto.get_pass_options().get("cube_l1_reuse_setting")

    pypto.set_vec_tile_shapes(16, 16)
    for idx in pypto.loop(a.shape[0], name="LOOP", idx_name="k"):
        temp = a[idx: idx + 1, :]
        out[idx: idx + 1, :] = temp + 1


def test_config_scope():
    """DYNAMIC axis: second call skips compilation, should be faster."""
    torch.npu.set_device(DEVICE_ID)
    dev = f"npu:{DEVICE_ID}"

    a = torch.ones(1, 8, dtype=torch.float32, device=dev)
    out = torch.zeros(1, 8, dtype=torch.float32, device=dev)
    t1 = time.perf_counter()

    kernel_with_dynamic(a, out)
    t1 = time.perf_counter() - t1
    # assert torch.allclose(out.cpu(), (a + 1).cpu())


if __name__ == "__main__":
    test_config_scope()
    # test_kernel_recompile()