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
"""测试 pypto.frontend.jit 对非 pypto.Tensor 参数的支持。"""

import os
import pypto
import torch


# =============================================================================
# 通过 kwargs 传入 non-tensor 参数
# =============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_kernel(
    a: pypto.Tensor((32, 32), pypto.DT_INT32),
    b: pypto.Tensor((32, 32), pypto.DT_INT32),
    scalar=None,
) -> pypto.Tensor((32, 32), pypto.DT_INT32):
    pypto.set_vec_tile_shapes(16, 16)
    c = a + b
    out = c + scalar
    return out


def test_add_with_tiling_kwargs():
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", 0)
    a = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    b = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    r = add_kernel(a, b, scalar=5)
    assert r.shape == (32, 32)
    assert torch.allclose(r.cpu().float(), torch.ones(32, 32) * 2 + 5)


# =============================================================================
# 通过 args 传入 non-tensor 参数
# =============================================================================

def test_add_with_tiling_args():
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", 0)
    a = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    b = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    r = add_kernel(a, b, 5)
    assert r.shape == (32, 32)
    assert torch.allclose(r.cpu().float(), torch.ones(32, 32) * 2 + 5)


# =============================================================================
# 混合情况
# =============================================================================
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def add_npu_with_tiling(
    a: pypto.Tensor((32, 32), pypto.DT_INT32),
    b: pypto.Tensor((32, 32), pypto.DT_INT32),
    tiling=None,
    scalar=1,
) -> pypto.Tensor((32, 32), pypto.DT_INT32):
    pypto.set_vec_tile_shapes(tiling, tiling)
    c = a + b
    d = c + scalar
    return d


def test_add_npu_with_tiling():
    device_id = os.environ.get("TILE_FWK_DEVICE_ID", 0)
    a = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")
    b = torch.ones(32, 32, dtype=torch.int32, device=f"npu:{device_id}")

    r1 = add_npu_with_tiling(a, b, 32)
    r2 = add_npu_with_tiling(a, b, 32, scalar=2)

    assert torch.allclose(r1.cpu().float(), torch.ones(32, 32) * 2 + 1)
    assert torch.allclose(r2.cpu().float(), torch.ones(32, 32) * 2 + 2)