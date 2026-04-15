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
"""
Fused SwiGLU Operator Test
"""

import os
import sys
import math
import torch
import numpy as np
from numpy.testing import assert_allclose
import pypto


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set TILE_FWK_DEVICE_ID")
        return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])


def golden_fused_swiglu_fwd(x, w_g, w_fc, b_g, b_fc):
    gate = torch.nn.functional.silu(torch.nn.functional.linear(x.float(), w_g.float().T, b_g))
    fc = torch.nn.functional.linear(x.float(), w_fc.float().T, b_fc).to(torch.bfloat16)
    y = (gate * fc).to(torch.bfloat16)
    return y


def test_fwd(M, K, N, device_id):
    print(f"\n=== Forward Test [{M}, {K}] @ [{K}, {N}] ===")
    device = f'npu:{device_id}'
    np.random.seed(0)
    torch.manual_seed(0)

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) / math.sqrt(M)
    w_g = torch.randn(K, N, dtype=torch.bfloat16, device=device) / math.sqrt(K)
    w_fc = torch.randn(K, N, dtype=torch.bfloat16, device=device) / math.sqrt(K)
    b_g = torch.randn(N, dtype=torch.bfloat16, device=device) / math.sqrt(N)
    b_fc = torch.randn(N, dtype=torch.bfloat16, device=device) / math.sqrt(N)
    y_golden = golden_fused_swiglu_fwd(x, w_g, w_fc, b_g, b_fc)
    y_out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    from fused_swiglu_impl import fused_swiglu_fwd_kernel
    fused_swiglu_fwd_kernel(x, w_g, w_fc, b_g, b_fc, y_out, K, N)

    assert_allclose(y_out.cpu().float().numpy(), y_golden.cpu().float().numpy(), rtol=0.01, atol=0.01)
    print(f"Output range: [{y_golden.min().item():.4f}, {y_golden.max().item():.4f}]")
    print("✓ Forward passed")


def main():
    device_id = get_device_id()
    if device_id is None:
        return
    torch.npu.set_device(device_id)
    print("Fused SwiGLU Test")
    test_fwd(220000, 512, 1024, device_id)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()