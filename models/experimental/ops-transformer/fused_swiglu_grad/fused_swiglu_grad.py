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
Fused SwiGLU Grad Operator Test
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


def golden_fused_swiglu_bwd(dy, g, fc, w_g, w_fc, x):
    sigmoid_g = torch.sigmoid(g)
    silu_g = g * sigmoid_g
    dg = dy.float() * fc * sigmoid_g * (1 + g * (1 - sigmoid_g))
    dfc = dy.float() * silu_g
    db_g = (dg.sum(dim=0)).to(torch.bfloat16)
    db_fc = (dfc.sum(dim=0)).to(torch.bfloat16)
    dw_g = (x.float().T @ dg).to(torch.bfloat16)
    dw_fc = (x.float().T @ dfc).to(torch.bfloat16)
    dx = (dg @ w_g.float().T + dfc @ w_fc.float().T).to(torch.bfloat16)
    return dx, dw_g, dw_fc, db_g, db_fc


def test_bwd(M, K, N, device_id):
    print(f"\n=== Backward Test [{M}, {K}] @ [{K}, {N}] ===")
    device = f'npu:{device_id}'
    np.random.seed(0)
    torch.manual_seed(0)

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) / math.sqrt(M)
    w_g = torch.randn(K, N, dtype=torch.bfloat16, device=device) / math.sqrt(K)
    w_fc = torch.randn(K, N, dtype=torch.bfloat16, device=device) / math.sqrt(K)
    dy = torch.randn(M, N, dtype=torch.bfloat16, device=device) / math.sqrt(M)
    g = torch.randn(M, N, dtype=torch.bfloat16, device=device) / math.sqrt(M)
    fc = torch.randn(M, N, dtype=torch.bfloat16, device=device) / math.sqrt(M)
    dx_golden, dw_g_golden, dw_fc_golden, db_g_golden, db_fc_golden = golden_fused_swiglu_bwd(dy, g, fc, w_g, w_fc, x)

    dx_out = torch.empty(M, K, dtype=torch.bfloat16, device=device)
    dw_g_out = torch.zeros(K, N, dtype=torch.bfloat16, device=device)
    dw_fc_out = torch.zeros(K, N, dtype=torch.bfloat16, device=device)
    db_g_out = torch.zeros(N, dtype=torch.bfloat16, device=device)
    db_fc_out = torch.zeros(N, dtype=torch.bfloat16, device=device)
    dg_out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    dfc_out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    from fused_swiglu_grad_impl import fused_swiglu_bwd_b_kernel, fused_swiglu_bwd_w_kernel, fused_swiglu_bwd_x_kernel
    fused_swiglu_bwd_b_kernel(dy, g, fc, dg_out, dfc_out, db_g_out, db_fc_out, K, N)
    fused_swiglu_bwd_w_kernel(x, dg_out, dfc_out, dw_g_out, dw_fc_out, K, N)
    fused_swiglu_bwd_x_kernel(dg_out, dfc_out, w_g, w_fc, dx_out, K, N)

    print("Checking gradients...")
    assert_allclose(dx_out.cpu().float().numpy(), dx_golden.cpu().float().numpy(), rtol=0.01, atol=0.01)
    print("  ✓ dx")
    assert_allclose(dw_g_out.cpu().float().numpy(), dw_g_golden.cpu().float().numpy(), rtol=0.01, atol=0.01)
    print("  ✓ dw_g")
    assert_allclose(dw_fc_out.cpu().float().numpy(), dw_fc_golden.cpu().float().numpy(), rtol=0.01, atol=0.01)
    print("  ✓ dw_fc")
    assert_allclose(db_g_out.cpu().float().numpy(), db_g_golden.cpu().float().numpy(), rtol=0.01, atol=0.01)
    print("  ✓ db_g")
    assert_allclose(db_fc_out.cpu().float().numpy(), db_fc_golden.cpu().float().numpy(), rtol=0.01, atol=0.01)
    print("  ✓ db_fc")
    print("✓ Backward passed")


def main():
    device_id = get_device_id()
    if device_id is None:
        return
    torch.npu.set_device(device_id)
    print("Fused SwiGLU Grad Test")
    test_bwd(220000, 512, 1024, device_id)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()