#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
"""
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose

def add_op(b: torch.Tensor) -> torch.Tensor:
    b_shape = b.shape
    out_shape = (16, 16)
    mode = pypto.RunMode.NPU
    
    @pypto.frontend.jit(
    host_options={"only_codegen": True}, runtime_options={"run_mode": mode}
    )
    def add_kernel(
        b: pypto.Tensor(b_shape, pypto.DT_FP32)
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        pypto.set_options('profile_enable', True)
        pypto.set_debug_options(runtime_debug_mode=1)

        pypto.set_vec_tile_shapes(16, 16)
        
        c = b[0:16, 0:16]
        for i in range(1,4,1): # or [1,2,3]
            c = pypto.add(c, b[i*16:(i+1)*16, i*16:(i+1)*16])
        
        return c

    out = add_kernel(b)
    return out


def test_add(device_id: int = 1):
    """Test basic concat"""
    print("=" * 60)
    print("Test: Basic Concat")
    print("=" * 60)
    
    device = f'npu:{device_id}'
    
    torch.manual_seed(42)
    
    b = torch.rand((64, 64), dtype=torch.float32, device=device)
    expected = b[:16,:16]+b[16:32,16:32]+b[32:48,32:48]+b[48:,48:]

    out = add_op(b)
    assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(), rtol=1e-3, atol=1e-3)
    print(f"Output: {out}")
    print(f"Expected: {expected}")
    print("✓ Basic concat completed successfully")

if __name__ == "__main__":
    test_add()

