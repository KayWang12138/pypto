# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for pl.getval and pl.setval functions.

Tests that getval and setval work correctly with Tile operations.
"""

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm


# ---------------------------------------------------------------------------
# Test kernels — defined at module level so @fe.kernel runs at import time
# ---------------------------------------------------------------------------

# Kernel: get first element and set it to another position
@fe.kernel
def tile_getval_setval_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:

    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                              addr=0x0000, size=16384)
    with pl.section_vector():
        plm.load(tile_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.S, event_id=0)
        value = pl.block.getval(tile_a, 0)
        tile_a = pl.block.setval(tile_a, 1, value)
        pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(a, tile_a, [0, 0])
    pl.system.bar_all()
    tensor_value = pl.tensor.getval(a, 100)
    pl.tensor.setval(a, 101, tensor_value)
    pl.system.bar_all()
    return a


# ---------------------------------------------------------------------------
# Test functions (run with: python test_getval_setval.py)
# ---------------------------------------------------------------------------

def _verify_outputs(tag, tensor, original_values):
    """Verify getval/setval modified input correctly.
    
    Args:
        tag: Backend name (CCE/PTO)
        tensor: Modified tensor after kernel execution
        original_values: Dict with original values before modification
    """
    # block.getval/setval: a[0,1] should equal original a[0,0]
    block_src = original_values['a[0,0]']
    block_dst = tensor[0, 1].item()
    
    # tensor.getval/setval: a[0,101] should equal original a[0,100]
    tensor_src = original_values['a[0,100]']
    tensor_dst = tensor[0, 101].item()
    
    print(f"\n[{tag}] block: src a[0,0]={block_src}, dst a[0,1]={block_dst}")
    print(f"[{tag}] tensor: src a[0,100]={tensor_src}, dst a[0,101]={tensor_dst}")
    
    block_diff = abs(block_src - block_dst)
    tensor_diff = abs(tensor_src - tensor_dst)
    
    assert block_diff < 1e-3, f"[{tag}] block getval/setval failed: src={block_src}, dst={block_dst}, diff={block_diff}"
    assert tensor_diff < 1e-3, f"[{tag}] tensor getval/setval failed: src={tensor_src}, dst={tensor_dst}, diff={tensor_diff}"
    print(f"[{tag}] PASS (block_diff={block_diff:.6f}, tensor_diff={tensor_diff:.6f})")


@fe.jit()
def test_tile_getval_setval():
    device = "npu:7"
    torch.npu.set_device(device)

    shape = [64, 128]
    torch.manual_seed(0)
    dtype = torch.float16
    
    print("=" * 60)
    print("Test: pl.block.getval/setval + pl.tensor.getval/setval")
    print("=" * 60)

    # --- CCE backend ---
    print("\n--- Testing CCE backend ---")
    torch.manual_seed(0)
    a_cce = torch.rand(shape, device=device, dtype=dtype)
    original_cce = {
        'a[0,0]': a_cce[0, 0].item(),
        'a[0,100]': a_cce[0, 100].item()
    }
    print(f"Original values: a[0,0]={original_cce['a[0,0]']}, a[0,100]={original_cce['a[0,100]']}")
    
    compiled_cce = fe.compile(tile_getval_setval_kernel, arch="a3", codegen_mode="cce")
    print(f"CCE compiled lib: {compiled_cce.lib_path}")
    fe.launch(None, 1, compiled_cce, a_cce)
    torch.npu.synchronize()
    _verify_outputs("CCE", a_cce, original_cce)

    # --- PTO backend ---
    print("\n--- Testing PTO backend ---")
    torch.manual_seed(0)
    a_pto = torch.rand(shape, device=device, dtype=dtype)
    original_pto = {
        'a[0,0]': a_pto[0, 0].item(),
        'a[0,100]': a_pto[0, 100].item()
    }
    print(f"Original values: a[0,0]={original_pto['a[0,0]']}, a[0,100]={original_pto['a[0,100]']}")
    
    compiled_pto = fe.compile(tile_getval_setval_kernel, arch="a3", codegen_mode="pto")
    print(f"PTO compiled lib: {compiled_pto.lib_path}")
    fe.launch(None, 1, compiled_pto, a_pto)
    torch.npu.synchronize()
    _verify_outputs("PTO", a_pto, original_pto)


if __name__ == "__main__":
    test_tile_getval_setval()
    print("\n" + "=" * 60)
    print("All getval/setval tests passed!")
    print("=" * 60)
