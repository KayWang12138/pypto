# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for plm.sort32 function.

Tests sort32 with both aligned (no tmp) and non-aligned (with tmp) cases.
"""

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm


# ============================================================================
# sort32 without tmp (32-aligned case)
# ============================================================================

@fe.kernel
def sort32_kernel(
    a: pl.Tensor[[1, 32], pl.FP16],
    idx_in: pl.Tensor[[1, 32], pl.UINT32],
    sorted_out: pl.Tensor[[1, 128], pl.FP16],
) -> pl.Tensor[[1, 128], pl.FP16]:
    """Sort 32 elements using sort32 (no tmp).
    
    For FP16, dst has 4x columns of src because:
    - TYPE_COEF = sizeof(float)/sizeof(half) = 2
    - dst stores value-index pairs, so cols = src_cols * TYPE_COEF * 2
    - For 32 FP16 elements: dst_cols = 32 * 2 * 2 = 128
    """
    pl.system.bar_all()
    
    tile_src = plm.make_tile(
        plm.TileType(shape=[1, 32], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=64
    )
    tile_dst = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0040, size=256
    )
    tile_idx = plm.make_tile(
        plm.TileType(shape=[1, 32], dtype=pl.UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x0140, size=128
    )
    
    with pl.section_vector():
        plm.load(tile_src, a, [0, 0])
        plm.load(tile_idx, idx_in, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.sort32(tile_dst, tile_src, tile_idx)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(sorted_out, tile_dst, [0, 0])
    
    return sorted_out


@fe.jit()
def test_sort32():
    """Test sort32 (32-aligned, no tmp) with CCE and PTO backends."""
    device = "npu:7"
    torch.npu.set_device(device)

    torch.manual_seed(0)
    dtype = torch.float16
    
    print("\n=== sort32 test (32-aligned, no tmp) ===")
    
    a = torch.rand([1, 32], device=device, dtype=dtype)
    idx_in = torch.arange(32, device=device, dtype=torch.int32).unsqueeze(0)
    
    print("Input tensor:")
    print(f"  a[:8] = {a[0, :8].tolist()}")
    print(f"  idx_in[:8] = {idx_in[0, :8].tolist()}")
    
    expected_sorted, expected_idx = torch.sort(a, dim=1, descending=True)
    print("\nExpected sorted (descending):")
    print(f"  [ref] vals : {expected_sorted[0].tolist()}")
    print(f"  [ref] idx  : {expected_idx[0].tolist()}")

    compiled_cce = fe.compile(sort32_kernel, arch="a3", codegen_mode="cce")
    compiled_pto = fe.compile(sort32_kernel, arch="a3", codegen_mode="pto")
    
    for tag, lib in [("CCE", compiled_cce), ("PTO", compiled_pto)]:
        print(f"\n--- {tag} mode ---")
        print("compiled lib path:", lib.lib_path)
        
        sorted_out = torch.zeros([1, 128], device=device, dtype=dtype)
        fe.launch(None, 1, lib, a.clone(), idx_in.clone(), sorted_out)
        torch.npu.synchronize()
        
        print(f"\n***********{tag} npu output***********")
        
        values = sorted_out[0, 0::4]
        sorted_out_cpu = sorted_out.cpu()
        indices = torch.zeros(32, dtype=torch.int32)
        for i in range(32):
            idx_low = sorted_out_cpu[0, i*4 + 2].view(torch.uint16).item()
            idx_high = sorted_out_cpu[0, i*4 + 3].view(torch.uint16).item()
            indices[i] = idx_low | (idx_high << 16)
        
        print(f"  [{tag}] vals : {values.tolist()}")
        print(f"  [{tag}] idx  : {indices.tolist()}")
        print(f"  [ref] vals  : {expected_sorted[0].tolist()}")
        print(f"  [ref] idx   : {expected_idx[0].tolist()}")
        
        vals_diff = torch.abs(values - expected_sorted).max().item()
        print(f"\nMax diff ({tag} vs ref): {vals_diff}")
        
        assert vals_diff < 1e-3, f"{tag} sort32 failed: max diff {vals_diff}"
        print(f"{tag} sort32 test passed!")


# ============================================================================
# sort32 with tmp (tail handling, < 32 elements)
# ============================================================================

@fe.kernel
def sort32_tail_kernel(
    a: pl.Tensor[[1, 16], pl.FP16],
    idx_in: pl.Tensor[[1, 16], pl.UINT32],
    sorted_out: pl.Tensor[[1, 64], pl.FP16],
) -> pl.Tensor[[1, 64], pl.FP16]:
    """Sort 16 elements using sort32 with tmp buffer for tail handling.
    
    16 < 32, all elements are in tail block.
    Tail requires tmp buffer for padding to 32 elements.
    For FP16: dst_cols = 16 * 2 * 2 = 64
    
    Note: Tile alignment constraint for RowMajor+NoneBox:
    Cols * sizeof(dtype) % 32 == 0
    For FP16 (2 bytes): Cols must be multiple of 16.
    16 * 2 = 32 bytes ✓
    """
    pl.system.bar_all()
    
    # src: 16 FP16 elements (< 32, pure tail, aligned: 16*2=32 bytes)
    tile_src = plm.make_tile(
        plm.TileType(shape=[1, 16], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=32
    )
    # dst: 64 FP16 (16 pairs × 4, aligned: 64*2=128 bytes)
    tile_dst = plm.make_tile(
        plm.TileType(shape=[1, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0020, size=128
    )
    # idx: 16 UINT32 (aligned: 16*4=64 bytes)
    tile_idx = plm.make_tile(
        plm.TileType(shape=[1, 16], dtype=pl.UINT32, target_memory=pl.MemorySpace.Vec),
        addr=0x00A0, size=64
    )
    # tmp: scratch buffer for padding to 32
    tile_tmp = plm.make_tile(
        plm.TileType(shape=[1, 32], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0120, size=64
    )
    
    with pl.section_vector():
        plm.load(tile_src, a, [0, 0])
        plm.load(tile_idx, idx_in, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.sort32(tile_tmp, tile_src, tile_idx, tile_dst)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(sorted_out, tile_dst, [0, 0])
    
    return sorted_out


@fe.jit()
def test_sort32_tail():
    """Test sort32 with tmp for pure tail (16 elements < 32)."""
    device = "npu:7"
    torch.npu.set_device(device)

    torch.manual_seed(100)
    dtype = torch.float16
    num_elements = 16  # pure tail, < 32, aligned
    
    print("\n=== sort32 test (pure tail with tmp) ===")
    print(f"Testing {num_elements} elements (pure tail, < 32)")
    print("All elements need tmp buffer for padding to 32")
    print("Cols=16 satisfies alignment: 16*2=32 bytes % 32 == 0")
    
    a = torch.rand([1, num_elements], device=device, dtype=dtype)
    idx_in = torch.arange(num_elements, device=device, dtype=torch.int32).unsqueeze(0)
    
    print(f"\nInput tensor:")
    print(f"  a = {a[0].tolist()}")
    
    # Expected: sort all 16 elements descending
    expected_sorted, expected_idx = torch.sort(a, dim=1, descending=True)
    print(f"\nExpected sorted (descending):")
    print(f"  [ref] vals : {expected_sorted[0].tolist()}")
    print(f"  [ref] idx  : {expected_idx[0].tolist()}")

    compiled_cce = fe.compile(sort32_tail_kernel, arch="a3", codegen_mode="cce")
    compiled_pto = fe.compile(sort32_tail_kernel, arch="a3", codegen_mode="pto")
    
    for tag, lib in [("CCE", compiled_cce), ("PTO", compiled_pto)]:
        print(f"\n--- {tag} mode ---")
        print("compiled lib path:", lib.lib_path)
        
        sorted_out = torch.zeros([1, num_elements * 4], device=device, dtype=dtype)
        fe.launch(None, 1, lib, a.clone(), idx_in.clone(), sorted_out)
        torch.npu.synchronize()
        
        print(f"\n***********{tag} npu output***********")
        
        # Extract values and indices from interleaved format
        values = sorted_out[0, 0::4]
        sorted_out_cpu = sorted_out.cpu()
        indices = torch.zeros(num_elements, dtype=torch.int32)
        for i in range(num_elements):
            idx_low = sorted_out_cpu[0, i*4 + 2].view(torch.uint16).item()
            idx_high = sorted_out_cpu[0, i*4 + 3].view(torch.uint16).item()
            indices[i] = idx_low | (idx_high << 16)
        
        print(f"  [{tag}] vals : {values.tolist()}")
        print(f"  [{tag}] idx  : {indices.tolist()}")
        print(f"  [ref] vals  : {expected_sorted[0].tolist()}")
        print(f"  [ref] idx   : {expected_idx[0].tolist()}")
        
        vals_diff = torch.abs(values - expected_sorted).max().item()
        print(f"\nMax diff ({tag} vs ref): {vals_diff}")
        
        assert vals_diff < 1e-3, f"{tag} sort32_tail failed: max diff {vals_diff}"
        print(f"{tag} sort32_tail test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("sort32 tests")
    print("=" * 60)
    test_sort32()
    test_sort32_tail()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
