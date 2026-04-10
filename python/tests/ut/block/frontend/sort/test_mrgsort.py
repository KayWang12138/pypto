# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for plm.mrgsort functions.

mrgsort (format1): Merge sorted blocks within a single tile.

Both operations produce descending-order output in [val, idx] pair format.
"""

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm


# ============================================================================
# mrgsort (format1) tests
# ============================================================================

@fe.kernel
def mrgsort_kernel(
    a: pl.Tensor[[1, 1024], pl.FP16],
    sorted_out: pl.Tensor[[1, 1024], pl.FP16],
) -> pl.Tensor[[1, 1024], pl.FP16]:
    """Merge sort using mrgsort with blockLen=256.
    
    Input must be pre-sorted in blocks of 64 elements in value-index pair format.
    For FP16, each element occupies 4 FP16 values (value + pad + idx_low + idx_high).
    """
    pl.system.bar_all()
    
    tile_src = plm.make_tile(
        plm.TileType(shape=[1, 1024], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=2048
    )
    tile_dst = plm.make_tile(
        plm.TileType(shape=[1, 1024], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x0800, size=2048
    )
    
    with pl.section_vector():
        plm.load(tile_src, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.mrgsort(tile_dst, tile_src, block_len=256)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(sorted_out, tile_dst, [0, 0])
    
    return sorted_out


def test_mrgsort():
    """Test mrgsort (format1) with blockLen=256."""
    device = "npu:7"
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16
    
    block_len = 64
    num_blocks = 4
    total_elements = block_len * num_blocks
    
    a = torch.rand([1, total_elements], device=device, dtype=dtype)
    
    print("\n=== mrgsort (format1) test ===")
    print("Original data:")
    print("a[0, :16] =", a[0, :16])
    
    for i in range(num_blocks):
        start = i * block_len
        end = (i + 1) * block_len
        block = a[0, start:end]
        sorted_block, _ = torch.sort(block, descending=True)
        a[0, start:end] = sorted_block
    
    print("\nAfter sorting each block internally (descending):")
    for i in range(num_blocks):
        start = i * block_len
        end = (i + 1) * block_len
        print(f"Block {i}: {a[0, start:start+8]}...")
    
    input_pairs = torch.zeros([1, total_elements * 4], device=device, dtype=dtype)
    for i in range(total_elements):
        input_pairs[0, i*4] = a[0, i]
        idx_val = i
        input_pairs[0, i*4 + 2] = torch.tensor(idx_val & 0xFFFF, dtype=torch.uint16).view(dtype)
        input_pairs[0, i*4 + 3] = torch.tensor((idx_val >> 16) & 0xFFFF, dtype=torch.uint16).view(dtype)
    
    sorted_out = torch.zeros([1, total_elements * 4], device=device, dtype=dtype)
    
    expected_sorted, expected_indices = torch.sort(a, dim=1, descending=True)
    print("\nExpected fully sorted output (descending):")
    print(f"  [ref] vals[:16] : {expected_sorted[0, :16].tolist()}")
    print(f"  [ref] idx[:16]  : {expected_indices[0, :16].tolist()}")

    compiled_cce = fe.compile(mrgsort_kernel, arch="a3", codegen_mode="cce")
    compiled_pto = fe.compile(mrgsort_kernel, arch="a3", codegen_mode="pto")
    
    for tag, lib in [("CCE", compiled_cce), ("PTO", compiled_pto)]:
        print(f"\n--- {tag} mode ---")
        print("compiled lib path:", lib.lib_path)
        
        sorted_out.zero_()
        fe.launch(None, 1, lib, input_pairs, sorted_out)
        torch.npu.synchronize()
        
        print(f"\n***********{tag} npu output***********")
        print("sorted_out shape:", sorted_out.shape)
        
        values = sorted_out[0, 0::4]
        indices = sorted_out[0, 2::4].view(torch.uint16).to(torch.int32) | \
                  (sorted_out[0, 3::4].view(torch.uint16).to(torch.int32) << 16)
        print(f"  [{tag}] vals[:16] : {values[:16].tolist()}")
        print(f"  [{tag}] idx[:16]  : {indices[:16].tolist()}")
        
        sorted_diff = torch.abs(values - expected_sorted).max().item()
        print(f"\nMax diff ({tag} vs ref): {sorted_diff}")
        
        assert sorted_diff < 1e-3, f"{tag} mrgsort failed: max diff {sorted_diff}"
        print(f"{tag} mrgsort test passed!")

if __name__ == "__main__":
    print("mrgsort (format1) tests")
    print("=" * 60)
    test_mrgsort()
    print("\nAll tests passed!")