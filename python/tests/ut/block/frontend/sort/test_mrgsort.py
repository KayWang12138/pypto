# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for plm.mrgsort and plm.mrgsort2 functions.

mrgsort (format1): Merge sorted blocks within a single tile.
mrgsort2 (format2): Merge multiple pre-sorted tiles into one.

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
        print(f"Block {i}: vals={a[0, start:start+4].tolist()} idx={list(range(start, start+4))}...")

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
        print(f"  [ref] vals[:16]   : {expected_sorted[0, :16].tolist()}")
        print(f"  [ref] idx[:16]    : {expected_indices[0, :16].tolist()}")
        
        sorted_diff = torch.abs(values - expected_sorted).max().item()
        print(f"\nMax diff ({tag} vs ref): {sorted_diff}")
        
        assert sorted_diff < 1e-3, f"{tag} mrgsort failed: max diff {sorted_diff}"
        print(f"{tag} mrgsort test passed!")


# ============================================================================
# mrgsort2 (format2) tests
# ============================================================================

@fe.kernel
def mrgsort2_kernel(
    src0_tensor: pl.Tensor[[1, 256], pl.FP32],
    src1_tensor: pl.Tensor[[1, 256], pl.FP32],
    sorted_out: pl.Tensor[[1, 256], pl.FP32],
) -> pl.Tensor[[1, 256], pl.FP32]:
    """Merge two pre-sorted tiles using mrgsort2 (format2).

    Input tiles contain [val, idx] pairs (2 FP32 per pair): 128 pairs = 256 FP32.
    Output will be merged in descending order.
    """
    pl.system.bar_all()
    
    # src0: 128 pairs = 256 FP32
    tile_src0 = plm.make_tile(
        plm.TileType(shape=[1, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=1024
    )
    # src1: 128 pairs = 256 FP32
    tile_src1 = plm.make_tile(
        plm.TileType(shape=[1, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0400, size=1024
    )
    # dst: 128 pairs = 256 FP32
    tile_dst = plm.make_tile(
        plm.TileType(shape=[1, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0800, size=1024
    )
    # tmp: same shape as dst (format2: tmp shape must equal dst shape)
    tile_tmp = plm.make_tile(
        plm.TileType(shape=[1, 256], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0C00, size=1024
    )
    
    with pl.section_vector():
        plm.load(tile_src0, src0_tensor, [0, 0])
        plm.load(tile_src1, src1_tensor, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, exhausted=False)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(sorted_out, tile_dst, [0, 0])
    
    return sorted_out


def test_mrgsort2_2src():
    """Test mrgsort2 (format2) merging two pre-sorted lists."""
    device = "npu:7"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    dtype = torch.float32
    
    num_pairs_src0 = 128
    num_pairs_src1 = 128
    
    # Create two sorted lists
    vals0 = torch.rand([num_pairs_src0], device=device, dtype=dtype)
    vals1 = torch.rand([num_pairs_src1], device=device, dtype=dtype)
    
    vals0_sorted, idx0 = torch.sort(vals0, descending=True)
    vals1_sorted, idx1 = torch.sort(vals1, descending=True)
    
    print("\n=== mrgsort2 (format2) 2-source test ===")
    print(f"src0: {num_pairs_src0} pairs, top vals: {vals0_sorted[:4].tolist()} idx: {idx0[:4].tolist()}")
    print(f"src1: {num_pairs_src1} pairs, top vals: {vals1_sorted[:4].tolist()} idx: {(idx1[:4] + num_pairs_src0).tolist()}")
    
    # Pack into [val, idx] pairs (FP32 format: val, idx)
    src0_pairs = torch.zeros([num_pairs_src0 * 2], device=device, dtype=dtype)
    src1_pairs = torch.zeros([num_pairs_src1 * 2], device=device, dtype=dtype)
    
    for i in range(num_pairs_src0):
        src0_pairs[i*2] = vals0_sorted[i]
        src0_pairs[i*2 + 1] = idx0[i].float()                          # indices 0..127

    for i in range(num_pairs_src1):
        src1_pairs[i*2] = vals1_sorted[i]
        src1_pairs[i*2 + 1] = (idx1[i] + num_pairs_src0).float()       # indices 128..255

    src0_tensor = src0_pairs.unsqueeze(0)
    src1_tensor = src1_pairs.unsqueeze(0)

    # Expected: merge both sorted lists
    all_vals = torch.cat([vals0_sorted, vals1_sorted])
    all_indices = torch.cat([idx0.float(), idx1.float() + num_pairs_src0])
    expected_vals, sort_order = torch.sort(all_vals, descending=True)
    expected_indices = all_indices[sort_order]
    
    print(f"\nExpected merged top 16 values: {expected_vals[:16].tolist()}")
    print(f"Expected merged top 16 indices: {expected_indices[:16].tolist()}")
    
    sorted_out = torch.zeros([1, num_pairs_src0 * 2], device=device, dtype=dtype)
    
    compiled_cce = fe.compile(mrgsort2_kernel, arch="a3", codegen_mode="cce")
    compiled_pto = fe.compile(mrgsort2_kernel, arch="a3", codegen_mode="pto")
    
    for tag, lib in [("CCE", compiled_cce), ("PTO", compiled_pto)]:
        print(f"\n--- {tag} mode ---")
        print("compiled lib path:", lib.lib_path)
        
        sorted_out.zero_()
        fe.launch(None, 1, lib, src0_tensor, src1_tensor, sorted_out)
        torch.npu.synchronize()
        
        print(f"\n***********{tag} npu output***********")
        
        output_vals = sorted_out[0, 0::2]
        output_indices = sorted_out[0, 1::2]
        print(f"  [{tag}] vals[:16] : {output_vals[:16].tolist()}")
        print(f"  [{tag}] idx[:16]  : {output_indices[:16].tolist()}")
        print(f"  [ref] vals[:16]   : {expected_vals[:16].tolist()}")
        print(f"  [ref] idx[:16]    : {expected_indices[:16].tolist()}")
        
        # Compare with expected (only check top num_pairs_src0 pairs)
        vals_diff = torch.abs(output_vals[:num_pairs_src0] - expected_vals[:num_pairs_src0]).max().item()
        idx_diff = torch.abs(output_indices[:num_pairs_src0] - expected_indices[:num_pairs_src0]).max().item()
        print(f"\nMax val diff ({tag} vs ref): {vals_diff}")
        print(f"Max idx diff ({tag} vs ref): {idx_diff}")

        assert vals_diff < 1e-3, f"{tag} mrgsort2_2src failed: val max diff {vals_diff}"
        assert idx_diff < 1e-3, f"{tag} mrgsort2_2src failed: idx max diff {idx_diff}"
        print(f"{tag} mrgsort2_2src test passed!")


# ============================================================================
# mrgsort2 (format2) - 3 source test
# ============================================================================

@fe.kernel
def mrgsort2_3src_kernel(
    src0_tensor: pl.Tensor[[1, 128], pl.FP32],
    src1_tensor: pl.Tensor[[1, 128], pl.FP32],
    src2_tensor: pl.Tensor[[1, 128], pl.FP32],
    sorted_out: pl.Tensor[[1, 128], pl.FP32],
) -> pl.Tensor[[1, 128], pl.FP32]:
    """Merge three pre-sorted tiles using mrgsort2 (format2).

    Each input tile contains 64 [val, idx] pairs (128 FP32).
    Output will contain top-64 merged results (128 FP32).
    """
    pl.system.bar_all()

    tile_src0 = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=512
    )
    tile_src1 = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0200, size=512
    )
    tile_src2 = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0400, size=512
    )
    tile_dst = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0600, size=512
    )
    tile_tmp = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0800, size=512
    )

    with pl.section_vector():
        plm.load(tile_src0, src0_tensor, [0, 0])
        plm.load(tile_src1, src1_tensor, [0, 0])
        plm.load(tile_src2, src2_tensor, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp,
                     tile_src2, exhausted=False)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(sorted_out, tile_dst, [0, 0])

    return sorted_out


def test_mrgsort2_3src():
    """Test mrgsort2 (format2) merging three pre-sorted lists."""
    device = "npu:7"
    torch.npu.set_device(device)
    torch.manual_seed(7)
    dtype = torch.float32

    num_pairs_per_src = 64

    vals0 = torch.rand([num_pairs_per_src], device=device, dtype=dtype)
    vals1 = torch.rand([num_pairs_per_src], device=device, dtype=dtype)
    vals2 = torch.rand([num_pairs_per_src], device=device, dtype=dtype)

    vals0_sorted, idx0 = torch.sort(vals0, descending=True)
    vals1_sorted, idx1 = torch.sort(vals1, descending=True)
    vals2_sorted, idx2 = torch.sort(vals2, descending=True)

    print("\n=== mrgsort2 (format2) 3-source test ===")
    print(f"src0: {num_pairs_per_src} pairs, top vals: {vals0_sorted[:4].tolist()} idx: {idx0[:4].tolist()}")
    print(f"src1: {num_pairs_per_src} pairs, top vals: {vals1_sorted[:4].tolist()} idx: {(idx1[:4] + num_pairs_per_src).tolist()}")
    print(f"src2: {num_pairs_per_src} pairs, top vals: {vals2_sorted[:4].tolist()} idx: {(idx2[:4] + 2 * num_pairs_per_src).tolist()}")

    def pack_pairs(vals_sorted, indices, offset=0):
        pairs = torch.zeros([len(vals_sorted) * 2], device=device, dtype=dtype)
        for i in range(len(vals_sorted)):
            pairs[i*2] = vals_sorted[i]
            pairs[i*2 + 1] = (indices[i] + offset).float()
        return pairs

    src0_pairs = pack_pairs(vals0_sorted, idx0, 0)
    src1_pairs = pack_pairs(vals1_sorted, idx1, num_pairs_per_src)
    src2_pairs = pack_pairs(vals2_sorted, idx2, 2 * num_pairs_per_src)

    src0_tensor = src0_pairs.unsqueeze(0)
    src1_tensor = src1_pairs.unsqueeze(0)
    src2_tensor = src2_pairs.unsqueeze(0)

    all_vals = torch.cat([vals0_sorted, vals1_sorted, vals2_sorted])
    all_indices = torch.cat([
        idx0.float(),
        idx1.float() + num_pairs_per_src,
        idx2.float() + 2 * num_pairs_per_src,
    ])
    expected_vals, sort_order = torch.sort(all_vals, descending=True)
    expected_indices = all_indices[sort_order]

    print(f"\nExpected merged top 16 values: {expected_vals[:16].tolist()}")
    print(f"Expected merged top 16 indices: {expected_indices[:16].tolist()}")

    sorted_out = torch.zeros([1, num_pairs_per_src * 2], device=device, dtype=dtype)

    compiled_cce = fe.compile(mrgsort2_3src_kernel, arch="a3", codegen_mode="cce")
    compiled_pto = fe.compile(mrgsort2_3src_kernel, arch="a3", codegen_mode="pto")

    for tag, lib in [("CCE", compiled_cce), ("PTO", compiled_pto)]:
        print(f"\n--- {tag} mode ---")
        print("compiled lib path:", lib.lib_path)

        sorted_out.zero_()
        fe.launch(None, 1, lib, src0_tensor, src1_tensor, src2_tensor, sorted_out)
        torch.npu.synchronize()

        print(f"\n***********{tag} npu output***********")

        output_vals = sorted_out[0, 0::2]
        output_indices = sorted_out[0, 1::2]
        print(f"  [{tag}] vals[:16] : {output_vals[:16].tolist()}")
        print(f"  [{tag}] idx[:16]  : {output_indices[:16].tolist()}")
        print(f"  [ref] vals[:16]   : {expected_vals[:16].tolist()}")
        print(f"  [ref] idx[:16]    : {expected_indices[:16].tolist()}")

        vals_diff = torch.abs(output_vals[:num_pairs_per_src] - expected_vals[:num_pairs_per_src]).max().item()
        idx_diff = torch.abs(output_indices[:num_pairs_per_src] - expected_indices[:num_pairs_per_src]).max().item()
        print(f"\nMax val diff ({tag} vs ref): {vals_diff}")
        print(f"Max idx diff ({tag} vs ref): {idx_diff}")

        assert vals_diff < 1e-3, f"{tag} mrgsort2_3src failed: val max diff {vals_diff}"
        assert idx_diff < 1e-3, f"{tag} mrgsort2_3src failed: idx max diff {idx_diff}"
        print(f"{tag} mrgsort2_3src test passed!")


# ============================================================================
# mrgsort2 (format2) - 4 source test
# ============================================================================

@fe.kernel
def mrgsort2_4src_kernel(
    src0_tensor: pl.Tensor[[1, 128], pl.FP32],
    src1_tensor: pl.Tensor[[1, 128], pl.FP32],
    src2_tensor: pl.Tensor[[1, 128], pl.FP32],
    src3_tensor: pl.Tensor[[1, 128], pl.FP32],
    sorted_out: pl.Tensor[[1, 128], pl.FP32],
) -> pl.Tensor[[1, 128], pl.FP32]:
    """Merge four pre-sorted tiles using mrgsort2 (format2).

    Each input tile contains 64 [val, idx] pairs (128 FP32).
    Output will contain top-64 merged results (128 FP32).
    """
    pl.system.bar_all()
    
    # Each tile: 64 pairs = 128 FP32
    tile_src0 = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0000, size=512
    )
    tile_src1 = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0200, size=512
    )
    tile_src2 = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0400, size=512
    )
    tile_src3 = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0600, size=512
    )
    # dst: 64 pairs = 128 FP32
    tile_dst = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0800, size=512
    )
    # tmp: same shape as dst (format2: tmp shape must equal dst shape)
    tile_tmp = plm.make_tile(
        plm.TileType(shape=[1, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
        addr=0x0A00, size=512
    )
    
    with pl.section_vector():
        plm.load(tile_src0, src0_tensor, [0, 0])
        plm.load(tile_src1, src1_tensor, [0, 0])
        plm.load(tile_src2, src2_tensor, [0, 0])
        plm.load(tile_src3, src3_tensor, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp,
                     tile_src2, tile_src3, exhausted=False)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(sorted_out, tile_dst, [0, 0])
    
    return sorted_out


def test_mrgsort2_4src():
    """Test mrgsort2 (format2) merging four pre-sorted lists."""
    device = "npu:7"
    torch.npu.set_device(device)
    torch.manual_seed(123)
    dtype = torch.float32
    
    num_pairs_per_src = 64
    
    # Create four sorted lists
    vals0 = torch.rand([num_pairs_per_src], device=device, dtype=dtype)
    vals1 = torch.rand([num_pairs_per_src], device=device, dtype=dtype)
    vals2 = torch.rand([num_pairs_per_src], device=device, dtype=dtype)
    vals3 = torch.rand([num_pairs_per_src], device=device, dtype=dtype)
    
    vals0_sorted, idx0 = torch.sort(vals0, descending=True)
    vals1_sorted, idx1 = torch.sort(vals1, descending=True)
    vals2_sorted, idx2 = torch.sort(vals2, descending=True)
    vals3_sorted, idx3 = torch.sort(vals3, descending=True)
    
    print("\n=== mrgsort2 (format2) 4-source test ===")
    print(f"src0: {num_pairs_per_src} pairs, top vals: {vals0_sorted[:4].tolist()} idx: {idx0[:4].tolist()}")
    print(f"src1: {num_pairs_per_src} pairs, top vals: {vals1_sorted[:4].tolist()} idx: {(idx1[:4] + num_pairs_per_src).tolist()}")
    print(f"src2: {num_pairs_per_src} pairs, top vals: {vals2_sorted[:4].tolist()} idx: {(idx2[:4] + 2 * num_pairs_per_src).tolist()}")
    print(f"src3: {num_pairs_per_src} pairs, top vals: {vals3_sorted[:4].tolist()} idx: {(idx3[:4] + 3 * num_pairs_per_src).tolist()}")
    
    # Pack into [val, idx] pairs (FP32 format: val, idx)
    def pack_pairs(vals_sorted, indices, offset=0):
        pairs = torch.zeros([len(vals_sorted) * 2], device=device, dtype=dtype)
        for i in range(len(vals_sorted)):
            pairs[i*2] = vals_sorted[i]
            pairs[i*2 + 1] = (indices[i] + offset).float()
        return pairs
    
    src0_pairs = pack_pairs(vals0_sorted, idx0, 0)
    src1_pairs = pack_pairs(vals1_sorted, idx1, num_pairs_per_src)
    src2_pairs = pack_pairs(vals2_sorted, idx2, 2 * num_pairs_per_src)
    src3_pairs = pack_pairs(vals3_sorted, idx3, 3 * num_pairs_per_src)
    
    src0_tensor = src0_pairs.unsqueeze(0)
    src1_tensor = src1_pairs.unsqueeze(0)
    src2_tensor = src2_pairs.unsqueeze(0)
    src3_tensor = src3_pairs.unsqueeze(0)

    # Expected: merge all four sorted lists
    all_vals = torch.cat([vals0_sorted, vals1_sorted, vals2_sorted, vals3_sorted])
    all_indices = torch.cat([
        idx0.float(),
        idx1.float() + num_pairs_per_src,
        idx2.float() + 2 * num_pairs_per_src,
        idx3.float() + 3 * num_pairs_per_src
    ])
    expected_vals, sort_order = torch.sort(all_vals, descending=True)
    expected_indices = all_indices[sort_order]
    
    print(f"\nExpected merged top 16 values: {expected_vals[:16].tolist()}")
    print(f"Expected merged top 16 indices: {expected_indices[:16].tolist()}")
    
    sorted_out = torch.zeros([1, num_pairs_per_src * 2], device=device, dtype=dtype)
    
    compiled_cce = fe.compile(mrgsort2_4src_kernel, arch="a3", codegen_mode="cce")
    compiled_pto = fe.compile(mrgsort2_4src_kernel, arch="a3", codegen_mode="pto")
    
    for tag, lib in [("CCE", compiled_cce), ("PTO", compiled_pto)]:
        print(f"\n--- {tag} mode ---")
        print("compiled lib path:", lib.lib_path)
        
        sorted_out.zero_()
        fe.launch(None, 1, lib, src0_tensor, src1_tensor, src2_tensor, src3_tensor, sorted_out)
        torch.npu.synchronize()
        
        print(f"\n***********{tag} npu output***********")
        
        output_vals = sorted_out[0, 0::2]
        output_indices = sorted_out[0, 1::2]
        print(f"  [{tag}] vals[:16] : {output_vals[:16].tolist()}")
        print(f"  [{tag}] idx[:16]  : {output_indices[:16].tolist()}")
        print(f"  [ref] vals[:16]   : {expected_vals[:16].tolist()}")
        print(f"  [ref] idx[:16]    : {expected_indices[:16].tolist()}")
        
        # Compare with expected (only check top num_pairs_per_src pairs)
        vals_diff = torch.abs(output_vals[:num_pairs_per_src] - expected_vals[:num_pairs_per_src]).max().item()
        idx_diff = torch.abs(output_indices[:num_pairs_per_src] - expected_indices[:num_pairs_per_src]).max().item()
        print(f"\nMax val diff ({tag} vs ref): {vals_diff}")
        print(f"Max idx diff ({tag} vs ref): {idx_diff}")

        assert vals_diff < 1e-3, f"{tag} mrgsort2_4src failed: val max diff {vals_diff}"
        assert idx_diff < 1e-3, f"{tag} mrgsort2_4src failed: idx max diff {idx_diff}"
        print(f"{tag} mrgsort2_4src test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("mrgsort tests (format1 and format2)")
    print("=" * 60)
    test_mrgsort()
    test_mrgsort2_2src()
    test_mrgsort2_3src()
    test_mrgsort2_4src()
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)