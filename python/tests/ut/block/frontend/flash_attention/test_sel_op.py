"""Test sel operation with correct parameter order.

This test verifies the correct usage of the sel operation based on PTOAS definition:
    sel(out, mask, lhs, rhs, tmp)
    
Where:
    - out: Output tile
    - mask: Predicate mask tile (INT8 type, bit-packed)
    - lhs: Value selected when mask bit is 1
    - rhs: Value selected when mask bit is 0
    - tmp: Temporary workspace tile
    
The operation performs: out[i] = lhs[i] if mask_bit[i] else rhs[i]

IMPORTANT: Mask is bit-packed!
    - Each byte contains 8 bits for 8 consecutive elements
    - bit k in byte j corresponds to element at column (j*8 + k)
    - bit=1: select lhs, bit=0: select rhs
    
Example:
    mask[0, 0] = 0b00000001  # bit 0 = 1, bits 1-7 = 0
    # This means: col 0 -> lhs, cols 1-7 -> rhs
    
This test uses the same parameter order as in test_fa_BNSD_causal.py,
which is the CORRECT usage pattern.
"""

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm

NEG_INF = -1e9


@fe.kernel
def test_sel_correct_kernel(
    data: pl.Tensor[[64, 128], pl.FP32],
    mask: pl.Tensor[[64, 128], pl.INT8],
    result: pl.Tensor[[64, 128], pl.FP32],
) -> pl.Tensor[[64, 128], pl.FP32]:
    data_tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    mask_tile_type = plm.TileType(shape=[64, 128], dtype=pl.INT8, target_memory=pl.MemorySpace.Vec)
    
    data_tile = plm.make_tile(data_tile_type, addr=0x0000, size=32768)
    mask_tile = plm.make_tile(mask_tile_type, addr=0x8000, size=8192)  # int8 = 1字节 → 64*128=8192
    result_tile = plm.make_tile(data_tile_type, addr=0x10000, size=32768)
    neg_inf_tile = plm.make_tile(data_tile_type, addr=0x18000, size=32768)
    tmp_tile = plm.make_tile(data_tile_type, addr=0x20000, size=32768)

    with pl.section_vector():
        plm.load_tile(data_tile, data, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        
        plm.load_tile(mask_tile, mask, [0, 0])
        pl.system.bar_v()
        
        plm.expands(neg_inf_tile, NEG_INF)
        pl.system.bar_v()
             
        plm.sel(result_tile, mask_tile, data_tile, neg_inf_tile, tmp_tile)
        pl.system.bar_v()

        
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store_tile(result, result_tile, [0, 0])

    return result


def test_sel_correct():
    compiled = fe.compile(test_sel_correct_kernel, arch="a3")
    print("compiled:", compiled.lib_path)
    
    device = "npu:5"
    torch.npu.set_device(device)
    
    data = torch.ones((64, 128), device=device, dtype=torch.float32) * 100.0
    mask = torch.zeros((64, 128), device=device, dtype=torch.int8)
    
    # Mask is bit-packed: each byte contains 8 bits for 8 elements
    # bit=1: select lhs (data), bit=0: select rhs (neg_inf)
    # To set bit k in byte j: mask[i, j] |= (1 << k)
    
    # Row 0: [1, 0, 0, 0, ...] -> byte 0 = 0b00000001
    mask[0, 0] = 0b00000001  # bit 0 = 1, others = 0
    
    # Row 1: [1, 1, 0, 0, ...] -> byte 0 = 0b00000011
    mask[1, 0] = 0b00000011  # bit 0,1 = 1, others = 0
    
    # Row 2: [1, 1, 1, 0, ...] -> byte 0 = 0b00000111
    mask[2, 0] = 0b00000111  # bit 0,1,2 = 1, others = 0
    
    # Row 3: [0, 0, 0, 0, ...] -> byte 0 = 0b00000000
    mask[3, 0] = 0b00000000  # all bits = 0
    
    result = torch.zeros((64, 128), device=device, dtype=torch.float32)
    
    fe.launch(None, 1, compiled, data, mask, result)
    torch.npu.synchronize()
    
    print("\n=== Input Data (first 4x4) ===")
    print(data[:4, :4])
    
    print("\n=== Mask (first 4x8, bit-packed) ===")
    for i in range(4):
        bits = []
        for j in range(8):
            bit = (mask[i, 0].item() >> j) & 1
            bits.append(bit)
        print(f"Row {i}: byte={mask[i, 0].item():3d} (0b{mask[i, 0].item():08b}) -> bits {bits}")
    
    print("\n=== Result (first 4x8) ===")
    print(result[:4, :8])
    
    print("\n=== Expected (first 4x8) ===")
    expected = data.clone()
    # Calculate expected based on bit-packed mask
    for i in range(4):
        for j in range(8):
            bit = (mask[i, 0].item() >> j) & 1
            if bit == 0:
                expected[i, j] = NEG_INF
    print(expected[:4, :8])
    
    print("\n=== Verification ===")
    if torch.allclose(result[:4, :8], expected[:4, :8], rtol=1e-3, atol=1e-3):
        print("✓ sel operation works correctly!")
        print("  Parameter order: sel(out, mask, lhs, rhs, tmp)")
        print("  - mask bit=1: select lhs[i] (data)")
        print("  - mask bit=0: select rhs[i] (-1e9)")
        print("  Mask is bit-packed: each byte contains 8 bits for 8 elements")
    else:
        print("✗ sel operation FAILED!")
        diff = (result[:4, :8] - expected[:4, :8]).abs().max().item()
        print(f"Max difference: {diff}")


if __name__ == "__main__":
    print("=" * 80)
    print("Test sel operation with CORRECT parameter order")
    print("=" * 80)
    print("\nCorrect usage: sel(out, mask, lhs, rhs, tmp)")
    print("  - out: output tile")
    print("  - mask: predicate (INT8, bit-packed)")
    print("  - lhs: value when mask bit is 1")
    print("  - rhs: value when mask bit is 0")
    print("  - tmp: temporary workspace")
    print("\nIMPORTANT: Mask is bit-packed!")
    print("  Each byte contains 8 bits for 8 consecutive elements")
    print("  bit=1: select lhs, bit=0: select rhs")
    print("\nThis matches the usage in test_fa_BNSD_causal.py")
    print("=" * 80)
    test_sel_correct()