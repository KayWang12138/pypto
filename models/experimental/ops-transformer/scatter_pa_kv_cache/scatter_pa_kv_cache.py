#!/usr/bin/env python3
# coding: utf-8
"""
ScatterPaKvCache Operator for PyPTO

This operator updates KV cache at specified positions.
Implements a simplified version focusing on scatter_update functionality.

Mathematical Formula:
    keyCache[slotMapping[i]] = key[i]  for all i
    valueCache[slotMapping[i]] = value[i]  for all i
"""

import os
import torch
import pypto


BATCH_SIZE = 2
SEQ_LEN = 4
NUM_HEADS = 1
HEAD_DIM = 64
NUM_BLOCKS = 16
BLOCK_SIZE = 16
DTYPE = pypto.DT_BF16


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set environment variable TILE_FWK_DEVICE_ID:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be integer")
        return None


def scatter_pa_kv_cache_golden(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple:
    """PyTorch reference implementation for ScatterPaKvCache.
    
    Args:
        key: [batch * seq_len, num_heads, head_dim]
        value: [batch * seq_len, num_heads, head_dim]
        key_cache: [num_blocks, block_size, num_heads, head_dim]
        value_cache: [num_blocks, block_size, num_heads, head_dim]
        slot_mapping: [batch * seq_len]
    
    Returns:
        Updated key_cache and value_cache
    """
    num_tokens = key.shape[0]
    
    for i in range(num_tokens):
        slot_idx = slot_mapping[i].item()
        block_idx = slot_idx // BLOCK_SIZE
        block_offset = slot_idx % BLOCK_SIZE
        key_cache[block_idx, block_offset, :, :] = key[i, :, :]
        value_cache[block_idx, block_offset, :, :] = value[i, :, :]
    
    return key_cache, value_cache


@pypto.frontend.jit
def scatter_pa_kv_cache_kernel(
    key: pypto.Tensor((BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM), DTYPE),
    value: pypto.Tensor((BATCH_SIZE * SEQ_LEN, NUM_HEADS, HEAD_DIM), DTYPE),
    key_cache: pypto.Tensor((NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM), DTYPE),
    value_cache: pypto.Tensor((NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM), DTYPE),
    slot_mapping: pypto.Tensor((BATCH_SIZE * SEQ_LEN,), pypto.DT_INT64),
    key_cache_out: pypto.Tensor((NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM), DTYPE),
    value_cache_out: pypto.Tensor((NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM), DTYPE),
):
    """ScatterPaKvCache kernel implementation.
    
    Updates key_cache and value_cache at positions specified by slot_mapping.
    Uses scatter_update for efficient cache update.
    """
    tile_bs = BATCH_SIZE
    
    key_4d = pypto.reshape(key, [tile_bs, SEQ_LEN, NUM_HEADS, HEAD_DIM])
    value_4d = pypto.reshape(value, [tile_bs, SEQ_LEN, NUM_HEADS, HEAD_DIM])
    slot_2d = pypto.reshape(slot_mapping, [tile_bs, SEQ_LEN])
    
    pypto.set_vec_tile_shapes(tile_bs, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    
    key_cache_out[:] = pypto.scatter_update(key_cache, -2, slot_2d, key_4d)
    value_cache_out[:] = pypto.scatter_update(value_cache, -2, slot_2d, value_4d)


def test_scatter_pa_kv_cache(device_id: int = None, run_mode: str = "npu"):
    """Test ScatterPaKvCache operator."""
    print("=" * 60)
    print("Test: ScatterPaKvCache")
    print("=" * 60)
    
    device = f'npu:{device_id}' if (run_mode == "npu" and device_id is not None) else 'cpu'
    
    num_tokens = BATCH_SIZE * SEQ_LEN
    
    key = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    key_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    value_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    
    slot_mapping = torch.randint(0, NUM_BLOCKS * BLOCK_SIZE, (num_tokens,), dtype=torch.int64, device=device)
    slot_mapping = torch.unique(slot_mapping)[:num_tokens]
    while len(slot_mapping) < num_tokens:
        remaining = num_tokens - len(slot_mapping)
        new_slots = torch.randint(0, NUM_BLOCKS * BLOCK_SIZE, (remaining * 2,), dtype=torch.int64, device=device)
        slot_mapping = torch.unique(torch.cat([slot_mapping, new_slots]))[:num_tokens]
    
    key_cache_out = key_cache.clone()
    value_cache_out = value_cache.clone()
    
    scatter_pa_kv_cache_kernel(key, value, key_cache, value_cache, slot_mapping, key_cache_out, value_cache_out)
    
    key_cache_golden = key_cache.clone()
    value_cache_golden = value_cache.clone()
    key_cache_golden, value_cache_golden = scatter_pa_kv_cache_golden(
        key, value, key_cache_golden, value_cache_golden, slot_mapping
    )
    
    print(f"Input key shape: {key.shape}")
    print(f"Input value shape: {value.shape}")
    print(f"Key cache shape: {key_cache.shape}")
    print(f"Value cache shape: {value_cache.shape}")
    print(f"Slot mapping shape: {slot_mapping.shape}")
    print(f"Slot mapping values: {slot_mapping.tolist()}")
    
    if run_mode == "npu":
        key_diff = (key_cache_out - key_cache_golden).abs().max().item()
        value_diff = (value_cache_out - value_cache_golden).abs().max().item()
        print(f"Key cache max diff: {key_diff:.6f}")
        print(f"Value cache max diff: {value_diff:.6f}")
        
        key_match = torch.allclose(key_cache_out, key_cache_golden, rtol=1e-3, atol=1e-3)
        value_match = torch.allclose(value_cache_out, value_cache_golden, rtol=1e-3, atol=1e-3)
        
        if key_match and value_match:
            print("✓ ScatterPaKvCache test passed")
        else:
            print("✗ ScatterPaKvCache test failed")
            print(f"  Key cache match: {key_match}")
            print(f"  Value cache match: {value_match}")
    else:
        print("✓ ScatterPaKvCache execution completed")
    
    print()
    return key_match and value_match if run_mode == "npu" else True


def main():
    """Main function to run ScatterPaKvCache test."""
    print("\n" + "=" * 60)
    print("PyPTO ScatterPaKvCache Operator Test")
    print("=" * 60 + "\n")
    
    device_id = get_device_id()
    if device_id is None:
        return
    
    import torch_npu
    torch.npu.set_device(device_id)
    print(f"Running on NPU device {device_id}\n")
    
    try:
        success = test_scatter_pa_kv_cache(device_id, "npu")
        if success:
            print("=" * 60)
            print("All tests passed!")
            print("=" * 60)
        else:
            print("=" * 60)
            print("Some tests failed!")
            print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()