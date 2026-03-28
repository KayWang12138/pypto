#!/usr/bin/env python3
# coding: utf-8
"""Minimal repro for dynamic loop index codegen issue (VALUE_loop_idx_3)."""

import os

import pypto
import torch


NUM_HEADS = 8
HEAD_DIM = 64


def _get_device_id() -> int:
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        raise RuntimeError("Please set TILE_FWK_DEVICE_ID")
    return int(os.environ["TILE_FWK_DEVICE_ID"])


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def loop_index_codegen_kernel(
    k: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    tile_b: int,
    tile_k: int,
):
    bs = k.shape[0]
    sk = k.shape[2]
    b_loop = (bs + tile_b - 1) // tile_b
    k_loop = (sk + tile_k - 1) // tile_k

    for b_idx in pypto.loop(b_loop):
        b0 = b_idx * tile_b
        b1 = pypto.min(b0 + tile_b, bs)
        valid_b = b1 - b0

        for k_idx in pypto.loop(k_loop - 1):
            # This (k_idx + 1) pattern on dynamic views is known to be problematic.
            k0 = (k_idx + 1) * tile_k
            k1 = pypto.min(k0 + tile_k, sk)
            valid_k = k1 - k0
            kv = pypto.view(
                k,
                [tile_b, NUM_HEADS, tile_k, HEAD_DIM],
                [b0, 0, k0, 0],
                valid_shape=[valid_b, NUM_HEADS, valid_k, HEAD_DIM],
            )
            out_view = pypto.view(
                out,
                [tile_b, NUM_HEADS, tile_k, HEAD_DIM],
                [b0, 0, k0, 0],
                valid_shape=[valid_b, NUM_HEADS, valid_k, HEAD_DIM],
            )
            pypto.set_vec_tile_shapes(tile_b, NUM_HEADS, tile_k, HEAD_DIM)
            out_view[:] = kv


def main() -> None:
    import torch_npu  # noqa: F401

    device_id = _get_device_id()
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    batch = 2
    seq = 64
    tile_b = 2
    tile_k = 32
    k = torch.randn(batch, NUM_HEADS, seq, HEAD_DIM, dtype=torch.float32, device=device)
    out = torch.zeros(batch, NUM_HEADS, seq, HEAD_DIM, dtype=torch.float32, device=device)
    loop_index_codegen_kernel(k, out, tile_b, tile_k)
    torch.npu.synchronize()
    print("loop-index repro kernel launch finished")


if __name__ == "__main__":
    main()

