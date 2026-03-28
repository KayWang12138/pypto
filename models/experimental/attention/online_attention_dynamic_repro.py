#!/usr/bin/env python3
# coding: utf-8
"""
Minimal repro for dynamic online-softmax attention compiler issues in PyPTO.

Repro intent:
- Dynamic batch axis
- Dynamic sequence axis
- Online softmax recurrence over KV tiles
"""

import argparse
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
def repro_online_softmax_kernel(
    q: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    tile_b: int,
    tile_q: int,
    tile_k: int,
):
    bs = q.shape[0]
    sq = q.shape[2]
    sk = k.shape[2]
    scale = 1.0 / (HEAD_DIM ** 0.5)
    b_loop = (bs + tile_b - 1) // tile_b
    k_loop = (sk + tile_k - 1) // tile_k

    for b_idx in pypto.loop(b_loop):
        b0 = b_idx * tile_b
        b1 = pypto.min(b0 + tile_b, bs)
        valid_b = b1 - b0
        q_view = pypto.view(
            q,
            [tile_b, NUM_HEADS, tile_q, HEAD_DIM],
            [b0, 0, 0, 0],
            valid_shape=[valid_b, NUM_HEADS, sq, HEAD_DIM],
        )

        # First K block init
        k0 = 0
        k1 = pypto.min(tile_k, sk)
        valid_k = k1 - k0
        k_view = pypto.view(
            k,
            [tile_b, NUM_HEADS, tile_k, HEAD_DIM],
            [b0, 0, k0, 0],
            valid_shape=[valid_b, NUM_HEADS, valid_k, HEAD_DIM],
        )
        v_view = pypto.view(
            v,
            [tile_b, NUM_HEADS, tile_k, HEAD_DIM],
            [b0, 0, k0, 0],
            valid_shape=[valid_b, NUM_HEADS, valid_k, HEAD_DIM],
        )
        pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
        scores = pypto.matmul(q_view, k_view, out_dtype=pypto.DT_FP32, a_trans=False, b_trans=True)
        pypto.set_vec_tile_shapes(1, NUM_HEADS, tile_q, tile_k)
        scores = pypto.mul(scores, scale)
        scores = pypto.view(
            scores,
            [tile_b, NUM_HEADS, tile_q, tile_k],
            [0, 0, 0, 0],
            valid_shape=[valid_b, NUM_HEADS, sq, valid_k],
        )
        local_max = pypto.amax(scores, dim=-1, keepdim=True)
        p = pypto.exp(pypto.sub(scores, local_max))
        local_sum = pypto.sum(p, dim=-1, keepdim=True)
        p_cast = pypto.cast(p, pypto.DT_FP32)
        o_acc = pypto.matmul(p_cast, v_view, out_dtype=pypto.DT_FP32)
        l_acc = local_sum
        m_acc = local_max

        # Remaining K blocks update
        for k_idx in pypto.loop(k_loop - 1):
            k0 = (k_idx + 1) * tile_k
            k1 = pypto.min(k0 + tile_k, sk)
            valid_k = k1 - k0
            k_view = pypto.view(
                k,
                [tile_b, NUM_HEADS, tile_k, HEAD_DIM],
                [b0, 0, k0, 0],
                valid_shape=[valid_b, NUM_HEADS, valid_k, HEAD_DIM],
            )
            v_view = pypto.view(
                v,
                [tile_b, NUM_HEADS, tile_k, HEAD_DIM],
                [b0, 0, k0, 0],
                valid_shape=[valid_b, NUM_HEADS, valid_k, HEAD_DIM],
            )
            pypto.set_cube_tile_shapes([64, 64], [64, 64], [64, 64])
            scores = pypto.matmul(q_view, k_view, out_dtype=pypto.DT_FP32, a_trans=False, b_trans=True)
            pypto.set_vec_tile_shapes(1, NUM_HEADS, tile_q, tile_k)
            scores = pypto.mul(scores, scale)
            scores = pypto.view(
                scores,
                [tile_b, NUM_HEADS, tile_q, tile_k],
                [0, 0, 0, 0],
                valid_shape=[valid_b, NUM_HEADS, sq, valid_k],
            )
            local_max = pypto.amax(scores, dim=-1, keepdim=True)
            p = pypto.exp(pypto.sub(scores, local_max))
            local_sum = pypto.sum(p, dim=-1, keepdim=True)
            p_cast = pypto.cast(p, pypto.DT_FP32)
            oi_tmp = pypto.matmul(p_cast, v_view, out_dtype=pypto.DT_FP32)

            pypto.set_vec_tile_shapes(1, NUM_HEADS, tile_q, 1)
            m_new = pypto.maximum(m_acc, local_max)
            a_old = pypto.exp(pypto.sub(m_acc, m_new))
            a_new = pypto.exp(pypto.sub(local_max, m_new))
            l_acc = pypto.add(pypto.mul(l_acc, a_old), pypto.mul(local_sum, a_new))
            pypto.set_vec_tile_shapes(1, NUM_HEADS, tile_q, HEAD_DIM)
            o_acc = pypto.add(pypto.mul(o_acc, a_old), pypto.mul(oi_tmp, a_new))
            m_acc = m_new

        pypto.set_vec_tile_shapes(1, NUM_HEADS, tile_q, HEAD_DIM)
        pypto.assemble(pypto.div(o_acc, l_acc), [b0, 0, 0, 0], out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--tile_b", type=int, default=2)
    parser.add_argument("--tile_q", type=int, default=64)
    parser.add_argument("--tile_k", type=int, default=32)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    device_id = _get_device_id()
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)

    q = torch.randn(args.batch, NUM_HEADS, args.seq, HEAD_DIM, dtype=torch.float32, device=device)
    k = torch.randn(args.batch, NUM_HEADS, args.seq, HEAD_DIM, dtype=torch.float32, device=device)
    v = torch.randn(args.batch, NUM_HEADS, args.seq, HEAD_DIM, dtype=torch.float32, device=device)
    out = torch.zeros(args.batch, NUM_HEADS, args.seq, HEAD_DIM, dtype=torch.float32, device=device)
    repro_online_softmax_kernel(q, k, v, out, args.tile_b, args.tile_q, args.tile_k)
    torch.npu.synchronize()
    print("repro kernel launch finished")


if __name__ == "__main__":
    main()

