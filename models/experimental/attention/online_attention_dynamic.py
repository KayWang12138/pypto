#!/usr/bin/env python3
# coding: utf-8
"""
Dynamic-shape online softmax attention (no dropout) implemented with PyPTO.

Dynamic axes:
- batch axis (dim 0): dynamic
- sequence axis (dim 2 for q/k/v): dynamic

Static axes:
- num_heads
- head_dim
"""

import argparse
import math
import os
import sys
from typing import List, Tuple

import numpy as np
import pypto
import pypto.pypto_impl as pypto_impl
import torch
from numpy.testing import assert_allclose


NUM_HEADS = 8
HEAD_DIM = 64


def _peek_run_mode_from_argv(default: str = "npu") -> str:
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            if value in ("npu", "sim"):
                return value
        if arg.startswith("--run_mode="):
            value = arg.split("=", 1)[1]
            if value in ("npu", "sim"):
                return value
    return default


global_run_mode = pypto.RunMode.NPU
if _peek_run_mode_from_argv("npu") == "sim":
    global_run_mode = pypto.RunMode.SIM


def _get_device_id_from_env() -> int:
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        raise RuntimeError("Please export TILE_FWK_DEVICE_ID before running in NPU mode.")
    return int(os.environ["TILE_FWK_DEVICE_ID"])


def _torch_golden_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


@pypto.frontend.jit(
    runtime_options={
        "run_mode": global_run_mode,
        "stitch_function_max_num": 128,
    },
    pass_options={
        "pg_upper_bound": 1536,
        "cube_l1_reuse_setting": {0: 4},
    },
)
def online_attention_dynamic_kernel_v2(
    q: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    k: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    v: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, NUM_HEADS, pypto.DYNAMIC, HEAD_DIM], pypto.DT_FP32),
    tile_b: int,
    tile_q: int,
    tile_k: int,
):
    # Match existing attention kernels: enable axis-combine for dynamic loops.
    pypto.experimental.set_operation_options(combine_axis=True)

    bs = q.shape[0]
    sq = q.shape[2]
    sk = k.shape[2]
    scale = np.float32(1.0 / (HEAD_DIM ** 0.5))

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

        # Initialize online softmax state with first KV block.
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
        scores = pypto.matmul(
            q_view,
            k_view,
            out_dtype=pypto.DT_FP32,
            a_trans=False,
            b_trans=True,
        )
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

        # Update state for remaining KV blocks.
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
            scores = pypto.matmul(
                q_view,
                k_view,
                out_dtype=pypto.DT_FP32,
                a_trans=False,
                b_trans=True,
            )
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
        out_tile = pypto.div(o_acc, l_acc)
        pypto.assemble(out_tile, [b0, 0, 0, 0], out)


def _run_one_case(
    case: Tuple[int, int, int],
    tile_b: int,
    tile_q: int,
    tile_k: int,
    device: str,
) -> None:
    bsz, sq, sk = case
    if sq > tile_q:
        raise ValueError(
            f"Current kernel uses one q-tile, require seq_q <= tile_q, got seq_q={sq}, tile_q={tile_q}"
        )
    q = torch.randn(bsz, NUM_HEADS, sq, HEAD_DIM, dtype=torch.float32, device=device)
    k = torch.randn(bsz, NUM_HEADS, sk, HEAD_DIM, dtype=torch.float32, device=device)
    v = torch.randn(bsz, NUM_HEADS, sk, HEAD_DIM, dtype=torch.float32, device=device)
    out = torch.zeros(bsz, NUM_HEADS, sq, HEAD_DIM, dtype=torch.float32, device=device)

    online_attention_dynamic_kernel_v2(q, k, v, out, tile_b, tile_q, tile_k)

    golden = _torch_golden_attention(q, k, v)
    if device.startswith("npu"):
        torch.npu.synchronize()
    out_cpu = out.detach().cpu().numpy()
    golden_cpu = golden.detach().cpu().numpy()
    max_diff = float(np.max(np.abs(out_cpu - golden_cpu)))
    assert_allclose(out_cpu, golden_cpu, rtol=1e-3, atol=1e-3)
    print(
        f"case(b={bsz}, sq={sq}, sk={sk}) "
        f"tile=({tile_b},{tile_q},{tile_k}) max_diff={max_diff:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic online softmax attention in PyPTO")
    parser.add_argument("--run_mode", type=str, default="npu", choices=["npu", "sim"])
    parser.add_argument("--disable_ooo_pass", action="store_true", default=True)
    parser.add_argument("--tile_b", type=int, default=2)
    parser.add_argument("--tile_q", type=int, default=32)
    parser.add_argument("--tile_k", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq_q", type=int, default=64)
    parser.add_argument("--seq_kv", type=int, default=64)
    args = parser.parse_args()

    if args.disable_ooo_pass:
        pypto_impl.SetPassConfig("PVC2_OOO", "OoOSchedule", "disable_pass", True)

    if args.run_mode == "npu":
        import torch_npu  # noqa: F401

        device_id = _get_device_id_from_env()
        torch.npu.set_device(device_id)
        device = f"npu:{device_id}"
    else:
        device = "cpu"

    case = (args.batch, args.seq_q, args.seq_kv)
    _run_one_case(case, args.tile_b, args.tile_q, args.tile_k, device)
    print("Dynamic online-softmax attention test passed.")


if __name__ == "__main__":
    main()

