#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Simplified repro (d_emb only): first loop writes per-tile ``dx`` into ``tmp``;
# second loop copies ``tmp`` into ``d_emb`` without ``ch`` / ``sum``, to isolate
# ``tmp`` -> ``d_emb`` stitch / cell-match behavior.
#
# Do NOT add ``from __future__ import annotations`` — it breaks @jit parameter parsing.
#
# Run (NPU):
#   export PTO_TILE_LIB_CODE_PATH=<path-to>/pto-isa
#   export TILE_FWK_DEVICE_ID=0
#   python issue/minimal_608_embedding_tmp_to_d_emb_only.py
#
# The original full-case script remains: issue/minimal_608_embedding_tmp_sum_case.py

import os
import sys

import pypto
import torch

os.environ["PTO_TILE_LIB_CODE_PATH"] = "/mnt/workspace/gitCode/cann/pto-isa"

B_STATIC = 1
L_STATIC = 127
H_STATIC = 1
D_STATIC = 16
UNROLL = [1]

RTOL = 1e-3
ATOL = 1e-3


def print_diff_stats(name: str, got: torch.Tensor, ref: torch.Tensor, topk: int = 8):
    nan_mask = torch.isnan(got)
    nan_count = int(nan_mask.sum().item())
    if nan_count > 0:
        nan_idx = torch.nonzero(nan_mask, as_tuple=False)
        first_nan = tuple(int(x) for x in nan_idx[0].tolist())
        last_nan = tuple(int(x) for x in nan_idx[-1].tolist())
        print(f"[DIFF] {name}: nan_count={nan_count}, first_nan={first_nan}, last_nan={last_nan}")
    diff = (got - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel = diff / (ref.abs() + 1e-6)
    max_rel = rel.max().item()
    mismatch = ~(torch.isclose(got, ref, rtol=RTOL, atol=ATOL))
    mismatch_count = int(mismatch.sum().item())
    total_count = mismatch.numel()
    print(
        f"[DIFF] {name}: mismatch={mismatch_count}/{total_count}, "
        f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e}"
    )
    if mismatch_count == 0:
        return
    flat_diff = diff.reshape(-1)
    k = min(topk, flat_diff.numel())
    vals, idxs = torch.topk(flat_diff, k=k, largest=True, sorted=True)
    shape = got.shape
    stride = []
    acc = 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    stride = list(reversed(stride))
    for rank, (idx, val) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
        rem = idx
        coords = []
        for st, dim in zip(stride, shape):
            c = rem // st
            rem = rem % st
            coords.append(int(c))
        got_v = got[tuple(coords)].item()
        ref_v = ref[tuple(coords)].item()
        print(f"  top{rank}: idx={tuple(coords)} got={got_v:.6e} ref={ref_v:.6e} abs={val:.6e}")


def golden_d_emb_only(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Same dx as first-loop ``tmp``; second loop is identity copy when H_STATIC==1."""
    dx = (dy[:, :, 0, :].reshape(-1, dy.shape[-1]) @ weight[0].T).reshape(
        x.shape[0], x.shape[1], x.shape[2]
    )
    return dx


def linear_dx_only(dy, x, weight):
    del x  # keep signature for tiling parity with full case; only dx needed
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    return pypto.matmul(dy, weight, pypto.DT_FP32, b_trans=True)


@pypto.frontend.jit(debug_options=dict(runtime_debug_mode=1))
def k_tmp_to_d_emb(
    dy: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    d_emb: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
):
    b, l, _, _ = dy.shape
    h_idx = 0
    tmp = pypto.tensor([b, l, H_STATIC, D_STATIC], d_emb.dtype, "tmp")
    for l_idx, tile in pypto.loop_unroll(0, l, 1, unroll_list=UNROLL):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        dy_v = dy[0, l_idx : l_idx + tile, h_idx]
        x_v = x[0, l_idx : l_idx + tile]
        dx = linear_dx_only(dy_v, x_v, weight[h_idx])
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        tmp[0, l_idx : l_idx + tile, h_idx] = dx
    for b_idx in pypto.loop(0, b, 1, name="b_loop_1"):
        for l_idx, tile in pypto.loop_unroll(0, l, 1, name="l_loop_2", unroll_list=UNROLL):
            pypto.set_vec_tile_shapes(1, 64, 1, 512)
            d_emb[b_idx, l_idx : l_idx + tile] = tmp[b_idx, l_idx : l_idx + tile, h_idx]


def main() -> int:
    if not os.environ.get("PTO_TILE_LIB_CODE_PATH"):
        print(
            "warning: PTO_TILE_LIB_CODE_PATH is unset; set it to your pto-isa path "
            "before running on NPU.",
            file=sys.stderr,
        )
    dev_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    device = f"npu:{dev_id}"
    torch.npu.set_device(dev_id)

    torch.manual_seed(44)
    dy = torch.randn(B_STATIC, L_STATIC, H_STATIC, D_STATIC, dtype=torch.float32, device=device)
    x = torch.randn(B_STATIC, L_STATIC, D_STATIC, dtype=torch.float32, device=device)
    weight = torch.randn(H_STATIC, D_STATIC, D_STATIC, dtype=torch.float32, device=device)

    ref_emb = golden_d_emb_only(dy, x, weight)

    d_emb = torch.zeros(B_STATIC, L_STATIC, D_STATIC, dtype=torch.float32, device=device)
    k_tmp_to_d_emb(dy, x, weight, d_emb)
    torch.npu.synchronize()

    ok_e = torch.allclose(d_emb, ref_emb, rtol=RTOL, atol=ATOL)
    print_diff_stats("d_emb", d_emb, ref_emb)
    print(f"tmp->d_emb: d_emb={'OK' if ok_e else 'FAIL'}")
    return 0 if ok_e else 1


if __name__ == "__main__":
    sys.exit(main())
