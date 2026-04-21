#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Isolated test case for k_linear_tmp_sum only.
#
# Do NOT add ``from __future__ import annotations`` — it breaks @jit parameter parsing.
#
# Run (NPU):
#   export PTO_TILE_LIB_CODE_PATH=<path-to>/pto-isa
#   export TILE_FWK_DEVICE_ID=0
#   python issue/minimal_608_embedding_tmp_sum_case.py

import os
import sys

import pypto
import torch

os.environ["PTO_TILE_LIB_CODE_PATH"] = "/mnt/workspace/gitCode/cann/pto-isa"

# Problem scale aligned with minimal_608_embedding_two_loop.py
B_STATIC = 1
L_STATIC = 4096
H_STATIC = 1
D_STATIC = 512
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
        print(
            f"[DIFF] {name}: nan_count={nan_count}, first_nan={first_nan}, last_nan={last_nan}"
        )
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
        print(
            f"  top{rank}: idx={tuple(coords)} got={got_v:.6e} "
            f"ref={ref_v:.6e} abs={val:.6e}"
        )


def linear_backward_golden(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor):
    dx = dy @ weight.T
    d_w = x.T @ dy
    d_b = dy.sum(dim=0)
    return dx, d_w, d_b


def golden_full(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
):
    dx, d_w, d_b = linear_backward_golden(
        dy[:, :, 0, :].reshape(-1, dy.shape[-1]),
        x.reshape(-1, x.shape[-1]),
        weight[0],
    )
    d_emb = dx.reshape(x.shape[0], x.shape[1], x.shape[2])
    return d_emb, d_w.unsqueeze(0), d_b.unsqueeze(0)


def linear_backward_module(dy, x, weight):
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dx = pypto.matmul(dy, weight, pypto.DT_FP32, b_trans=True)
    d_w = pypto.matmul(x, dy, pypto.DT_FP32, a_trans=True)
    pypto.set_vec_tile_shapes(16, 1024)
    db = pypto.sum(dy, 0)
    return dx, d_w, db


@pypto.frontend.jit(debug_options=dict(runtime_debug_mode=1))
def k_linear_tmp_sum(
    dy: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    d_emb: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
    d_w: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    d_b: pypto.Tensor([H_STATIC, D_STATIC], pypto.DT_FP32),
):
    b, l, _, _ = dy.shape
    h_idx = 0
    tmp = pypto.tensor([b, l, H_STATIC, D_STATIC], d_emb.dtype, "tmp")
    pypto.set_vec_tile_shapes(128, 128)
    acc_w = pypto.full(size=[D_STATIC, D_STATIC], fill_value=0.0, dtype=d_w.dtype)
    pypto.set_vec_tile_shapes(1024)
    acc_b = pypto.full(size=[D_STATIC], fill_value=0.0, dtype=d_b.dtype)
    for l_idx, tile in pypto.loop_unroll(0, l, 1, unroll_list=UNROLL):
        pypto.set_vec_tile_shapes(1, 64, 1, 256)
        dy_v = dy[0, l_idx : l_idx + tile, h_idx]
        x_v = x[0, l_idx : l_idx + tile]
        dx, dw, db = linear_backward_module(dy_v, x_v, weight[h_idx])
        pypto.set_vec_tile_shapes(128, 128)
        acc_w[:] = acc_w + dw
        pypto.set_vec_tile_shapes(1024)
        acc_b[:] = acc_b + db
        pypto.set_vec_tile_shapes(1, 64, 1, 512)
        tmp[0, l_idx : l_idx + tile, h_idx] = dx
    for b_idx in pypto.loop(0, b, 1, name="b_loop_1"):
        for l_idx, tile in pypto.loop_unroll(
            0, l, 1, name="l_loop_2", unroll_list=UNROLL
        ):
            pypto.set_vec_tile_shapes(1, 64, 1, 256)
            ch = tmp[b_idx, l_idx : l_idx + tile]
            pypto.set_vec_tile_shapes(64, 1, 256)
            sm = pypto.sum(ch, 1)
            pypto.set_vec_tile_shapes(1, 64, 512)
            d_emb[b_idx, l_idx : l_idx + tile] = sm
    pypto.set_vec_tile_shapes(1, 128, 128)
    d_w[0] = acc_w
    pypto.set_vec_tile_shapes(1024)
    d_b[0] = acc_b


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

    ref_emb, ref_w, ref_b = golden_full(dy, x, weight)

    d_emb = torch.zeros(B_STATIC, L_STATIC, D_STATIC, dtype=torch.float32, device=device)
    d_w = torch.zeros_like(weight)
    d_b = torch.zeros(H_STATIC, D_STATIC, dtype=torch.float32, device=device)
    k_linear_tmp_sum(dy, x, weight, d_emb, d_w, d_b)
    torch.npu.synchronize()

    ok_e = torch.allclose(d_emb, ref_emb, rtol=RTOL, atol=ATOL)
    ok_w = torch.allclose(d_w, ref_w, rtol=RTOL, atol=ATOL)
    ok_b = torch.allclose(d_b, ref_b, rtol=RTOL, atol=ATOL)
    print_diff_stats("d_emb", d_emb, ref_emb)
    print_diff_stats("d_w", d_w, ref_w)
    print_diff_stats("d_b", d_b, ref_b)
    print(
        f"tmp+sum: d_emb={'OK' if ok_e else 'FAIL'}, "
        f"d_w={'OK' if ok_w else 'FAIL'}, d_b={'OK' if ok_b else 'FAIL'}"
    )
    return 0 if (ok_e and ok_w and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
