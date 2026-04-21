#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Minimal reproducer: linear backward only, matching 608's "tmp + b_loop_1/l_loop_2"
# pattern for d_embeddings. See issue/minimal_608_embedding_two_loop_analysis.md.
#
# Do NOT add ``from __future__ import annotations`` — it breaks @jit parameter parsing.
#
# Run (NPU):
#   export PTO_TILE_LIB_CODE_PATH=<path-to>/pto-isa
#   export TILE_FWK_DEVICE_ID=0
#   python issue/minimal_608_embedding_two_loop.py

import os
import sys

import pypto
import torch

os.environ["PTO_TILE_LIB_CODE_PATH"] = "/mnt/workspace/gitCode/cann/pto-isa"

# Problem scale (see analysis §8): B=H=1, L=128, D=16; UNROLL 默认 [1]（可改 [64] 做多 L tile 对照）。
B_STATIC = 1
L_STATIC = 256
H_STATIC = 1
D_STATIC = 16
UNROLL = [1]

RTOL = 1e-3
ATOL = 1e-3


def linear_backward_golden(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor):
    """dy: [tile, D_out], x: [tile, D_in], weight: [D_in, D_out] -> dx, dW, db."""
    dx = dy @ weight.T
    d_w = x.T @ dy
    d_b = dy.sum(dim=0)
    return dx, d_w, d_b


def golden_full(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
):
    """dy [B,L,H,D], x [B,L,D], weight [H,D,D] — single-head linear bwd."""
    dx, d_w, d_b = linear_backward_golden(
        dy[:, :, 0, :].reshape(-1, dy.shape[-1]),
        x.reshape(-1, x.shape[-1]),
        weight[0],
    )
    d_emb = dx.reshape(x.shape[0], x.shape[1], x.shape[2])
    return d_emb, d_w.unsqueeze(0), d_b.unsqueeze(0)


def linear_backward_module(dy, x, weight):
    """Tile-local linear backward (aligned with issue/608.py)."""
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
    dx = pypto.matmul(dy, weight, pypto.DT_FP32, b_trans=True)
    d_w = pypto.matmul(x, dy, pypto.DT_FP32, a_trans=True)
    pypto.set_vec_tile_shapes(16, 1024)
    db = pypto.sum(dy, 0)
    return dx, d_w, db


@pypto.frontend.jit(debug_options=dict(runtime_debug_mode=1))
def k_linear_direct(
    dy: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    d_emb: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
    d_w: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    d_b: pypto.Tensor([H_STATIC, D_STATIC], pypto.DT_FP32),
):
    b, l, _, _ = dy.shape
    h_idx = 0
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
        pypto.set_vec_tile_shapes(1, 64, 512)
        d_emb[0, l_idx : l_idx + tile] = dx
    pypto.set_vec_tile_shapes(1, 128, 128)
    d_w[0] = acc_w
    pypto.set_vec_tile_shapes(1024)
    d_b[0] = acc_b


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
    # 仅写 tmp[0, ...] 与 golden 只用 batch 0 对齐；B>1 时需按 b 写入/读取再测。
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


@pypto.frontend.jit(debug_options=dict(runtime_debug_mode=1))
def k_linear_tmp_sum_external_tmp(
    dy: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    x: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    tmp: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_STATIC, D_STATIC], pypto.DT_FP32),
    d_emb: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D_STATIC], pypto.DT_FP32),
    d_w: pypto.Tensor([H_STATIC, D_STATIC, D_STATIC], pypto.DT_FP32),
    d_b: pypto.Tensor([H_STATIC, D_STATIC], pypto.DT_FP32),
):
    """Same as k_linear_tmp_sum but tmp is an I/O tensor (host-visible buffer)."""
    b, l, _, _ = dy.shape
    h_idx = 0
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


def _check(tag: str, got_emb, got_w, got_b, ref_emb, ref_w, ref_b) -> bool:
    ok_e = torch.allclose(got_emb, ref_emb, rtol=RTOL, atol=ATOL)
    ok_w = torch.allclose(got_w, ref_w, rtol=RTOL, atol=ATOL)
    ok_b = torch.allclose(got_b, ref_b, rtol=RTOL, atol=ATOL)
    print(
        f"{tag}: d_emb={'OK' if ok_e else 'FAIL'}, "
        f"d_w={'OK' if ok_w else 'FAIL'}, d_b={'OK' if ok_b else 'FAIL'}"
    )
    if not ok_e:
        d = (got_emb - ref_emb).abs().max().item()
        print(f"  d_emb max abs diff vs golden: {d}")
    return ok_e and ok_w and ok_b


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

    def zeros_like_outputs():
        d_emb = torch.zeros(B_STATIC, L_STATIC, D_STATIC, dtype=torch.float32, device=device)
        d_w = torch.zeros_like(weight)
        d_b = torch.zeros(H_STATIC, D_STATIC, dtype=torch.float32, device=device)
        return d_emb, d_w, d_b

    print("linear backward, direct")
    d_emb, d_w, d_b = zeros_like_outputs()
    k_linear_direct(dy, x, weight, d_emb, d_w, d_b)
    torch.npu.synchronize()
    ok_direct = _check("direct", d_emb, d_w, d_b, ref_emb, ref_w, ref_b)

    print("linear backward, tmp + b_loop_1 / l_loop_2")
    d_emb, d_w, d_b = zeros_like_outputs()
    k_linear_tmp_sum(dy, x, weight, d_emb, d_w, d_b)
    torch.npu.synchronize()
    ok_tmp = _check("tmp+sum", d_emb, d_w, d_b, ref_emb, ref_w, ref_b)

    print("linear backward, external tmp + b_loop_1 / l_loop_2")
    d_emb, d_w, d_b = zeros_like_outputs()
    tmp_ext = torch.zeros(B_STATIC, L_STATIC, H_STATIC, D_STATIC, dtype=torch.float32, device=device)
    k_linear_tmp_sum_external_tmp(dy, x, weight, tmp_ext, d_emb, d_w, d_b)
    torch.npu.synchronize()
    ok_ext = _check("tmp_out+sum", d_emb, d_w, d_b, ref_emb, ref_w, ref_b)

    if ok_direct and not ok_tmp:
        print(
            "\nREPRODUCED: matmul backward + internal tmp + two-phase sum "
            "mismatches golden (typically d_emb and d_w; d_b often still OK). "
            "See issue/minimal_608_embedding_two_loop_analysis.md."
        )
        return 1
    if ok_direct and ok_tmp:
        print("\nAll paths aligned golden within rtol/atol (issue may be fixed).")
        return 0
    print("\nUnexpected: direct path did not fully match golden; check env / shapes.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
