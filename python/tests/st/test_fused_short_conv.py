#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import gc
import os
import time
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pypto
import pytest
import torch
import torch.nn.functional as F
import torch_npu


@pypto.jit
def pypto_short_conv_fwd_kernel(
    x_padded: pypto.Tensor,   # [B, T + W - 1, D]
    weight: pypto.Tensor,     # [1, W, D]
    out: pypto.Tensor,        # [B, T, D]
    kernel_size: int,
    channel_step: int,
    seq_step: int,
    activation: str = "silu",
) -> None:
    b_size, t_size, d_size = out.shape
    w_size = kernel_size

    pypto.set_vec_tile_shapes(1, seq_step, channel_step)
    for seq_idx in pypto.loop(0, t_size, seq_step, name="seq_idx_loop", idx_name="seq_idx"):
        for batch_idx in pypto.loop(0, b_size, name="batch_idx_loop", idx_name="batch_idx"):
            for dim_idx in pypto.loop(0, d_size, channel_step, name="dim_idx_loop", idx_name="dim_idx"):
                y_tile = pypto.full([1, seq_step, channel_step], 0.0, dtype=x_padded.dtype)
                for wi in range(w_size):
                    w_tile = pypto.view(weight, shape=[1, 1, channel_step], offsets=[0, wi, dim_idx])
                    x_tile = pypto.view(
                        x_padded,
                        shape=[1, seq_step, channel_step],
                        offsets=[batch_idx, seq_idx + wi, dim_idx],
                    )
                    y_tile[:] = pypto.add(y_tile, pypto.mul(x_tile, w_tile))
                if activation == "silu":
                    y_tile = pypto.mul(y_tile, pypto.sigmoid(y_tile))
                out[batch_idx:batch_idx + 1, seq_idx:seq_idx + seq_step, dim_idx:dim_idx + channel_step] = y_tile


@pypto.jit
def pypto_silu_grad_kernel(
    y_pre: pypto.Tensor,      # [B, T, D]
    grad_out: pypto.Tensor,   # [B, T, D]
    grad_silu: pypto.Tensor,  # [B, T, D]
    channel_step: int,
    seq_step: int,
) -> None:
    b_size, t_size, d_size = y_pre.shape
    pypto.set_vec_tile_shapes(1, seq_step, channel_step)
    for seq_idx in pypto.loop(0, t_size, seq_step, name="seq_idx_loop", idx_name="seq_idx"):
        for batch_idx in pypto.loop(0, b_size, name="batch_idx_loop", idx_name="batch_idx"):
            for dim_idx in pypto.loop(0, d_size, channel_step, name="dim_idx_loop", idx_name="dim_idx"):
                dy_tile = pypto.view(grad_out, [1, seq_step, channel_step], [batch_idx, seq_idx, dim_idx])
                y_tile = pypto.view(y_pre, [1, seq_step, channel_step], [batch_idx, seq_idx, dim_idx])
                y_sigmoid = pypto.sigmoid(y_tile)
                one_minus_sigmoid = pypto.add(pypto.neg(y_sigmoid), 1.0)
                local_grad = pypto.mul(y_sigmoid, pypto.add(pypto.mul(y_tile, one_minus_sigmoid), 1.0))
                grad_silu[batch_idx:batch_idx + 1, seq_idx:seq_idx + seq_step, dim_idx:dim_idx + channel_step] = pypto.mul(
                    dy_tile, local_grad
                )


@pypto.jit
def pypto_short_conv_bwd_kernel(
    x: pypto.Tensor,          # [B, T, D]
    grad_out_padded: pypto.Tensor,  # [B, T + W - 1, D]
    weight: pypto.Tensor,     # [1, W, D]
    grad_x: pypto.Tensor,     # [B, T, D]
    grad_w_partial: pypto.Tensor,  # [B * NT, W, D]
    kernel_size: int,
    channel_step: int,
    seq_step: int,
) -> None:
    b_size, t_size, d_size = x.shape
    w_size = kernel_size
    nt_size = t_size // seq_step
    pypto.set_vec_tile_shapes(1, seq_step, channel_step)

    for seq_idx in pypto.loop(0, t_size, seq_step, name="seq_idx_loop", idx_name="seq_idx"):
        for batch_idx in pypto.loop(0, b_size, name="batch_idx_loop", idx_name="batch_idx"):
            for dim_idx in pypto.loop(0, d_size, channel_step, name="dim_idx_loop", idx_name="dim_idx"):
                dx_tile = pypto.full([1, seq_step, channel_step], 0.0, dtype=x.dtype)
                x_tile = pypto.view(x, [1, seq_step, channel_step], [batch_idx, seq_idx, dim_idx])

                for wi in range(w_size):
                    dy_tile = pypto.view(grad_out_padded, [1, seq_step, channel_step], [batch_idx, seq_idx + wi, dim_idx])
                    w_tile = pypto.view(weight, [1, 1, channel_step], [0, w_size - wi - 1, dim_idx])
                    dx_tile[:] = pypto.add(dx_tile, pypto.mul(dy_tile, w_tile))
                    dw_part = pypto.sum(pypto.mul(dy_tile, x_tile), dim=1, keepdim=True)
                    row_idx = batch_idx * nt_size + seq_idx // seq_step
                    grad_w_partial[row_idx:row_idx + 1, w_size - wi - 1:w_size - wi, dim_idx:dim_idx + channel_step] = dw_part

                grad_x[batch_idx:batch_idx + 1, seq_idx:seq_idx + seq_step, dim_idx:dim_idx + channel_step] = dx_tile


@pypto.jit
def pypto_short_conv_fwd_kernel_unroll_opt(
    x_padded: pypto.Tensor,   # [B, T + W - 1, D]
    weight: pypto.Tensor,     # [1, W, D]
    out: pypto.Tensor,        # [B, T, D]
    kernel_size: int,
    channel_step: int,
    seq_step: int,
    activation: str = "silu",
) -> None:
    b_size, t_size, d_size = out.shape
    w_size = kernel_size
    channel_tile_num = d_size // channel_step

    for seq_idx, seq_step_unroll_size in pypto.loop_unroll(
        0, t_size, seq_step, name="seq_idx_loop", idx_name="seq_idx", unroll_list=[2]
    ):
        pypto.set_vec_tile_shapes(1, seq_step_unroll_size * seq_step, channel_step)
        for batch_idx in pypto.loop(0, b_size, name="batch_idx_loop", idx_name="batch_idx"):
            for channel_tile_idx in range(channel_tile_num):
                dim_idx = channel_tile_idx * channel_step
                y_tile = pypto.full(
                    [1, seq_step_unroll_size * seq_step, channel_step],
                    0.0,
                    dtype=x_padded.dtype,
                )
                for wi in range(w_size):
                    w_tile = pypto.view(weight, [1, 1, channel_step], [0, wi, dim_idx])
                    x_tile = pypto.view(
                        x_padded,
                        [1, seq_step_unroll_size * seq_step, channel_step],
                        [batch_idx, seq_idx + wi, dim_idx],
                    )
                    y_tile[:] = pypto.add(y_tile, pypto.mul(x_tile, w_tile))
                if activation == "silu":
                    y_tile = pypto.mul(y_tile, pypto.sigmoid(y_tile))
                out[
                    batch_idx:batch_idx + 1,
                    seq_idx:seq_idx + seq_step_unroll_size * seq_step,
                    dim_idx:dim_idx + channel_step,
                ] = y_tile


@pypto.jit
def pypto_silu_grad_kernel_unroll_opt(
    y_pre: pypto.Tensor,
    grad_out: pypto.Tensor,
    grad_silu: pypto.Tensor,
    channel_step: int,
    seq_step: int,
) -> None:
    b_size, t_size, d_size = y_pre.shape
    channel_tile_num = d_size // channel_step
    for seq_idx, seq_step_unroll_size in pypto.loop_unroll(
        0, t_size, seq_step, name="seq_idx_loop", idx_name="seq_idx", unroll_list=[2]
    ):
        pypto.set_vec_tile_shapes(1, seq_step_unroll_size * seq_step, channel_step)
        for batch_idx in pypto.loop(0, b_size, name="batch_idx_loop", idx_name="batch_idx"):
            for channel_tile_idx in range(channel_tile_num):
                dim_idx = channel_tile_idx * channel_step
                dy_tile = pypto.view(
                    grad_out,
                    [1, seq_step_unroll_size * seq_step, channel_step],
                    [batch_idx, seq_idx, dim_idx],
                )
                y_tile = pypto.view(
                    y_pre,
                    [1, seq_step_unroll_size * seq_step, channel_step],
                    [batch_idx, seq_idx, dim_idx],
                )
                y_sigmoid = pypto.sigmoid(y_tile)
                one_minus_sigmoid = pypto.add(pypto.neg(y_sigmoid), 1.0)
                y_times_one_minus_sigmoid = pypto.mul(y_tile, one_minus_sigmoid)
                one_plus = pypto.add(y_times_one_minus_sigmoid, 1.0)
                silu_grad_tile = pypto.mul(dy_tile, pypto.mul(y_sigmoid, one_plus))
                grad_silu[
                    batch_idx:batch_idx + 1,
                    seq_idx:seq_idx + seq_step_unroll_size * seq_step,
                    dim_idx:dim_idx + channel_step,
                ] = silu_grad_tile


@pypto.jit
def pypto_short_conv_bwd_kernel_unroll_opt(
    x: pypto.Tensor,               # [B, T, D]
    grad_out_padded: pypto.Tensor, # [B, T + W - 1, D]
    weight: pypto.Tensor,          # [1, W, D]
    grad_x: pypto.Tensor,          # [B, T, D]
    grad_w_partial: pypto.Tensor,  # [B * NT, W, D]
    kernel_size: int,
    channel_step: int,
    seq_step: int,
) -> None:
    b_size, t_size, d_size = x.shape
    w_size = kernel_size
    nt_size = t_size // seq_step
    for seq_idx, seq_step_unroll_size in pypto.loop_unroll(
        0, t_size, seq_step, name="seq_idx_loop", idx_name="seq_idx", unroll_list=[2]
    ):
        pypto.set_vec_tile_shapes(1, seq_step_unroll_size * seq_step, channel_step)
        for batch_idx in pypto.loop(0, b_size, name="batch_idx_loop", idx_name="batch_idx"):
            for dim_idx in pypto.loop(0, d_size, channel_step, name="dim_idx_loop", idx_name="dim_idx"):
                dx_tile = pypto.full(
                    [1, seq_step_unroll_size * seq_step, channel_step], 0.0, dtype=x.dtype
                )
                for wi in range(w_size):
                    dy_tile = pypto.view(
                        grad_out_padded,
                        [1, seq_step_unroll_size * seq_step, channel_step],
                        [batch_idx, seq_idx + wi, dim_idx],
                    )
                    w_tile = pypto.view(weight, [1, 1, channel_step], [0, w_size - wi - 1, dim_idx])
                    dx_tile[:] = pypto.add(dx_tile, pypto.mul(dy_tile, w_tile))

                    x_tile = pypto.view(
                        x,
                        [1, seq_step_unroll_size * seq_step, channel_step],
                        [batch_idx, seq_idx, dim_idx],
                    )
                    dw_part = pypto.sum(pypto.mul(dy_tile, x_tile), dim=1, keepdim=True)
                    row_idx = batch_idx * nt_size + seq_idx // seq_step
                    grad_w_partial[
                        row_idx:row_idx + 1 + seq_step_unroll_size,
                        w_size - wi - 1:w_size - wi,
                        dim_idx:dim_idx + channel_step,
                    ] = dw_part

                grad_x[
                    batch_idx:batch_idx + 1,
                    seq_idx:seq_idx + seq_step_unroll_size * seq_step,
                    dim_idx:dim_idx + channel_step,
                ] = dx_tile


def compute_silu_grad_pypto_unroll_opt(
    y_pre: torch.Tensor,
    grad_out: torch.Tensor,
    channel_step: int,
    seq_step: int,
) -> torch.Tensor:
    grad_silu = torch.zeros_like(y_pre)
    pypto_silu_grad_kernel_unroll_opt(
        pypto.from_torch(y_pre, "y_pre"),
        pypto.from_torch(grad_out, "grad_out"),
        pypto.from_torch(grad_silu, "grad_silu"),
        channel_step,
        seq_step,
    )
    return grad_silu


def compute_silu_grad_pypto_raw(
    y_pre: torch.Tensor,
    grad_out: torch.Tensor,
    channel_step: int,
    seq_step: int,
) -> torch.Tensor:
    grad_silu = torch.zeros_like(y_pre)
    pypto_silu_grad_kernel(
        pypto.from_torch(y_pre, "y_pre"),
        pypto.from_torch(grad_out, "grad_out"),
        pypto.from_torch(grad_silu, "grad_silu"),
        channel_step,
        seq_step,
    )
    return grad_silu


def short_conv_torch_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation: Optional[str],
) -> torch.Tensor:
    _, t_size, d_size = x.shape
    w_size = weight.shape[1]
    # Use functional conv to keep `weight` in autograd graph.
    y = F.conv1d(
        x.transpose(1, 2),
        weight.unsqueeze(1),
        bias=None,
        stride=1,
        padding=w_size - 1,
        dilation=1,
        groups=d_size,
    )[:, :, :t_size].transpose(1, 2)
    if activation == "silu":
        y = F.silu(y)
    return y


def short_conv_pypto(
    x: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: int,
    activation: Optional[str],
    channel_step: int,
    seq_step: int,
    use_opt: bool = True,
) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=x.dtype)
    x_padded = F.pad(x, (0, 0, kernel_size - 1, 0), mode="constant", value=0.0)
    if use_opt:
        pypto_short_conv_fwd_kernel_unroll_opt(
            pypto.from_torch(x_padded, name="x_padded"),
            pypto.from_torch(weight.transpose(1, 0).contiguous().unsqueeze(0), name="weight"),
            pypto.from_torch(out, name="out"),
            kernel_size=kernel_size,
            channel_step=channel_step,
            seq_step=seq_step,
            activation=activation if activation is not None else "none",
        )
    else:
        pypto_short_conv_fwd_kernel(
            pypto.from_torch(x_padded, name="x_padded"),
            pypto.from_torch(weight.transpose(1, 0).contiguous().unsqueeze(0), name="weight"),
            pypto.from_torch(out, name="out"),
            kernel_size=kernel_size,
            channel_step=channel_step,
            seq_step=seq_step,
            activation=activation if activation is not None else "none",
        )
    return out


def short_conv_pypto_bwd_silu(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: int,
    channel_step: int,
    seq_step: int,
    use_opt: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y_pre = short_conv_pypto(
        x, weight, kernel_size, activation=None, channel_step=channel_step, seq_step=seq_step, use_opt=use_opt
    )
    if use_opt:
        grad_silu = compute_silu_grad_pypto_unroll_opt(
            y_pre.contiguous(),
            grad_out.contiguous(),
            channel_step=channel_step,
            seq_step=seq_step,
        )
    else:
        grad_silu = compute_silu_grad_pypto_raw(
            y_pre.contiguous(),
            grad_out.contiguous(),
            channel_step=channel_step,
            seq_step=seq_step,
        )

    b_size, t_size, _ = x.shape
    grad_x = torch.zeros_like(x)
    grad_w_partial = torch.zeros(b_size * t_size // seq_step, kernel_size, x.shape[-1], dtype=x.dtype, device=x.device)
    grad_out_padded = F.pad(grad_silu, (0, 0, 0, kernel_size - 1), mode="constant", value=0.0)

    if use_opt:
        pypto_short_conv_bwd_kernel_unroll_opt(
            pypto.from_torch(x.contiguous(), name="x"),
            pypto.from_torch(grad_out_padded, name="grad_out_padded"),
            pypto.from_torch(weight.transpose(1, 0).contiguous().unsqueeze(0), name="weight"),
            pypto.from_torch(grad_x, name="grad_x"),
            pypto.from_torch(grad_w_partial, name="grad_w_partial"),
            kernel_size=kernel_size,
            channel_step=channel_step,
            seq_step=seq_step,
        )
    else:
        pypto_short_conv_bwd_kernel(
            pypto.from_torch(x.contiguous(), name="x"),
            pypto.from_torch(grad_out_padded, name="grad_out_padded"),
            pypto.from_torch(weight.transpose(1, 0).contiguous().unsqueeze(0), name="weight"),
            pypto.from_torch(grad_x, name="grad_x"),
            pypto.from_torch(grad_w_partial, name="grad_w_partial"),
            kernel_size=kernel_size,
            channel_step=channel_step,
            seq_step=seq_step,
        )
    grad_w = grad_w_partial.sum(0).transpose(1, 0).contiguous()
    return grad_x, grad_w


class FusedShortConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, kernel_size, channel_step, seq_step):
        y = short_conv_pypto(
            x,
            weight,
            kernel_size=kernel_size,
            activation="silu",
            channel_step=channel_step,
            seq_step=seq_step,
            use_opt=True,
        )
        ctx.save_for_backward(x.contiguous(), weight.contiguous())
        ctx.kernel_size = kernel_size
        ctx.channel_step = channel_step
        ctx.seq_step = seq_step
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x, grad_w = short_conv_pypto_bwd_silu(
            x=x,
            grad_out=grad_out,
            weight=weight,
            kernel_size=ctx.kernel_size,
            channel_step=ctx.channel_step,
            seq_step=ctx.seq_step,
            use_opt=True,
        )
        return grad_x, grad_w, None, None, None


class FusedShortConvFunctionRaw(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, kernel_size, channel_step, seq_step):
        y = short_conv_pypto(
            x,
            weight,
            kernel_size=kernel_size,
            activation="silu",
            channel_step=channel_step,
            seq_step=seq_step,
            use_opt=False,
        )
        ctx.save_for_backward(x.contiguous(), weight.contiguous())
        ctx.kernel_size = kernel_size
        ctx.channel_step = channel_step
        ctx.seq_step = seq_step
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x, grad_w = short_conv_pypto_bwd_silu(
            x=x,
            grad_out=grad_out,
            weight=weight,
            kernel_size=ctx.kernel_size,
            channel_step=ctx.channel_step,
            seq_step=ctx.seq_step,
            use_opt=False,
        )
        return grad_x, grad_w, None, None, None


def test_fused_short_conv_fwd_bwd():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    torch.manual_seed(0)

    b_size, t_size, d_size, w_size = 1, 8192, 4096, 4
    channel_step, seq_step = 64, 128
    dtype = torch.float32

    x_data = torch.randn(b_size, t_size, d_size, dtype=dtype, device=device) * 0.01
    w_data = torch.randn(d_size, w_size, dtype=dtype, device=device) * 0.01
    grad_out = torch.randn(b_size, t_size, d_size, dtype=dtype, device=device) * 0.01

    x_torch = x_data.clone().requires_grad_(True)
    w_torch = w_data.clone().requires_grad_(True)
    y_torch = short_conv_torch_reference(x_torch, w_torch, activation="silu")
    y_torch.backward(grad_out)
    grad_x_torch = x_torch.grad.detach().clone()
    grad_w_torch = w_torch.grad.detach().clone()

    x_pypto = x_data.clone().requires_grad_(True)
    w_pypto = w_data.clone().requires_grad_(True)
    y_pypto = FusedShortConvFunction.apply(x_pypto, w_pypto, w_size, channel_step, seq_step)
    y_pypto.backward(grad_out)
    grad_x_pypto = x_pypto.grad.detach().clone()
    grad_w_pypto = w_pypto.grad.detach().clone()

    np.testing.assert_allclose(
        y_pypto.detach().float().cpu().numpy(),
        y_torch.detach().float().cpu().numpy(),
        atol=1e-3,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        grad_x_pypto.float().cpu().numpy(),
        grad_x_torch.float().cpu().numpy(),
        atol=1e-3,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        grad_w_pypto.float().cpu().numpy(),
        grad_w_torch.float().cpu().numpy(),
        atol=1e-3,
        rtol=1e-3,
    )


def _npu_clear_device(device: str) -> None:
    """段间清理：同步、回收 Python 侧循环引用、尽量归还 NPU 缓存。"""
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()


def _perf_subtest_torch_vs_pypto_variant(
    device: str,
    *,
    b_size: int,
    t_size: int,
    d_size: int,
    w_size: int,
    channel_step: int,
    seq_step: int,
    dtype: torch.dtype,
    num_rounds: int,
    variant: Literal["raw", "opt"],
) -> Dict[str, float]:
    """单段性能：每轮先 Torch 再 PyPTO（raw 或 opt），每轮内做数值校验；不打印逐轮日志。"""
    x_data = torch.randn(b_size, t_size, d_size, dtype=dtype, device=device) * 0.01
    w_data = torch.randn(d_size, w_size, dtype=dtype, device=device) * 0.01
    x_warm_t = x_data.clone().requires_grad_(True)
    w_warm_t = w_data.clone().requires_grad_(True)
    y_w = short_conv_torch_reference(x_warm_t, w_warm_t, activation="silu")
    grad_warm = torch.randn_like(y_w)
    y_w.backward(grad_warm)
    torch.npu.synchronize()
    x_warm_p = x_data.clone().requires_grad_(True)
    w_warm_p = w_data.clone().requires_grad_(True)
    if variant == "raw":
        y_p = FusedShortConvFunctionRaw.apply(x_warm_p, w_warm_p, w_size, channel_step, seq_step)
    else:
        y_p = FusedShortConvFunction.apply(x_warm_p, w_warm_p, w_size, channel_step, seq_step)
    y_p.backward(grad_warm)
    torch.npu.synchronize()
    del x_data, w_data, x_warm_t, w_warm_t, y_w, grad_warm, x_warm_p, w_warm_p, y_p
    _npu_clear_device(device)

    torch_times: List[float] = []
    torch_mem_peaks: List[float] = []
    pypto_times: List[float] = []
    pypto_mem_peaks: List[float] = []

    for round_idx in range(num_rounds):
        x_data = torch.randn(b_size, t_size, d_size, dtype=dtype, device=device) * 0.01
        w_data = torch.randn(d_size, w_size, dtype=dtype, device=device) * 0.01
        grad_out = torch.randn(b_size, t_size, d_size, dtype=dtype, device=device) * 0.01

        x_torch = x_data.clone().detach().requires_grad_(True)
        w_torch = w_data.clone().detach().requires_grad_(True)
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        y_torch = short_conv_torch_reference(x_torch, w_torch, activation="silu")
        _ = short_conv_torch_reference(x_torch, w_torch, activation=None)
        y_torch.backward(grad_out)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        torch_times.append(t1 - t0)
        torch_mem_peaks.append(torch.npu.max_memory_allocated(device) / (1024 ** 2))
        grad_x_torch = x_torch.grad.detach().clone()
        grad_w_torch = w_torch.grad.detach().clone()
        y_torch_save = y_torch.detach().clone()

        x_pypto = x_data.clone().detach().requires_grad_(True)
        w_pypto = w_data.clone().detach().requires_grad_(True)
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        if variant == "raw":
            y_pypto = FusedShortConvFunctionRaw.apply(x_pypto, w_pypto, w_size, channel_step, seq_step)
        else:
            y_pypto = FusedShortConvFunction.apply(x_pypto, w_pypto, w_size, channel_step, seq_step)
        y_pypto.backward(grad_out)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        pypto_times.append(t1 - t0)
        pypto_mem_peaks.append(torch.npu.max_memory_allocated(device) / (1024 ** 2))
        grad_x_pypto = x_pypto.grad.detach().clone()
        grad_w_pypto = w_pypto.grad.detach().clone()
        y_pypto_save = y_pypto.detach().clone()

        label = "raw" if variant == "raw" else "opt"
        np.testing.assert_allclose(
            y_pypto_save.float().cpu().numpy(),
            y_torch_save.float().cpu().numpy(),
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"[{label}] Round {round_idx + 1}: y 不一致",
        )
        np.testing.assert_allclose(
            grad_x_pypto.float().cpu().numpy(),
            grad_x_torch.float().cpu().numpy(),
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"[{label}] Round {round_idx + 1}: dx 不一致",
        )
        np.testing.assert_allclose(
            grad_w_pypto.float().cpu().numpy(),
            grad_w_torch.float().cpu().numpy(),
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"[{label}] Round {round_idx + 1}: dw 不一致",
        )

        del (
            x_data, w_data, grad_out, y_torch_save, y_pypto_save,
            x_torch, w_torch, y_torch,
            x_pypto, w_pypto, y_pypto,
            grad_x_torch, grad_w_torch,
            grad_x_pypto, grad_w_pypto,
        )
        torch.npu.empty_cache()

    avg_torch_t = sum(torch_times) / len(torch_times)
    avg_torch_m = sum(torch_mem_peaks) / len(torch_mem_peaks)
    avg_pypto_t = sum(pypto_times) / len(pypto_times)
    avg_pypto_m = sum(pypto_mem_peaks) / len(pypto_mem_peaks)
    speedup = avg_torch_t / avg_pypto_t if avg_pypto_t > 0 else 0.0
    return {
        "avg_torch_t": avg_torch_t,
        "avg_torch_m": avg_torch_m,
        "avg_pypto_t": avg_pypto_t,
        "avg_pypto_m": avg_pypto_m,
        "speedup": speedup,
    }


def test_fused_short_conv_fwd_bwd_performance(
    device_id: int = 4,
    num_rounds: int = 10,
):
    """主测试：分测1 Torch vs raw，清理 NPU；分测2 Torch vs opt；最后合并打印对比成绩。"""
    device = f"npu:{device_id}"
    torch.npu.set_device(device)

    b_size, t_size, d_size, w_size = 1, 8192, 4096, 4
    channel_step, seq_step = 64, 128
    dtype = torch.float32

    # 分测1：Torch vs PyPTO(raw)
    stats_raw = _perf_subtest_torch_vs_pypto_variant(
        device,
        b_size=b_size,
        t_size=t_size,
        d_size=d_size,
        w_size=w_size,
        channel_step=channel_step,
        seq_step=seq_step,
        dtype=dtype,
        num_rounds=num_rounds,
        variant="raw",
    )
    _npu_clear_device(device)

    # 分测2：Torch vs PyPTO(opt)（与 raw 段隔离，避免同段内两套 kernel 常驻干扰峰值对比）
    stats_opt = _perf_subtest_torch_vs_pypto_variant(
        device,
        b_size=b_size,
        t_size=t_size,
        d_size=d_size,
        w_size=w_size,
        channel_step=channel_step,
        seq_step=seq_step,
        dtype=dtype,
        num_rounds=num_rounds,
        variant="opt",
    )
    _npu_clear_device(device)

    # 仅输出两段对比汇总（每段内 Torch 与对应 PyPTO 的平均耗时 / 峰值显存 / speedup）
    print("\n" + "=" * 72)
    print(
        f"Short Conv Fwd+Bwd 性能汇总 | {device} | 每段各 {num_rounds} 轮 | "
        f"B={b_size}, T={t_size}, D={d_size}, W={w_size}"
    )
    print("=" * 72)
    print("[分测1] Torch vs PyPTO(raw)")
    print(
        f"  平均耗时: Torch {stats_raw['avg_torch_t'] * 1000:.2f} ms | "
        f"Raw {stats_raw['avg_pypto_t'] * 1000:.2f} ms | "
        f"Speedup {stats_raw['speedup']:.2f}x"
    )
    print(
        f"  平均峰值显存: Torch {stats_raw['avg_torch_m']:.1f} MB | "
        f"Raw {stats_raw['avg_pypto_m']:.1f} MB"
    )
    print("[分测2] Torch vs PyPTO(opt)")
    print(
        f"  平均耗时: Torch {stats_opt['avg_torch_t'] * 1000:.2f} ms | "
        f"Opt {stats_opt['avg_pypto_t'] * 1000:.2f} ms | "
        f"Speedup {stats_opt['speedup']:.2f}x"
    )
    print(
        f"  平均峰值显存: Torch {stats_opt['avg_torch_m']:.1f} MB | "
        f"Opt {stats_opt['avg_pypto_m']:.1f} MB"
    )
    print("=" * 72)


def test_fused_short_conv_fwd_bwd_performance_default():
    test_fused_short_conv_fwd_bwd_performance(
        device_id=int(os.environ.get("TILE_FWK_DEVICE_ID", 4)),
        num_rounds=10,
    )
