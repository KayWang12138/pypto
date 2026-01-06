#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Hello World Example for PyPTO

This example demonstrates the simplest tensor addition.
"""
import os
import sys
import pypto
import pytest
import torch
import torch_npu
from torch._dynamo import allow_in_graph
from hc_pre_golden import gen_hc_pre_data

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../deepseek_v32_exp/utils'))
from compare import compare


hc, d, sinkhorn_iters, norm_eps, hc_eps = 4, 512, 20, 1e-6, 1e-6
mix_hc = (2 + hc) * hc

def rms_norm_denom(x: pypto.Tensor) -> pypto.Tensor:
    print("rms_norm_denom input shape ", x.shape)
    # Compute RMS: sqrt(mean(x^2) + eps)
    squared = x * x
    mean_sq = pypto.sum(squared, dim=-1, keepdim=True)
    mean_sq = mean_sq / x.shape[-1]
    rms = pypto.sqrt((mean_sq + norm_eps))
    return rms


def sigmoid(x: pypto.Tensor) -> pypto.Tensor:
    # sigmoid(x) = 1 / (1 + exp(-x))
    x_neg = pypto.mul(x, -1.0)
    exp_neg = pypto.exp(x_neg)
    ones = pypto.full(exp_neg.shape, 1.0, exp_neg.dtype, valid_shape=exp_neg.shape)
    sigmoid = pypto.div(ones, exp_neg + 1.0)
    return sigmoid


def hc_split_sinkhorn(x: pypto.Tensor, hc_scale: pypto.Tensor, hc_base: pypto.Tensor, real_t, t_idx) \
    -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    tile_t, _ = x.shape # (tile_t, 24)

    pre = (x[:, :hc] + 0.0) * (hc_scale[0:1].reshape([1, 1])) + hc_base[:, :hc] # (tile_t, 4)
    # pre = x[:, :hc] * (hc_scale[0:1].reshape([1, 1])) + hc_base[:, :hc] # (tile_t, 4)
    pre = sigmoid(pre) + hc_eps # (tile_t, 4)

    post = x[:, hc: 2*hc] * (hc_scale[1:2].reshape([1, 1])) + hc_base[:, hc: 2*hc] # (tile_t, 4)
    post = sigmoid(post) * 2.0 # (tile_t, 4)

    comb_flag = (x[:, 2*hc: ] * (hc_scale[2:3].reshape([1, 1])) + hc_base[:, 2*hc: ])

    comb_flag = comb_flag.reshape([tile_t, hc, hc], valid_shape=[pypto.min(real_t - t_idx * tile_t, tile_t), hc, hc]) # (tile_t, 4, 4)
    pypto.set_vec_tile_shapes(16, 64, 64)
    row_max = pypto.amax(comb_flag, -1, True)   # (tile_t, 4, 1)
    comb_flag = pypto.exp(comb_flag - row_max)    # (tile_t, 4, 4)

    row_sum = pypto.sum(comb_flag, -1, True)    # (tile_t, 4, 1)
    comb_flag = comb_flag / row_sum + hc_eps # (tile_t, 4, 4)
    col_sum = pypto.sum(comb_flag, -2, True) # (tile_t, 1, 4)
    comb_flag = comb_flag / (col_sum + hc_eps) # (tile_t, 4, 4)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb_flag.sum(-1, keepdim=True) # (tile_t, 4, 4)
        comb_flag = comb_flag / (row_sum + hc_eps) # (tile_t, 4, 4)
        col_sum = comb_flag.sum(-2, keepdim=True) # (tile_t, 4, 4)
        comb_flag = comb_flag / (col_sum + hc_eps) # (tile_t, 4, 4)
    return pre, post, comb_flag


@pypto.jit(
    host_options={"only_codegen": True}
)
def hc_pre_kernel(x: pypto.Tensor, hc_fn: pypto.Tensor, hc_scale: pypto.Tensor, hc_base_: pypto.Tensor,
                y: pypto.Tensor, post: pypto.Tensor, comb: pypto.Tensor
):
    pypto.set_debug_options(runtime_debug_mode=1)

    pypto.set_vec_tile_shapes(64, 64)
    pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])

    tile_t = 16
    real_t = x.shape[0]
    loop_t_times = (real_t + tile_t - 1) // tile_t

    for _ in pypto.loop(1):
        x_2d = pypto.reshape(x, [real_t, hc*d], inplace=True)
        hc_base= pypto.reshape(hc_base_, [1, mix_hc], inplace=True)
    for t_idx in pypto.loop(loop_t_times, name="t_loop", idx_name="t_idx"):
        x_view = pypto.view(x_2d, [tile_t, hc*d], [t_idx*tile_t, 0])
        x_fp32 = pypto.cast(x_view, pypto.DT_FP32)
        mm_res = pypto.matmul(x_fp32, hc_fn, pypto.DT_FP32, b_trans=True)   # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)
        rms_res = rms_norm_denom(x_fp32)
        rms_res = mm_res / rms_res
        pre, post_, comb_ = hc_split_sinkhorn(rms_res, hc_scale, hc_base, real_t, t_idx)   # (tile_t, hc), (tile_t, hc), (tile_t, hc, hc)
        pre_3d = pre.reshape([tile_t, hc, 1], valid_shape=[pypto.min(real_t - t_idx * tile_t, tile_t), hc, 1])
        x_fp32_3d = x_fp32.reshape([tile_t, hc, d], valid_shape=[pypto.min(real_t - t_idx * tile_t, tile_t), hc, d])
        comb[t_idx*tile_t:, :, :] = comb_
        mul_res = pre_3d * x_fp32_3d
        res_fp32 = pypto.sum(mul_res, dim=-2)
        pypto.set_vec_tile_shapes(64, 64)
        y[t_idx*tile_t:, :] = pypto.cast(res_fp32, pypto.DT_BF16)
        post[t_idx*tile_t:, :] = post_ + 0.0


@allow_in_graph
def hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor)\
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = x.shape[0]
    hc = x.shape[1]
    d = x.shape[2]

    y = torch.zeros([t, d], dtype=x.dtype, device=f'{x.device}')
    post = torch.zeros([t, hc], dtype=hc_fn.dtype, device=f'{x.device}')
    comb = torch.zeros([t, hc, hc], dtype=hc_fn.dtype, device=f'{x.device}')

    in_outs = {
        x: [0],
        hc_fn: None,
        hc_scale: None,
        hc_base: None,
        y:[0],
        post:[0],
        comb:[0],
    }

    pto_in_outs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in in_outs.items()]
    hc_pre_kernel(*pto_in_outs)
    # pypto.runtime._device_synchronize()#内部接口，不推荐使用

    return y, post, comb


class HC_PRE(torch.nn.Module):
    def forward(self, x, hc_fn, hc_scale, hc_base):
        y, post, comb = hc_pre(x, hc_fn, hc_scale, hc_base)
        return y, post, comb

def test_hc_pre_inmodel(t = 16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)
    x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd = gen_hc_pre_data(t)
    print("gen golden success !!!")

    ### to device
    x = x.to(device=f'npu:{device_id}')
    hc_fn = hc_fn.to(device=f'npu:{device_id}')
    hc_scale = hc_scale.to(device=f'npu:{device_id}')
    hc_base = hc_base.to(device=f'npu:{device_id}')

    model = torch.compile(HC_PRE(), backend="eager", dynamic=True)
    # capture model
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y, post, comb = model(x, hc_fn, hc_scale, hc_base)

    for i in range(3):
        print(f"##### Iteration {i+1} before replay")
        # execute
        g.replay()
        pypto.runtime._device_synchronize()
        y, post, comb = y.cpu(), post.cpu(), comb.cpu()
        ### compare
        compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)
        print("y compare success!!!")
        compare(post, post_gd, "post", atol=0.000025, rtol=0.005)
        print("post compare success!!!")
        compare(comb, comb_gd, "comb", atol=0.000025, rtol=0.005)
        print("comb compare success!!!")
        print(f"##### Iteration {i+1} passed")

def test_hc_pre(t = 16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)

    x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd = gen_hc_pre_data(t)
    print("gen golden success !!!")

    y = torch.zeros_like(y_gd).to(device=f'npu:{device_id}')
    post = torch.zeros_like(post_gd).to(device=f'npu:{device_id}')
    comb = torch.zeros_like(comb_gd).to(device=f'npu:{device_id}')

    in_outs = {
        x.to(device=f'npu:{device_id}'): [0],
        hc_fn.to(device=f'npu:{device_id}'): None,
        hc_scale.to(device=f'npu:{device_id}'): None,
        hc_base.to(device=f'npu:{device_id}'): None,
        y:[0],
        post:[0],
        comb:[0],
    }

    pto_in_outs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in in_outs.items()]
    hc_pre_kernel(*pto_in_outs)
    torch_npu.npu.synchronize()

    y = y.cpu()
    post = post.cpu()
    comb = comb.cpu()

    # print("y", y.shape, y)
    # print("post", post.shape, post)
    # print("comb", comb.shape, comb)

    compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)
    print("y compare success!!!")
    compare(post, post_gd, "post", atol=0.000025, rtol=0.005)
    print("post compare success!!!")
    compare(comb, comb_gd, "comb", atol=0.000025, rtol=0.005)
    print("comb compare success!!!")


if __name__ == "__main__":
    print("start test !!!")
    # test_hc_pre_inmodel()
    t_list = {8, 15, 16, 31, 32, 512}
    for t in t_list:
        test_hc_pre(t)
