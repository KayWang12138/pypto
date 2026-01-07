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


def rms_norm_denom(x: pypto.Tensor) -> pypto.Tensor:
    norm_eps = 1e-6
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


def hc_split_sinkhorn(x: pypto.Tensor, hc_scale: pypto.Tensor, hc_base: pypto.Tensor, hc, hc_eps) \
    -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor]:
    sinkhorn_iters = 20
    tile_t, _ = x.shape # (tile_t, 24)
    print("x.shape ", x.shape)

    comb_flag = (x[:, 2*hc: ] * (hc_scale[2:3].reshape([1, 1])) + hc_base[:, 2*hc: ])
    comb_flag = comb_flag.reshape([tile_t, hc, hc]) # (tile_t, 4, 4)

    if tile_t <= 20:
        pypto.set_vec_tile_shapes(1, 16, 16)
    else:
        pypto.set_vec_tile_shapes(128, 16, 16)

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
    return comb_flag


@pypto.jit(
    host_options={"only_codegen": True},
    # for acl graph
    runtime_options={"cfgcache_device_task_num": 100,
                     "cfgcache_root_task_num": 1000,
                     "cfgcache_leaf_task_num": 10000}
)
def hc_pre_kernel(x: pypto.Tensor, hc_fn: pypto.Tensor, hc_scale: pypto.Tensor, hc_base_: pypto.Tensor,
                y: pypto.Tensor, post: pypto.Tensor, comb: pypto.Tensor,
):
    # pypto.set_debug_options(runtime_debug_mode=1)
    pypto.set_debug_options(runtime_debug_mode=2)   ## for acl graph


    pypto.set_vec_tile_shapes(16, 512)
    pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128])

    t = x.shape[0]
    hc = x.shape[1]
    d = x.shape[2]
    mix_hc = (2 + hc) * hc
    hc_eps = 1e-6

    unroll_list = [16, 1]
    # unroll_list=[128, 64, 32, 16, 8, 4, 2, 1]

    for _ in pypto.loop(1):
        x_2d = pypto.reshape(x, [t, hc*d], inplace=True)
        hc_base= pypto.reshape(hc_base_, [1, mix_hc], inplace=True)
    print("t in kernel is ", t)
    for t_idx, unrollLength in pypto.loop_unroll(0, t, 1, name="t_loop", idx_name="t_idx", unroll_list=unroll_list):
        pypto.set_vec_tile_shapes(16, 512)

        tile_t = unrollLength
        print("========================= tile_t: ", tile_t)
        print("========================= t_idx: ", t_idx)

        x_view = pypto.view(x_2d, [tile_t, hc*d], [t_idx, 0])
        x_fp32 = pypto.cast(x_view, pypto.DT_FP32)
        mm_res = pypto.matmul(x_view, hc_fn, pypto.DT_BF16, b_trans=True)   # (t, hc*d) @ (mix_hc, hc*d)^t = (t, mix_hc)
        mm_res = pypto.cast(mm_res, pypto.DT_FP32)

        rms_res = rms_norm_denom(x_fp32)    ## (t, hc*d) -> (t, 1)
        pypto.set_vec_tile_shapes(128, 16)
        rms_res = mm_res / rms_res  ## t, mix_hc

        pre = rms_res[:, :hc] * (hc_scale[0:1].reshape([1, 1])) + hc_base[:, :hc] # (tile_t, 4)
        pre = sigmoid(pre) + hc_eps # (tile_t, 4)

        pypto.set_vec_tile_shapes(128, 16)
        pre_3d = pre.reshape([tile_t, hc, 1])
        x_fp32_3d = x_fp32.reshape([tile_t, hc, d])
        pypto.set_vec_tile_shapes(128, 16, 16)

        mul_res = pre_3d * x_fp32_3d
        res_fp32 = pypto.sum(mul_res, dim=-2)
        res_bf16 = pypto.cast(res_fp32, pypto.DT_BF16)
        pypto.assemble(res_bf16, [t_idx, 0], y)

        post_ = rms_res[:, hc: 2*hc] * (hc_scale[1:2].reshape([1, 1])) + hc_base[:, hc: 2*hc] # (tile_t, 4)
        post_ = sigmoid(post_) * 2.0 # (tile_t, 4)
        pypto.assemble(post_, [t_idx, 0], post)

        comb_ = hc_split_sinkhorn(rms_res, hc_scale, hc_base, hc, hc_eps)   # (tile_t, hc), (tile_t, hc), (tile_t, hc, hc)
        pypto.assemble(comb_, [t_idx, 0, 0], comb)


@allow_in_graph
def npu_hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor)\
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = x.shape[0]
    hc = x.shape[1]
    d = x.shape[2]

    y = torch.zeros([t, d], dtype=x.dtype, device=f'{x.device}')
    post = torch.zeros([t, hc], dtype=hc_scale.dtype, device=f'{x.device}')
    comb = torch.zeros([t, hc, hc], dtype=hc_scale.dtype, device=f'{x.device}')

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

    return y, post, comb


class HC_PRE(torch.nn.Module):
    def forward(self, x, hc_fn, hc_scale, hc_base):
        y, post, comb = npu_hc_pre(x, hc_fn, hc_scale, hc_base)
        return y, post, comb

def test_hc_pre_inmodel(t = 16):
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch.npu.set_device(int(device_id))
    torch.manual_seed(42)
    x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd, mm_res_gd = gen_hc_pre_data(t)
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
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)
        torch.add(x, 0.0)

        y, post, comb = model(x, hc_fn, hc_scale, hc_base)

    for i in range(30):
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

    x, hc_fn, hc_scale, hc_base, y_gd, post_gd, comb_gd, mm_res_gd = gen_hc_pre_data(t)
    print("gen golden success !!!")

    y = torch.zeros_like(y_gd).to(device=f'npu:{device_id}')
    post = torch.zeros_like(post_gd).to(device=f'npu:{device_id}')
    comb = torch.zeros_like(comb_gd).to(device=f'npu:{device_id}')
    # mm_res = torch.zeros_like(mm_res_gd).to(device=f'npu:{device_id}')

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

    # mm_res = mm_res.cpu()
    y = y.cpu()
    post = post.cpu()
    comb = comb.cpu()

    # print("y", y.shape, y)
    # print("post", post.shape, post)
    # print("comb", comb.shape, comb)

    # compare(mm_res, mm_res_gd, "mm_res", atol=0.0001, rtol=0.0078125)
    # print("mm_res compare success!!!")

    compare(y, y_gd, "y", atol=0.0001, rtol=0.0078125)
    print("y compare success!!!")
    compare(post, post_gd, "post", atol=0.000025, rtol=0.005)
    print("post compare success!!!")
    compare(comb, comb_gd, "comb", atol=0.000025, rtol=0.005)
    print("comb compare success!!!")


if __name__ == "__main__":
    print("start test !!!")
    test_hc_pre_inmodel(16)
    # t_list = {8192, 127, 1}
    # for t_dyn in t_list:
    #     test_hc_pre(t_dyn)
    # test_hc_pre(16)

