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
""" """
import os
import torch
import torch_npu
import pypto
import logging
from attention_hc_post_impl import (
    npu_attention_hc_post,
    attention_hc_post_kernel,
    AttnHcPostConfig,
)
from utils.compare import compare


class AttentionHcPost(torch.nn.Module):
    def forward(self, x, cos, sin, wo_a, wo_b, residual, post, comb):
        for i in range(20):
            torch.add(x, 0.0)
        return npu_attention_hc_post(x, cos, sin, wo_a, wo_b, residual, post, comb)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成, 与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(
            low=min_value, high=max_value, size=data_shape, dtype=dtype
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    """
    q: (t, n_q, rope_dim), bf16
    cos: (t, rope_dim), bf16
    sin: (t, rope_dim), bf16
    """
    input_dtype = q.dtype
    q = q.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    cos = torch.unsqueeze(cos, dim=1)  # [t, 1, rope_dim]
    sin = torch.unsqueeze(sin, dim=1)  # [t, 1, rope_dim]

    t, n, d = q.shape
    q = q.reshape(t, n, d // 2, 2).permute(0, 1, 3, 2).reshape(t, n, d)

    # (t, n_q, rope_dim), (t, 1, rope_dim) = (t, n_q, rope_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)

    if input_dtype != torch.float32:
        q_embed = q_embed.to(input_dtype)
    return q_embed


def compute_attention_hc_post(inputs, params):
    x = inputs[0]
    cos = inputs[1]
    sin = inputs[2]
    wo_a = inputs[3]
    wo_b = inputs[4]
    residual = inputs[5]
    post = inputs[6]
    comb = inputs[7]

    t = params.get("t")
    n_q = params.get("n_q", 64)
    d = params.get("d", 512)
    rope_dim = params.get("rope_dim", 64)
    n_g = params.get("n_g", 8)
    o_lora_rank = params.get("o_lora_rank", 1024)
    attn_dtype = params.get("attn_dtype", torch.bfloat16)
    hc_dtype = params.get("hc_dtype", torch.float32)

    # rope
    rope_in = x[:, :, (d - rope_dim) :]  # (t, n_q, rope_dim), bf16
    nope_res = x[:, :, : (d - rope_dim)]  # (t, n_q, d - rope_dim), bf16
    rope_res = apply_rotary_pos_emb(rope_in, cos, sin)  # (t, n_q, rope_dim), bf16
    x_new = torch.cat((nope_res, rope_res), dim=-1)

    # (t, n_q, d) -> (n_g, t, n_q * d // n_g)
    mm1_left_trans = x_new.reshape(t, n_g, n_q * d // n_g).transpose(1, 0)
    # (n_g, t, n_q * d // n_g) @ (n_g, n_q * d // n_g, o_lora_rank) = (n_g, t, o_lora_rank)
    bmm_1_res = torch.bmm(mm1_left_trans.to(hc_dtype), wo_a.to(hc_dtype)).to(attn_dtype)

    # (n_g, t, o_lora_rank) -> (t, n_g * o_lora_rank)
    bmm_res = bmm_1_res.transpose(1, 0)  # (t, n_g, o_lora_rank)
    bmm_reshpe = bmm_res.reshape(t, n_g * o_lora_rank)
    # (t, n_g * o_lora_rank) @ (n_g * o_lora_rank, h) = (t, h)
    mm_res = torch.mm(bmm_reshpe.to(hc_dtype), wo_b.to(hc_dtype)).to(attn_dtype)
    hc_in = mm_res.to(hc_dtype)

    # hc_post
    y_left = post.unsqueeze(-1) * hc_in.unsqueeze(
        -2
    )  # (t, hc, 1) * (t, 1, h) -> (t, hc, h)
    # (t, hc, hc, 1) * (t, hc, 1, h) -> (t, hc, hc, h) -> (t, hc, h)
    y_right = torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=1)
    y = y_left + y_right  # (t, hc, h)
    y = y.to(attn_dtype)

    return rope_res, bmm_res, mm_res, y


def gen_attention_hc_post_golden(params):
    torch.manual_seed(42)
    t = params.get("t")
    n_q = params.get("n_q", 64)
    d = params.get("d", 512)
    rope_dim = params.get("rope_dim", 64)
    n_g = params.get("n_g", 8)
    o_lora_rank = params.get("o_lora_rank", 1024)
    h = params.get("h", 4096)
    hc = params.get("hc", 4)
    attn_dtype = params.get("attn_dtype", torch.bfloat16)
    hc_dtype = params.get("hc_dtype", torch.float32)

    x = gen_uniform_data([t, n_q, d], -1, 1, attn_dtype)
    cos = gen_uniform_data([t, rope_dim], -1, 1, attn_dtype)
    sin = gen_uniform_data([t, rope_dim], -1, 1, attn_dtype)
    wo_a = gen_uniform_data([n_g, n_q * d // n_g, o_lora_rank], -1, 1, attn_dtype)
    wo_b = gen_uniform_data([n_g * o_lora_rank, h], -1, 1, attn_dtype)
    residual = gen_uniform_data([t, hc, h], -1, 1, hc_dtype)
    post = gen_uniform_data([t, hc], -1, 1, hc_dtype)
    comb = gen_uniform_data([t, hc, hc], -1, 1, hc_dtype)

    inputs = [x, cos, sin, wo_a, wo_b, residual, post, comb]
    rope_res, bmm_res, mm_res, y = compute_attention_hc_post(inputs, params)
    return inputs, rope_res, bmm_res, mm_res, y


def do_attention_hc_post_func(inputs, params, golden_list):
    """
    INPUT 0	    x	           DT_BF16	 (t, n_q, dim)	                    ND
    INPUT 1	    cos	           DT_BF16	 (t, rope_dim)	                    ND
    INPUT 2	    sin	           DT_BF16	 (t, rope_dim)	                    ND
    INPUT 3	    wo_a	       DT_BF16	 (n_g, n_q*dim // ng, o_lora_rank)	ND
    INPUT 4	    wo_b	       DT_BF16	 (n_g * o_lora_rank, h)	            NZ
    INPUT 5	    residual	   DT_FP32	 (t, hc, h)	                        ND
    INPUT 6	    post	       DT_FP32	 (t, hc)	                        ND
    INPUT 7	    comb	       DT_FP32	 (t, hc, hc)	                    ND
    OUTPUT 0	y              DT_BF16	 (t, hc, h)	                        ND
    """
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    x = inputs[0].npu()
    cos = inputs[1].npu()
    sin = inputs[2].npu()
    wo_a = inputs[3].npu()
    wo_b = inputs[4].npu()
    wo_b_nz = torch_npu.npu_format_cast(wo_b, torch_npu.Format.FRACTAL_NZ)
    residual = inputs[5].npu()
    post = inputs[6].npu()
    comb = inputs[7].npu()

    t = params.get("t")
    h = params.get("h", 4096)
    hc = params.get("hc", 4)
    attn_dtype = params.get("attn_dtype", torch.bfloat16)

    # define npu outputs
    y = torch.zeros([t, hc, h]).to(attn_dtype).npu()

    x_pto = pypto.from_torch(x, dynamic_axis=[0], name="x")
    cos_pto = pypto.from_torch(cos, dynamic_axis=[0], name="cos")
    sin_pto = pypto.from_torch(sin, dynamic_axis=[0], name="sin")
    wo_a_pto = pypto.from_torch(wo_a, name="wo_a")
    wo_b_pto = pypto.from_torch(wo_b_nz, name="wo_b")
    re_pto = pypto.from_torch(residual, dynamic_axis=[0], name="residual")
    post_pto = pypto.from_torch(post, dynamic_axis=[0], name="post")
    comb_pto = pypto.from_torch(comb, dynamic_axis=[0], name="comb")

    y_pto = pypto.from_torch(y, dynamic_axis=[0], name="y")

    pto_inputs = [
        x_pto,
        cos_pto,
        sin_pto,
        wo_a_pto,
        wo_b_pto,
        re_pto,
        post_pto,
        comb_pto,
    ]
    pto_outputs = [y_pto]

    tile_config = AttnHcPostConfig(
        unroll_list=[128, 64, 32, 16, 8, 1],
        c1_tile=[[64, 64], [64, 64], [512, 512]],
        c2_tile=[[128, 128], [128, 128], [256, 256]],
    )

    # call main function
    attention_hc_post_kernel(*pto_inputs, *pto_outputs, tile_config)

    torch_npu.npu.synchronize()
    compare(y.cpu(), golden_list[3], "y", atol=0.0001, rtol=0.005)


def do_attention_hc_post_func_torch_graph(inputs, params, golden_list):
    """
    INPUT 0	    x	           DT_BF16	 (t, n_q, dim)	                    ND
    INPUT 1	    cos	           DT_BF16	 (t, rope_dim)	                    ND
    INPUT 2	    sin	           DT_BF16	 (t, rope_dim)	                    ND
    INPUT 3	    wo_a	       DT_BF16	 (n_g, n_q*dim // ng, o_lora_rank)	ND
    INPUT 4	    wo_b	       DT_BF16	 (n_g * o_lora_rank, h)	            NZ
    INPUT 5	    residual	   DT_FP32	 (t, hc, h)	                        ND
    INPUT 6	    post	       DT_FP32	 (t, hc)	                        ND
    INPUT 7	    comb	       DT_FP32	 (t, hc, hc)	                    ND
    OUTPUT 0	y              DT_BF16	 (t, hc, h)	                        ND
    """
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    x = inputs[0].npu()
    cos = inputs[1].npu()
    sin = inputs[2].npu()
    wo_a = inputs[3].npu()
    wo_b = inputs[4].npu()
    wo_b_nz = torch_npu.npu_format_cast(wo_b, torch_npu.Format.FRACTAL_NZ)
    residual = inputs[5].npu()
    post = inputs[6].npu()
    comb = inputs[7].npu()

    t = params.get("t")
    h = params.get("h", 4096)
    hc = params.get("hc", 4)
    attn_dtype = params.get("attn_dtype", torch.bfloat16)

    # define npu outputs
    y = torch.zeros([t, hc, h]).to(attn_dtype).npu()

    model = torch.compile(AttentionHcPost(), backend="eager", dynamic=True)

    # capture model
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        model(x, cos, sin, wo_a, wo_b_nz, residual, post, comb, y)

    for i in range(5):
        g.replay()
        pypto.runtime._device_synchronize()  # 内部接口，不推荐使用

    compare(y.cpu(), golden_list[3], "y", atol=0.0001, rtol=0.005)


def get_t_config(case_name: str):
    test_case_config = {
        "test_attention_hc_post_decode_min": 1,
        "test_attention_hc_post_decode_mid": 16,
        "test_attention_hc_post_decode_max": 64,
        "test_attention_hc_post_prefill_mid": 1024,
        "test_attention_hc_post_prefill_max": 8192,
        "test_attention_hc_post_prefill_general": 255,
    }
    return test_case_config.get(case_name)


def do_attention_hc_post_entry(case_name: str, is_torch_graph: bool = False):
    params = {
        "n_q": 64,
        "d": 512,
        "rope_dim": 64,
        "n_g": 8,
        "o_lora_rank": 1024,
        "h": 4096,
        "hc": 4,
        "attn_dtype": torch.bfloat16,
        "hc_dtype": torch.float32,
    }
    params["t"] = get_t_config(case_name)
    if not params:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    inputs, rope_golden, bmm_golden, mm_golden, y_golden = gen_attention_hc_post_golden(
        params
    )

    if is_torch_graph:
        print("\n =============== torch graph ====================")
        do_attention_hc_post_func_torch_graph(
            inputs, params, [rope_golden, bmm_golden, mm_golden, y_golden]
        )
    else:
        print("\n =============== st ====================")
        do_attention_hc_post_func(
            inputs, params, [rope_golden, bmm_golden, mm_golden, y_golden]
        )

    return True


def test_attention_hc_post_decode_min():
    """
    decode最小t值st用例
    """
    do_attention_hc_post_entry(
        "test_attention_hc_post_decode_min", is_torch_graph=False
    )


def test_attention_hc_post_decode_mid():
    """
    decode中等t值st用例
    """
    do_attention_hc_post_entry(
        "test_attention_hc_post_decode_mid", is_torch_graph=False
    )


def test_attention_hc_post_decode_max():
    """
    decode最大t值st用例
    """
    do_attention_hc_post_entry(
        "test_attention_hc_post_decode_max", is_torch_graph=False
    )


def test_attention_hc_post_prefill_mid():
    """
    prefill中等t值st用例
    """
    do_attention_hc_post_entry(
        "test_attention_hc_post_prefill_mid", is_torch_graph=False
    )


def test_attention_hc_post_prefill_max():
    """
    prefill最大t值st用例
    """
    do_attention_hc_post_entry(
        "test_attention_hc_post_prefill_max", is_torch_graph=False
    )


def test_attention_hc_post_prefill_general():
    """
    prefill泛化t值aclgraph用例
    """
    do_attention_hc_post_entry(
        "test_attention_hc_post_prefill_general", is_torch_graph=True
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    test_attention_hc_post_decode_min()
