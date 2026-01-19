#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
from quant_attention_post_impl import (
    npu_quant_attention_post,
    attention_post_quant_kernel,
    AttnPostQuantConfig,
)
from utils.golden.common_func import (
    apply_rotary_pos_emb,
    gen_uniform_data,
    quant_golden,
)
from utils.compare import compare


pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "quant_attn_post(Tensor atten_res, Tensor cos, Tensor sin, Tensor wo_a, Tensor wo_b, Tensor wo_b_scale) -> (Tensor)"
)


@torch.library.impl(pyptolib, "quant_attn_post", "Meta")
def quant_attn_post(atten_res, cos, sin, wo_a, wo_b, wo_b_scale):
    y = torch.empty(
        [atten_res.size(0), wo_b.size(1)],
        dtype=atten_res.dtype,
        device=atten_res.device,
    )
    return y


@torch.library.impl(pyptolib, "quant_attn_post", "NPU")
def quant_attn_post(atten_res, cos, sin, wo_a, wo_b, wo_b_scale):
    return npu_quant_attention_post(atten_res, cos, sin, wo_a, wo_b, wo_b_scale)


class QuantAttentionPost(torch.nn.Module):
    def forward(self, attn_res, cos, sin, wo_a, wo_b, wo_b_scale):
        for _ in range(20):
            torch.add(attn_res, 0)
        return torch.ops.pypto.quant_attn_post(
            attn_res, cos, sin, wo_a, wo_b, wo_b_scale
        )


def compute_quant_attention_post(inputs, params):
    atten_res = inputs[0]
    cos = inputs[1]
    sin = inputs[2]
    wo_a = inputs[3]
    wo_b = inputs[4]
    wo_b_scale = inputs[5]

    t = params.get("t")
    n_q = params.get("n_q")
    d = params.get("d")
    n_g = params.get("n_g")
    o_lora_rank = params.get("o_lora_rank")
    rope_dim = params.get("rope_dim")

    atten_dtype = params.get("atten_dtype")

    rope_in = atten_res[:, :, (d - rope_dim) :]  # (t, n_q, rope_dim), bf16
    # (t, n_q, d - rope_dim), bf16
    nope_res = atten_res[:, :, 0 : (d - rope_dim)]
    rope_res = apply_rotary_pos_emb(rope_in, cos, sin)  # (t, n_q, rope_dim), bf16
    atten_res_new = torch.cat((nope_res, rope_res), dim=-1)

    # batch_matmul
    mm1_left_trans = atten_res_new.reshape(t, n_g, n_q * d // n_g).transpose(
        1, 0
    )  # (n_g, t, n_q * d // n_g)
    # (n_g, t, n_q * d // n_g) @ (n_g, n_q * d // n_g, o_lora_rank) = (n_g, t, o_lora_rank)
    bmm_1_res = torch.bmm(mm1_left_trans.to(torch.float32), wo_a.to(torch.float32)).to(
        torch.bfloat16
    )
    bmm_res = bmm_1_res.transpose(1, 0)  # (t, n_g, o_lora_rank)

    bmm_reshape = bmm_res.reshape(t, n_g * o_lora_rank)  # (t, n_g * o_lora_rank)

    # quant:  (t, n_g * o_lora_rank), (t, 1)
    bmm_int8, bmm_scale = quant_golden(bmm_reshape)

    # (t, n_g * o_lora_rank) @ (n_g * o_lora_rank, h) = (t, n_q, h)
    mm_res = torch.mm(bmm_int8.to(torch.int32), wo_b.to(torch.int32)).to(torch.float32)

    res = mm_res * wo_b_scale.t() * bmm_scale
    res_b16 = res.to(atten_dtype)

    return rope_res, bmm_res, mm_res, res_b16


def gen_quant_attention_post_golden(params):
    torch.manual_seed(42)
    t = params.get("t")
    n_q = params.get("n_q")
    d = params.get("d")
    rope_dim = params.get("rope_dim")
    n_g = params.get("n_g")
    o_lora_rank = params.get("o_lora_rank")
    h = params.get("h")
    atten_dtype = params.get("atten_dtype")

    attn_res = gen_uniform_data([t, n_q, d], -1, 1, atten_dtype)
    cos = gen_uniform_data([t, rope_dim], -1, 1, atten_dtype)
    sin = gen_uniform_data([t, rope_dim], -1, 1, atten_dtype)
    wo_a = gen_uniform_data([n_g, n_q * d // n_g, o_lora_rank], -0.1, 0.1, atten_dtype)
    wo_b_ori = gen_uniform_data([n_g * o_lora_rank, h], -1, 1, torch.int8)
    wo_b, wo_b_scale = quant_golden(wo_b_ori.t())
    wo_b = wo_b.t()
    inputs = [attn_res, cos, sin, wo_a, wo_b, wo_b_scale]
    rope_res, bmm_res, mm_res, res_b16 = compute_quant_attention_post(inputs, params)
    return inputs, rope_res, bmm_res, mm_res, res_b16


def do_quant_attention_post_func(inputs, params, golden_list):
    """
    atten_res: (t, n_q, d), bf16
    cos: (t, rope_dim), bf16
    sin: (t, rope_dim), bf16
    wo_a: (n_g, n_q * d // n_g, o_lora_rank), bf16
    wo_b: (n_g * o_lora_rank, h), int8
    wo_b_scale: (h, 1), fp32
    """
    torch_npu.npu.config.allow_internal_format = True
    # rope + batch_matmul + matmul
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    atten_res = inputs[0].npu()
    cos = inputs[1].npu()
    sin = inputs[2].npu()
    wo_a = inputs[3].npu()
    wo_b = inputs[4].npu()
    # nz not use now
    # wo_b_nz = torch_npu.npu_format_cast(wo_b, torch_npu.Format.FRACTAL_NZ)
    wo_b_scale = inputs[5].npu()

    t = params.get("t")
    h = params.get("h")
    atten_dtype = params.get("atten_dtype", torch.bfloat16)

    # define npu outputs
    hidden_states = torch.zeros([t, h]).to(atten_dtype).npu()

    atten_res_pto = pypto.from_torch(atten_res, dynamic_axis=[0], name="atten_res")
    cos_pto = pypto.from_torch(cos, dynamic_axis=[0], name="cos")
    sin_pto = pypto.from_torch(sin, dynamic_axis=[0], name="sin")
    wo_a_pto = pypto.from_torch(wo_a, name="wo_a")
    wo_b_pto = pypto.from_torch(wo_b, name="wo_b")
    wo_b_scale_pto = pypto.from_torch(wo_b_scale, name="wo_b_scale")

    hidden_states_pto = pypto.from_torch(
        hidden_states, dynamic_axis=[0], name="hidden_states"
    )

    pto_inputs = [atten_res_pto, cos_pto, sin_pto, wo_a_pto, wo_b_pto, wo_b_scale_pto]
    pto_outputs = [hidden_states_pto]

    tile_config = AttnPostQuantConfig(
        unroll_list=[128, 64, 32, 16, 8, 1],
        c1_tile=[[64, 64], [64, 64], [512, 512]],
        c2_tile=[[128, 128], [128, 128], [256, 256]],
    )

    # call main function
    attention_post_quant_kernel(*pto_inputs, *pto_outputs, tile_config)

    torch_npu.npu.synchronize()
    compare(
        hidden_states.cpu(), golden_list[3], "hidden_states", atol=0.0001, rtol=0.007825
    )


def do_quant_attention_post_func_torch_graph(inputs, golden_list):
    """
    atten_res: (t, n_q, d), bf16
    cos: (t, rope_dim), bf16
    sin: (t, rope_dim), bf16
    wo_a: (n_g, n_q * d // n_g, o_lora_rank), bf16
    wo_b: (n_g * o_lora_rank, h), int8
    wo_b_scale: (h, 1), fp32
    """
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    # define npu inputs
    atten_res_npu = inputs[0].npu()
    cos_npu = inputs[1].npu()
    sin_npu = inputs[2].npu()
    wo_a_npu = inputs[3].npu()
    wo_b_npu = inputs[4].npu()
    wo_b_scale_npu = inputs[5].npu()
    # nz not use now
    # wo_b_nz = torch_npu.npu_format_cast(wo_b_npu, torch_npu.Format.FRACTAL_NZ)

    compiler_config = CompilerConfig()
    compiler_config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    model = torch.compile(
        QuantAttentionPost(), dynamic=False, fullgraph=True, backend=npu_backend
    )

    hidden_states = model(
        atten_res_npu, cos_npu, sin_npu, wo_a_npu, wo_b_npu, wo_b_scale_npu
    )
    pypto.runtime._device_synchronize()

    compare(
        hidden_states.cpu(), golden_list[3], "hidden_states", atol=0.0001, rtol=0.005
    )


def get_quant_attention_post_config(case_name: str):
    test_case_config = {
        "test_quant_attention_post_decode_min": 1,
        "test_quant_attention_post_decode_mid": 16,
        "test_quant_attention_post_decode_max": 64,
        "test_quant_attention_post_prefill_mid": 8192,
        "test_quant_attention_post_prefill_max": 65536,
        "test_quant_attention_post_graph_func": 255,
    }
    return test_case_config.get(case_name)


def do_quant_attention_post_entry(case_name: str, is_torch_graph: bool = False):
    params = {
        "n_q": 64,
        "d": 512,
        "rope_dim": 64,
        "n_g": 8,
        "o_lora_rank": 1024,
        "h": 4096,
        "hc": 4,
        "atten_dtype": torch.bfloat16,
    }
    params["t"] = get_quant_attention_post_config(case_name)
    if not params:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False

    inputs, rope_golden, bmm_golden, mm_golden, res_golden = (
        gen_quant_attention_post_golden(params)
    )

    if is_torch_graph:
        print("\n =============== torch graph ====================")
        do_quant_attention_post_func_torch_graph(
            inputs, [rope_golden, bmm_golden, mm_golden, res_golden]
        )
    else:
        print("\n =============== st ====================")
        do_quant_attention_post_func(
            inputs, params, [rope_golden, bmm_golden, mm_golden, res_golden]
        )

    return True


def test_quant_attention_post_decode_min():
    """
    quant_attention_post decode最小t值st用例
    """
    do_quant_attention_post_entry(
        "test_quant_attention_post_decode_min", is_torch_graph=False
    )


def test_quant_attention_post_decode_mid():
    """
    quant_attention_post docode中等t值st用例
    """
    do_quant_attention_post_entry(
        "test_quant_attention_post_decode_mid", is_torch_graph=False
    )


def test_quant_attention_post_decode_max():
    """
    quant_attention_post decode最大t值st用例
    """
    do_quant_attention_post_entry(
        "test_quant_attention_post_decode_max", is_torch_graph=False
    )


def test_quant_attention_post_prefill_mid():
    """
    quant_attention_post prefill中等t值st用例
    """
    do_quant_attention_post_entry(
        "test_quant_attention_post_prefill_mid", is_torch_graph=False
    )


def test_quant_attention_post_prefill_max():
    """
    quant_attention_post prefill最大t值st用例
    """
    do_quant_attention_post_entry(
        "test_quant_attention_post_prefill_max", is_torch_graph=False
    )


def test_quant_attention_post_graph_func():
    """
    attention post v4 torch graph
    """
    do_quant_attention_post_entry(
        "test_quant_attention_post_graph_func", is_torch_graph=True
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    test_quant_attention_post_decode_min()
