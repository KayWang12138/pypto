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
"""
'''
'''
import os
from pathlib import Path
import math
import pytest
import torch
import torch_npu
import pypto
import logging
import numpy as np
from mla_prolog_impl import mla_prolog_v4_in
from utils.compare import compare

torch.manual_seed(5)

def prep_env():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True


def rms_norm_new(x, eps=1e-6):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]

    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = torch.sum(mean_res, dim=-1, keepdims=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res = x_f32 / reduce_sqrt

    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def rms_norm(x, gamma, eps=1e-6):
    x_dtype = x.dtype
    mean_coff = 1.0 / x.shape[-1]
    gamma = gamma.to(torch.float32)
    x_f32 = x.to(torch.float32)
    square = x_f32 * x_f32
    mean_res = square * mean_coff

    reduce_sum = torch.sum(mean_res, dim=-1, keepdims=True) + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt

    res = res_div * gamma

    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


def apply_rotary_pos_emb_v2(q, k, cos, sin, unsqueeze_dim=1):
    input_dtype = q.dtype
    if input_dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
    if cos.dtype != torch.float32:
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

    cos = torch.unsqueeze(cos, dim=unsqueeze_dim)  # [b,s,1,qk_d]
    sin = torch.unsqueeze(sin, dim=unsqueeze_dim)  # [b,s,1,qk_d]
    logging.debug("expand sin.shape: %s", sin.shape)
    logging.debug("expand cos.shape: %s", cos.shape)

    t, h, d = q.shape
    q = q.reshape(t, h, d // 2, 2).permute(0, 1, 3, 2).reshape(t, h, d)  # [t,n,qk_d]

    t, h, d = k.shape
    k = k.reshape(t, h, d // 2, 2).permute(0, 1, 3, 2).reshape(t, h, d)  # [t,1,qk_d]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    if input_dtype != torch.float32:
        q_embed, k_embed = q_embed.to(input_dtype), k_embed.to(input_dtype)
    return q_embed, k_embed


def tensor_to_file(t: torch.Tensor, output: Path):
    with open(str(output), "wb") as f:
        dtype = t.dtype
        if dtype == torch.bfloat16:
            dtype = torch.int16
        for each in t:
            f.write(each.view(dtype).cpu().numpy().tobytes())


def mla_prolog_compute(inputs):
    dtype = inputs.get("dtype")
    x = inputs.get("x")
    wq_a = inputs.get("wq_a")
    wq_b = inputs.get("wq_b")
    qk_rope_head_dim = inputs.get("qk_rope_head_dim")
    gamma_cq = inputs.get("gamma_cq")
    gamma_ckv = inputs.get("gamma_ckv")
    w_kv = inputs.get("w_kv")
    cos = inputs.get("cos")
    sin = inputs.get("sin")

    t, h = x.shape
    qk_rope_head_dim = cos.shape[1]
    head_dim = w_kv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim

    ''' q '''
    q_a_proj = torch.matmul(x, wq_a)
    q_a_layernorm = rms_norm(q_a_proj, gamma_cq)
    logging.debug("q_a_layernorm.shape: %s %s", q_a_layernorm.shape, q_a_layernorm.dtype)

    q_b_proj = torch.matmul(q_a_layernorm, wq_b)
    logging.debug("q_b_proj.shape: %s %s", q_b_proj.shape, q_b_proj.dtype)
    q_reshape = q_b_proj.reshape(t, num_heads, head_dim)
    q_reshape = rms_norm_new(q_reshape)
    logging.debug("q_reshape.shape: %s %s", q_reshape.shape, q_reshape.dtype)

    """ kv """
    kv_a_proj = torch.matmul(x, w_kv)
    kv_a_proj_norm = rms_norm(kv_a_proj, gamma_ckv)
    logging.debug("kv_a_proj.shape: %s %s", kv_a_proj.shape, kv_a_proj.dtype)
    kv_reshape = kv_a_proj_norm.reshape(t, head_dim)
    logging.debug("kv_reshape.shape: %s %s", kv_reshape.shape, kv_reshape.dtype)

    """ rope"""
    q_pe = q_reshape[:, :, -qk_rope_head_dim:]

    k_pe = kv_reshape[:, -qk_rope_head_dim:]
    k_pe_r = k_pe.reshape(t, 1, qk_rope_head_dim)

    q_embed, k_embed = apply_rotary_pos_emb_v2(q_pe, k_pe_r, cos, sin, 1)
    k_embed_r = k_embed.reshape(t, qk_rope_head_dim)
    q_out = torch.concat([q_reshape[:, :, :-qk_rope_head_dim], q_embed], -1)
    kv_out = torch.concat([kv_reshape[:, :-qk_rope_head_dim], k_embed_r], -1)

    return q_out, kv_out, q_a_layernorm

def gen_mla_prolog_input_data(params, dtypes, is_nz=False):
    dtype = dtypes
    logging.debug(f"gen_mla_prolog_input_data dtype:{dtype}")
    t = params.get("t")
    h = params.get("h")
    n = params.get("num_heads")
    q_lora_rank = params.get("q_lora_rank")
    head_dim = params.get("head_dim")
    rope_head_dim = params.get("qk_rope_head_dim")

    x_shape = [t, h]
    wq_a_shape = [h, q_lora_rank]
    gamma_cq_shape = [q_lora_rank]
    gamma_ckv_shape = [head_dim]
    wq_b_shape = [q_lora_rank, n * head_dim]
    wkv_shape = [h, head_dim]
    cos_shape = [t, rope_head_dim]
    logging.debug("x shape is %s", x_shape)
    logging.debug("wq_a shape is %s", wq_a_shape)
    logging.debug("wq_b shape is %s", wq_b_shape)
    logging.debug("cos sin shape is %s %s", cos_shape)
    logging.debug("w_kv shape is %s", wkv_shape)
    logging.debug("cgamma_cq shape is %s", gamma_cq_shape)
    logging.debug("cgamma_ckv is %s", gamma_ckv_shape)

    res = [None] * 8
    x = torch.empty(x_shape, dtype=dtype).uniform_(-1, 1)
    res[0] = x

    wq_a = torch.empty(wq_a_shape, dtype=dtype).uniform_(-0.1, 0.1)
    wq_b = torch.empty(wq_b_shape, dtype=dtype).uniform_(-0.1, 0.1)
    w_kv = torch.empty(wkv_shape, dtype=dtype).uniform_(-0.1, 0.1)
    res[1] = wq_a
    res[2] = wq_b
    res[3] = w_kv

    gamma_cq = torch.empty(gamma_cq_shape, dtype=dtype).uniform_(-1, 1)
    gamma_ckv = torch.empty(gamma_ckv_shape, dtype=dtype).uniform_(-1, 1)
    res[4] = gamma_cq
    res[5] = gamma_ckv

    cos = torch.empty(cos_shape, dtype=dtype).uniform_(-1, 1)
    sin = torch.empty(cos_shape, dtype=dtype).uniform_(-1, 1)
    res[6] = cos
    res[7] = sin

    return res

def gen_mla_prolog_data(params, dtype, is_nz=False):
    logging.debug(f"gen_mla_prolog_data dtype: {dtype}")
    x, wq_a, wq_b, w_kv, gamma_cq, gamma_ckv, cos, sin = \
        gen_mla_prolog_input_data(params, dtype, is_nz)
    inputs = dict()
    inputs["x"] = x
    inputs["wq_a"] = wq_a
    inputs["wq_b"] = wq_b
    inputs["w_kv"] = w_kv
    inputs["cos"] = cos
    inputs["sin"] = sin
    inputs["gamma_cq"] = gamma_cq
    inputs["gamma_ckv"] = gamma_ckv

    if torch_npu.npu.is_available():
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.npu()
    
    q_out, kv_out, qr_out = mla_prolog_compute(inputs)
    outputs = {"q_golden": q_out, "kv_golden": kv_out, "qr_golden": qr_out}

    return inputs, outputs

def convert_pypto_to_torch_type(pypto_type):
    if pypto_type == pypto.DataType.DT_INT8:
        return torch.int8
    elif pypto_type == pypto.DataType.DT_INT32:
        return torch.int32
    elif pypto_type == pypto.DataType.DT_FP32:
        return torch.float32
    elif pypto_type == pypto.DataType.DT_FP16:
        return torch.float16
    elif pypto_type == pypto.DataType.DT_BF16:
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported pypto.DataType: {pypto_type}")


class MLA_MODEL(torch.nn.Module):
    def forward(self, token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv, output_q_data, output_kv_data, output_qr_data):
        return mla_prolog_v4_in(token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv, output_q_data, output_kv_data, output_qr_data)

def mla_prolog(params, input_tensors, golden_tensors, dtype, is_nz):
    d_type = pypto.DataType.DT_FP16 if dtype == pypto.DataType.DT_FP16 else pypto.DataType.DT_BF16
    t = params['t']
    n1 = params["num_heads"]
    h = params["h"]
    q_lora_rank = params["q_lora_rank"]
    qk_rope_head_dim = params["qk_rope_head_dim"]
    head_dim = params["head_dim"]

    token_x_shape = [t, h]
    wq_a_shape = [h, q_lora_rank]
    wq_b_shape = [q_lora_rank, n1 * head_dim]
    wkv_shape = [h, head_dim]
    rope_cos_shape = [t, qk_rope_head_dim]
    rmsnorm_gamma_cq_shape = [q_lora_rank]
    rmsnorm_gamma_ckv_shape = [head_dim]

    q_out_shape = [t, n1, head_dim]
    kv_out_shape = [t, head_dim]
    qr_out_shape = [t, q_lora_rank]
    
    if is_nz:
        wq_a_nz = torch_npu.npu_format_cast(input_tensors["wq_a"].reshape(wq_a_shape).npu().contiguous(), \
                                            torch_npu.Format.FRACTAL_NZ)
        wq_b_nz = torch_npu.npu_format_cast(input_tensors["wq_b"].reshape(wq_b_shape).npu().contiguous(), \
                                            torch_npu.Format.FRACTAL_NZ)
        wkv_nz = torch_npu.npu_format_cast(input_tensors["w_kv"].reshape(wkv_shape).npu().contiguous(), \
                                            torch_npu.Format.FRACTAL_NZ)
        input_tensors["wq_a"] = wq_a_nz
        input_tensors["wq_b"] = wq_b_nz
        input_tensors["w_kv"] = wkv_nz
    
    token_x = input_tensors["x"].reshape(token_x_shape).npu()
    wq_a = input_tensors["wq_a"].reshape(wq_a_shape).npu()
    wq_b = input_tensors["wq_b"].reshape(wq_b_shape).npu()
    wkv = input_tensors["w_kv"].reshape(wkv_shape).npu()
    rope_cos = input_tensors["cos"].reshape(rope_cos_shape).npu()
    rope_sin = input_tensors["sin"].reshape(rope_cos_shape).npu()
    gamma_cq = input_tensors["gamma_cq"].reshape(rmsnorm_gamma_cq_shape).npu()
    gamma_ckv = input_tensors["gamma_ckv"].reshape(rmsnorm_gamma_ckv_shape).npu()
    inputs = [token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv]

    output_q_data = torch.zeros(q_out_shape, dtype=convert_pypto_to_torch_type(d_type)).npu()
    output_kv_data = torch.zeros(kv_out_shape, dtype=convert_pypto_to_torch_type(d_type)).npu()
    output_qr_data = torch.zeros(qr_out_shape, dtype=convert_pypto_to_torch_type(d_type)).npu()
    outputs = [output_q_data, output_kv_data, output_qr_data]

    #capture model
    mla_prolog_model = torch.compile(MLA_MODEL(), backend="eager", dynamic=True)
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        output_q_data, output_kv_data, output_qr_data = mla_prolog_model(*inputs, *outputs)
    g.replay()
    pypto.runtime._device_synchronize()
    # golden data 
    golden1 = golden_tensors["q_golden"].reshape(q_out_shape)
    golden2 = golden_tensors["kv_golden"].reshape(kv_out_shape)
    golden3 = golden_tensors["qr_golden"].reshape(qr_out_shape)
    
    # compare
    print("q ================")
    compare(output_q_data.cpu(), golden1.cpu(), "qOut", 0.0001, 0.0078125, 0.005)
    print("kv ================")
    compare(output_kv_data.cpu(), golden2.cpu(), "kvOut", 0.0001, 0.0078125, 0.005)
    print("qr ================")
    compare(output_qr_data.cpu(), golden3.cpu(), "qrOut", 0.0001, 0.0078125, 0.005)


def test_t4_pa_nd_bf16():
    prep_env()
    params = {
        't': 4,
        'num_heads': 64,
        'h': 4096,
        'q_lora_rank': 1024,
        'head_dim': 512,
        'qk_rope_head_dim': 64,
    }
    dtype = pypto.DataType.DT_BF16
    is_nz = False
    input_tensors, golden_data = gen_mla_prolog_data(params, torch.bfloat16, is_nz)
    mla_prolog(params, input_tensors, golden_data, dtype, is_nz)


def test_t16_pa_nd_bf16():
    prep_env()
    params = {
        't': 16,
        'num_heads': 64,
        'h': 4096,
        'q_lora_rank': 1024,
        'head_dim': 512,
        'qk_rope_head_dim': 64,
    }
    dtype = pypto.DataType.DT_BF16
    is_nz = False
    input_tensors, golden_data = gen_mla_prolog_data(params, torch.bfloat16, is_nz)
    mla_prolog(params, input_tensors, golden_data, dtype, is_nz)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_t4_pa_nd_bf16()