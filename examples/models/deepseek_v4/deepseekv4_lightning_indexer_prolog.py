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
import torch
import torch_npu
import pypto
import os
import pytest
import math
import logging
from lightning_indexer_prolog_impl import (
    IndexerPrologInput, IndexerPrologOutput, IndexerPrologAttr, IndexerPrologConfigs,
    lightning_indexer_prolog)
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
from utils.compare import compare


class IP(torch.nn.Module):
    def forward(self, x, q_norm, w_qb, w_proj, cos_idx_rope, sin_idx_rope, hadamard_q,
                                    q_bf16, weights):
        lighting_indexer_prolog_dyn(x, q_norm, w_qb, w_proj, cos_idx_rope, sin_idx_rope, hadamard_q,
                                    q_bf16, weights)


def gen_dims(params):
    dims = {}
    dims["b"] = params["b"]
    dims["t"] = params["b"] * params["s1"]
    dims["h"] = 4096
    dims["q_lora_rank"] = 1024
    dims["idx_head_dim"] = 128
    dims["idx_n_heads"] = 32
    dims["rope_head_dim"] = 64
    return dims


def gen_inputs(dims, dtype=torch.bfloat16):
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    rope_head_dim = dims["rope_head_dim"]

    x = torch.empty((b, s, h), dtype=dtype).uniform_(-1, 1)
    q_norm = torch.empty((b, s, q_lora_rank), dtype=dtype).uniform_(-1, 1)
    w_idx_qb = torch.empty((q_lora_rank, n * d), dtype=dtype).uniform_(-1, 1)
    w_idx_proj = torch.empty((h, n), dtype=dtype).uniform_(-1, 1)

    random_angles = (torch.rand(b, s, rope_head_dim, dtype=torch.float32) * 2 * torch.pi)
    cos = torch.cos(random_angles).to(dtype)
    sin = torch.sin(random_angles).to(dtype)

    hadamard_q = torch.empty((d, d), dtype=dtype).uniform_(-1, 1)  # (128, 128)

    return {
        "token_x": x,  # input0, bf16
        "q_norm": q_norm,  # input1, bf16
        "w_idx_qb": w_idx_qb,  # input3, int8
        "w_idx_proj": w_idx_proj,  # input6, bf16
        "cos_idx_rope": cos,  # input9, bf16
        "sin_idx_rope": sin,  # input10, bf16
        "hadamard_q": hadamard_q,  # input11, bf16
    }


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def single_rope(x, cos_in, sin_in):
    logging.info("Entering into single_rope")
    # x: (b, s, n, d), cos_in: (b, s, d), sin_in: (b, s, d)
    x_dtype = x.dtype
    b, s, n, d = x.shape
    x_cast = x.to(torch.float32)
    cos_cast = cos_in.to(torch.float32)
    sin_cast = sin_in.to(torch.float32)
    cos_re = cos_cast.unsqueeze(2)  # (b, s, 1, d)
    sin_re = sin_cast.unsqueeze(2)  # (b, s, 1, d)
    res = x_cast * cos_re + rotate_half(x_cast) * sin_re  # (b, s, n, d)
    return res.to(x_dtype)


def indexer_prolog(inputs: dict, dims: dict):
    # input
    b, t, n, d = dims["b"], dims["t"], dims["idx_n_heads"], dims["idx_head_dim"]
    s = t // b

    rope_head_dim = dims["rope_head_dim"]
    x = inputs["token_x"]  # (b, s, h)
    q_norm = inputs["q_norm"]  # (b, s, q_lora_rank), bf16
    w_idx_qb = inputs["w_idx_qb"]  # (q_lora_rank, n * d), bf16
    w_idx_proj = inputs["w_idx_proj"]  # (h, n)
    cos = inputs["cos_idx_rope"]  # (b, s, rope_head_dim)
    sin = inputs["sin_idx_rope"]  # (b, s, rope_head_dim)
    hadamard_q = inputs["hadamard_q"]  # (d, d)
    x_dtype = x.dtype

    # calculate
    q_fp32 = torch.matmul(q_norm.to(torch.float32), w_idx_qb.to(torch.float32))  # (b, s, n * d)
    q_bf16 = q_fp32.reshape(b, s, n, d).to(torch.bfloat16)
    q_rope, q_nope = torch.split(q_bf16, [rope_head_dim, d - rope_head_dim], dim=-1)
    q_rope = single_rope(q_rope, cos, sin)
    q = torch.cat([q_rope, q_nope], dim=-1)
    # hadamard
    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    q = torch.matmul(q.to(torch.float32), hadamard_q.to(torch.float32)).to(x_dtype)  # (b, s, n, d)
    
    # matmul use float32 for arm, arm平台matmul在bfloat16数据类型下表现跟x86不一致，通过升精度保证正确性
    weights = torch.matmul(x.to(torch.float32), \
        w_idx_proj.to(torch.float32)).to(x_dtype).to(torch.float32)  # (b, s, n)
    weights = weights * (n ** -0.5) * (d ** -0.5)
    weights = weights.to(torch.bfloat16)

    # output dtype: bf16, bf16
    outputs = {"q_golden": q, "weights": weights}
    return outputs


def gen_data(case_name):
    if case_name.startswith("LightningIndexerPrologSTest.b1_s1_1"):
        params = {
            "b": 1,
            "s1": 1
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b4_s1_4"):
        params = {
            "b": 4,
            "s1": 4
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b8_s1_8"):
        params = {
            "b": 8,
            "s1": 8
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b2_s1_4k"):
        params = {
            "b": 2,
            "s1": 1024 * 4
        }
    elif case_name.startswith("LightningIndexerPrologSTest.b1_s1_8193"):
        params = {
            "b": 1,
            "s1": 8193
        }
    else:
        raise Exception(f"Can't get func to gen golden, Case({case_name})")

    seed = 0
    # PyTorch 随机数生成器
    torch.manual_seed(seed)
    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.bfloat16)
    outputs = indexer_prolog(inputs, dims)
    return dims, inputs, outputs


def gen_zero_tensor(t):
    return torch.zeros_like(t).npu()

def check_input_output_shape_dtype(x: torch.tensor,
                                    q_norm: torch.tensor,
                                    w_qb: torch.tensor,
                                    w_proj: torch.tensor,
                                    cos_idx_rope: torch.tensor,
                                    sin_idx_rope: torch.tensor,
                                    hadamard_q: torch.tensor,
                                    q_bf16: torch.tensor,
                                    weights: torch.tensor):
    assert x.size(1) == 4096 and x.dim() == 2, f"expected x axis1 4096, x dim num 2"
    assert q_norm.size(1) == 1024 and q_norm.dim() == 2, \
        f"expected q_norm axis1 1024, q_norm dim num 2"
    assert w_qb.size(0) == 1024 and w_qb.size(1) == 4096 and w_qb.dim() == 2, \
        f"expected w_qb axis0 1024, w_qb axis1 4096, w_qb dim num 2"
    assert w_proj.size(0) == 4096 and w_proj.size(1) == 32 and w_proj.dim() == 2, \
        f"expected w_proj axis0 4096, w_proj axis1 32, w_proj dim num 2"
    assert cos_idx_rope.size(1) == 64 and cos_idx_rope.dim() == 2, \
        f"expected cos_idx_rope axis1 64, cos_idx_rope dim num 2"
    assert sin_idx_rope.size(1) == 64 and sin_idx_rope.dim() == 2, \
        f"expected sin_idx_rope axis1 64, sin_idx_rope dim num 2"
    assert hadamard_q.size(0) == 128 and hadamard_q.size(1) == 128 and hadamard_q.dim() == 2, \
        f"expected hadamard_q axis0 128, hadamard_q axis1 128, hadamard_q dim num 2"
    assert q_bf16.size(1) == 32 and q_bf16.size(2) == 128 and q_bf16.dim() == 3, \
        f"expected q_bf16 axis1 32, q_bf16 axis2 128, q_bf16 dim num 3"
    assert weights.size(1) == 32 and weights.dim() == 2, \
        f"expected weights axis1 32, weights dim num 2"

    
    assert x.dtype == torch.bfloat16, f"x.dtype is {x.dtype}, expected torch.bfloat16"
    assert q_norm.dtype == torch.bfloat16, f"q_norm.dtype is {q_norm.dtype}, expected torch.bfloat16"
    assert w_qb.dtype == torch.bfloat16, f"w_qb.dtype is {w_qb.dtype}, expected torch.bfloat16"
    assert w_proj.dtype == torch.bfloat16, f"w_proj.dtype is {w_proj.dtype}, expected torch.bfloat16"
    assert cos_idx_rope.dtype == torch.bfloat16, f"cos_idx_rope.dtype is {cos_idx_rope.dtype}, expected torch.bfloat16"
    assert sin_idx_rope.dtype == torch.bfloat16, f"sin_idx_rope.dtype is {sin_idx_rope.dtype}, expected torch.bfloat16"
    assert hadamard_q.dtype == torch.bfloat16, f"hadamard_q.dtype is {hadamard_q.dtype}, expected torch.bfloat16"
    assert q_bf16.dtype == torch.bfloat16, f"q_bf16.dtype is {q_bf16.dtype}, expected torch.bfloat16"
    assert weights.dtype == torch.bfloat16, f"weights.dtype is {weights.dtype}, expected torch.bfloat16"
    

@allow_in_graph
def lighting_indexer_prolog_dyn(x: torch.tensor,
                                q_norm: torch.tensor,
                                w_qb: torch.tensor,
                                w_proj: torch.tensor,
                                cos_idx_rope: torch.tensor,
                                sin_idx_rope: torch.tensor,
                                hadamard_q: torch.tensor,
                                q_bf16: torch.tensor,
                                weights: torch.tensor):
    
    check_input_output_shape_dtype(x, q_norm, w_qb, w_proj, cos_idx_rope, sin_idx_rope, hadamard_q, q_bf16, weights)
    configs = IndexerPrologConfigs(
        q_linear=[16, 16, 128, 256, 256, 256],
        q_hd=[32, 32, 128, 128, 256, 256],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    input_tensors = {
        x: [0],
        q_norm: [0],
        w_qb: [],
        w_proj: [],
        cos_idx_rope: [0],
        sin_idx_rope: [0],
        hadamard_q: []
    }
    output_tensors = {
        q_bf16: [0],
        weights: [0]
    }
    if not isinstance(x, FakeTensor):
        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]
        lightning_indexer_prolog(*pto_inputs, *pto_outputs, configs)


def do_test_lighting_indexer_prolog(case_name, configs):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(1)

    print(f"=== run test case: {case_name} ===")

    dims, inputs_data, golden_data = gen_data(case_name)

    t = dims["t"]
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    idx_head_dim = dims["idx_head_dim"]
    head_num = dims["idx_n_heads"]
    rope_head_dim = dims["rope_head_dim"]

    torch_npu.npu.config.allow_internal_format = True

    x=inputs_data["token_x"].npu().reshape(t, h)
    q_norm=inputs_data["q_norm"].npu().reshape(t, q_lora_rank)
    w_qb=torch_npu.npu_format_cast(inputs_data["w_idx_qb"].npu().contiguous(), torch_npu.Format.FRACTAL_NZ)
    w_proj=torch_npu.npu_format_cast(
        inputs_data["w_idx_proj"].npu().contiguous(), torch_npu.Format.FRACTAL_NZ)
    cos_idx_rope=inputs_data["cos_idx_rope"].npu().reshape(t, rope_head_dim)
    sin_idx_rope=inputs_data["sin_idx_rope"].npu().reshape(t, rope_head_dim)
    hadamard_q=inputs_data["hadamard_q"].npu()

    q_golden = golden_data["q_golden"].reshape(t, head_num, idx_head_dim)
    weights_golden = golden_data["weights"].reshape(t, head_num)

    q_bf16 =gen_zero_tensor(q_golden)
    weights=gen_zero_tensor(weights_golden)
    model = torch.compile(IP(), backend="eager", dynamic=True)

    # capture model
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        model(x, q_norm, w_qb, w_proj, cos_idx_rope, sin_idx_rope, hadamard_q,
                                    q_bf16, weights)

    g.replay()
    pypto.runtime._device_synchronize()#内部接口，不推荐使用

    compare(q_bf16.cpu(), q_golden, "q_bf16", 0.0001, 0.0078125, 0.001)
    compare(weights.cpu(), weights_golden, "weights", 0.0001, 0.0078125, 0.001)

    print(f"=== {case_name}: PASS ===")


def test_b1_s1_1():
    configs = IndexerPrologConfigs(
        q_linear=[16, 16, 128, 256, 256, 256],
        q_hd=[32, 32, 128, 128, 256, 256],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lighting_indexer_prolog("LightningIndexerPrologSTest.b1_s1_1", configs)


# @pytest.mark.skip(reason="large test case")
def test_b4_s1_4():
    configs = IndexerPrologConfigs(
        q_linear=[16, 16, 128, 256, 256, 256],
        q_hd=[32, 32, 128, 128, 256, 256],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lighting_indexer_prolog("LightningIndexerPrologSTest.b4_s1_4", configs)


# @pytest.mark.skip(reason="large test case")
def test_b8_s1_8():
    configs = IndexerPrologConfigs(
        q_linear=[16, 16, 128, 256, 256, 256],
        q_hd=[32, 32, 128, 128, 256, 256],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lighting_indexer_prolog("LightningIndexerPrologSTest.b8_s1_8", configs)


# @pytest.mark.skip(reason="large test case")
def test_b2_s1_4k():
    configs = IndexerPrologConfigs(
        q_linear=[16, 16, 128, 256, 256, 256],
        q_hd=[32, 32, 128, 128, 256, 256],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lighting_indexer_prolog("LightningIndexerPrologSTest.b2_s1_4k", configs)

    
# @pytest.mark.skip(reason="large test case")
def test_b1_s1_8193():
    configs = IndexerPrologConfigs(
        q_linear=[16, 16, 128, 256, 256, 256],
        q_hd=[32, 32, 128, 128, 256, 256],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        cube_l1_reuse_setting={1: 4},
        mg_copyin_upper_bound=2 * 1024 * 1024,
        pg_upper_bound=8192,
        block_size=128,
        t_sub_tile=1,
        chunk_size=2,
        vec_nbuffer_mode=0,
    )
    do_test_lighting_indexer_prolog("LightningIndexerPrologSTest.b1_s1_8193", configs)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_b1_s1_1()