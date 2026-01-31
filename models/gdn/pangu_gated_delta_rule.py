#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""
Pangu Gated Delta Rule STest Module

This module provides test cases and wrapper functions for the Chunk Gated Delta Rule
attention mechanism. It includes both PyPTO implementation calls and PyTorch reference
implementations for validation.

Main Functions:
    - gen_dims: Generate dimension parameters
    - gen_inputs: Generate input tensors
    - gen_data: Generate test data based on case name
    - do_test_chunk_gated_delta_rule: Execute test case
    - pypto_chunk_gated_delta_rule_dyn: Dynamic wrapper for PyPTO implementation

Example:
    python pangu_gated_delta_rule.py
"""

import os

import pytest
import torch
import torch.nn.functional as F
import torch_npu

import pypto
from gated_delta_rule_impl import chunk_gated_delta_rule


def gen_dims(params):
    """Generate dimension parameters from input params."""
    dims = {}
    dims["T"] = params["T"]
    dims["B"] = params["B"]
    dims["Nqk"] = params["Nqk"]
    dims["Nv"] = params["Nv"]
    dims["D"] = 128
    dims["L"] = 128
    return dims


def gen_inputs(dims, dtype=torch.float32):
    """Generate input tensors for testing."""
    T = dims["T"]
    B = dims["B"]
    Nqk = dims["Nqk"]
    Nv = dims["Nv"]
    D = dims["D"]
    L = dims["L"]

    # Generate input data
    query = torch.rand([T, Nqk, D], dtype=dtype) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    key = torch.rand([T, Nqk, D], dtype=dtype) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    value = torch.rand([T, Nv, D], dtype=dtype) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    beta = torch.rand([T, Nv], dtype=dtype) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    gate = torch.rand([T, Nv], dtype=dtype) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    states = torch.zeros([B, Nv, D, D], dtype=dtype)

    # Generate act_seq_len based on B and T
    seq_len_per_batch = T // B
    act_seq_len = [i * seq_len_per_batch for i in range(B + 1)]
    act_seq_len = torch.tensor(act_seq_len, dtype=torch.int32)

    # Helper tensors
    mask = torch.tril(-torch.ones([L, L], dtype=dtype), diagonal=-1)
    tril_mask = torch.ones([L, L], dtype=dtype).tril()
    eye = torch.eye(16, dtype=dtype).repeat(1, 8)

    return {
        "query": query,
        "key": key,
        "value": value,
        "beta": beta,
        "gate": gate,
        "states": states,
        "act_seq_len": act_seq_len,
        "mask": mask,
        "tril_mask": tril_mask,
        "eye": eye,
    }


def golden_chunk_gated_delta_rule(inputs: dict, dims: dict):
    """Calculate golden output using PyTorch reference implementation."""
    query = inputs["query"]
    key = inputs["key"]
    value = inputs["value"]
    beta = inputs["beta"]
    gate = inputs["gate"]
    states = inputs["states"]
    act_seq_len = inputs["act_seq_len"]

    core_attn_out, final_state = segs_chunk_gated_delta_rule(
        query.clone(),
        key.clone(),
        value.clone(),
        gate.clone(),
        beta.clone(),
        initial_state=states.clone(),
        act_seq_len=act_seq_len.clone(),
    )

    return {
        "core_attn_out": core_attn_out,
        "final_state": final_state,
    }


def gen_data(case_name):
    """Generate test data based on case name."""
    if case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s4k"):
        params = {
            "T": 1024 * 8,
            "B": 2,
            "Nqk": 2,
            "Nv": 4,
        }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s4k"):
        params = {
            "T": 1024 * 8,
            "B": 2,
            "Nqk": 4,
            "Nv": 8,
        }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s8k"):
        params = {
            "T": 1024 * 16,
            "B": 2,
            "Nqk": 2,
            "Nv": 4,
        }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s8k"):
        params = {
            "T": 1024 * 16,
            "B": 2,
            "Nqk": 4,
            "Nv": 8,
        }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk16_nv32_s32k"):
        params = {
            "T": 1024 * 32,
            "B": 1,
            "Nqk": 16,
            "Nv": 32,
        }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s256k"):
        params = {
            "T": 1024 * 256,
            "B": 1,
            "Nqk": 2,
            "Nv": 4,
        }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s512k"):
        params = {
            "T": 1024 * 512,
            "B": 1,
            "Nqk": 2,
            "Nv": 4,
        }
    elif case_name.startswith("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s1m"):
        params = {
            "T": 1024 * 1024,
            "B": 1,
            "Nqk": 2,
            "Nv": 4,
        }
    else:
        raise Exception(f"Can't get func to gen golden, Case({case_name})")

    seed = 0
    torch.manual_seed(seed)
    dims = gen_dims(params)
    inputs = gen_inputs(dims, torch.float32)
    outputs = golden_chunk_gated_delta_rule(inputs, dims)
    return dims, inputs, outputs


def gen_zero_tensor(t):
    """Generate zero tensor with same shape and dtype."""
    return torch.zeros_like(t).npu()


def pypto_chunk_gated_delta_rule_dyn(inputs: dict, outputs: dict):
    """Dynamic wrapper for PyPTO chunk gated delta rule."""
    input_tensors = {
        inputs["query"]: [0],
        inputs["key"]: [0],
        inputs["value"]: [0],
        inputs["beta"]: [0],
        inputs["gate"]: [0],
        inputs["states"]: [],
        inputs["mask"]: [],
        inputs["tril_mask"]: [],
        inputs["eye"]: [],
        inputs["act_seq_len"]: [0],
    }
    output_tensors = {
        outputs["core_attn_out"]: [0],
        outputs["final_state"]: [],
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]
    chunk_gated_delta_rule(*pto_inputs, *pto_outputs)
    torch_npu.npu.synchronize()


def do_test_chunk_gated_delta_rule(case_name):
    """Execute test case for chunk gated delta rule."""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    print(f"=== run test case: {case_name} ===")

    dims, inputs_data, golden_data = gen_data(case_name)

    T = dims["T"]
    B = dims["B"]
    Nqk = dims["Nqk"]
    Nv = dims["Nv"]
    D = dims["D"]

    # Move inputs to NPU
    inputs = {
        "query": inputs_data["query"].npu(),
        "key": inputs_data["key"].npu(),
        "value": inputs_data["value"].npu(),
        "beta": inputs_data["beta"].npu(),
        "gate": inputs_data["gate"].npu(),
        "states": inputs_data["states"].npu(),
        "act_seq_len": inputs_data["act_seq_len"].npu(),
        "mask": inputs_data["mask"].npu(),
        "tril_mask": inputs_data["tril_mask"].npu(),
        "eye": inputs_data["eye"].npu(),
    }

    # Golden outputs
    core_attn_out_golden = golden_data["core_attn_out"]
    final_state_golden = golden_data["final_state"]

    # Prepare output tensors
    outputs = {
        "core_attn_out": gen_zero_tensor(core_attn_out_golden),
        "final_state": gen_zero_tensor(final_state_golden),
    }

    # Run PyPTO implementation
    pypto_chunk_gated_delta_rule_dyn(inputs, outputs)

    # Compare results
    compare(outputs["core_attn_out"].cpu(), core_attn_out_golden, "core_attn_out", 1e-3, 0, 1e-3)
    compare(outputs["final_state"].cpu(), final_state_golden, "final_state", 1e-3, 0, 1e-3)

    print(f"=== {case_name}: PASS ===")


def compare(actual, expected, name, rtol, atol_abs, atol_rel):
    """Compare two tensors with tolerance."""
    diff = torch.abs(actual.float() - expected.float())
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    tolerance = atol_abs + atol_rel * torch.abs(expected.float())
    out_of_tolerance = (diff > tolerance).sum().item()
    total = actual.numel()

    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
          f"out_of_tolerance={out_of_tolerance}/{total}")

    if out_of_tolerance > 0:
        ratio = out_of_tolerance / total
        if ratio > 0.01:  # More than 1% out of tolerance
            raise AssertionError(f"{name} comparison failed: {out_of_tolerance}/{total} elements out of tolerance")


def segs_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    act_seq_len,
    chunk_size=64,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    """Segmented chunk gated delta rule for batch processing."""
    t, n1, d = query.shape
    t, n, d = value.shape
    batch = act_seq_len.shape[0] - 1

    query = query.repeat_interleave(n // n1, dim=1)
    key = key.repeat_interleave(n // n1, dim=1)

    final_state = torch.zeros([batch, n, d, d], dtype=torch.float32, device=query.device)

    query, key, value, beta, g = [
        x.transpose(0, 1).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    final_attn = torch.zeros([t, n, d], dtype=torch.float32, device=query.device)

    for b_idx in range(batch):
        s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
        b_ofs = act_seq_len[b_idx]
        seg_s = 128
        pad_size = (chunk_size - s % chunk_size) % chunk_size
        pad_seq_length = s + pad_size
        batch_query = F.pad(query[:, b_ofs : b_ofs + s], (0, 0, 0, pad_size))
        batch_key = F.pad(key[:, b_ofs : b_ofs + s], (0, 0, 0, pad_size))
        batch_value = F.pad(value[:, b_ofs : b_ofs + s], (0, 0, 0, pad_size))
        batch_beta = F.pad(beta[:, b_ofs : b_ofs + s], (0, pad_size))
        batch_g = F.pad(g[:, b_ofs : b_ofs + s], (0, pad_size))
        result_list = []
        recurrent_state = initial_state[b_idx : b_idx + 1, ...]
        for s_idx in range(0, pad_seq_length, seg_s):
            chunk_query = batch_query[:, s_idx : s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_key = batch_key[:, s_idx : s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_value = batch_value[:, s_idx : s_idx + seg_s, :].reshape(1, n, seg_s, d)
            chunk_gate = batch_g[:, s_idx : s_idx + seg_s].reshape(1, n, seg_s)
            chunk_beta = batch_beta[:, s_idx : s_idx + seg_s].reshape(1, n, seg_s)
            cur_attn, cur_state = torch_chunk_gated_delta_rule(
                chunk_query,
                chunk_key,
                chunk_value,
                chunk_gate,
                chunk_beta,
                chunk_size,
                recurrent_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
            )
            result_list.append(cur_attn.squeeze(0))
            recurrent_state = cur_state
        batch_attn = torch.cat(result_list, dim=0)[:s]
        final_attn[b_ofs : b_ofs + s] = batch_attn
        final_state[b_idx : b_idx + 1, ...] = recurrent_state
    return final_attn, final_state


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
):
    """PyTorch reference implementation of chunk gated delta rule."""
    b, n, s, d = value.shape

    initial_state = initial_state.transpose(3, 2)
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=query.device).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    core_attn_out = torch.zeros_like(value).to(query.device)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    last_recurrent_state = last_recurrent_state.transpose(3, 2)

    return core_attn_out, last_recurrent_state


# ==================== Test Cases ====================

def test_b2_nqk2_nv4_s4k():
    """Test case: B=2, Nqk=2, Nv=4, S=4K"""
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s4k")


@pytest.mark.skip(reason="large test case")
def test_b2_nqk4_nv8_s4k():
    """Test case: B=2, Nqk=4, Nv=8, S=4K"""
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s4k")


@pytest.mark.skip(reason="large test case")
def test_b2_nqk2_nv4_s8k():
    """Test case: B=2, Nqk=2, Nv=4, S=8K"""
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b2_nqk2_nv4_s8k")


@pytest.mark.skip(reason="large test case")
def test_b2_nqk4_nv8_s8k():
    """Test case: B=2, Nqk=4, Nv=8, S=8K"""
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b2_nqk4_nv8_s8k")


@pytest.mark.skip(reason="large test case")
def test_b1_nqk16_nv32_s32k():
    """Test case: B=1, Nqk=16, Nv=32, S=32K"""
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk16_nv32_s32k")


@pytest.mark.skip(reason="large test case")
def test_b1_nqk2_nv4_s256k():
    """Test case: B=1, Nqk=2, Nv=4, S=256K"""
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s256k")


@pytest.mark.skip(reason="large test case")
def test_b1_nqk2_nv4_s512k():
    """Test case: B=1, Nqk=2, Nv=4, S=512K"""
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s512k")


@pytest.mark.skip(reason="large test case")
def test_b1_nqk2_nv4_s1m():
    """Test case: B=1, Nqk=2, Nv=4, S=1M"""
    do_test_chunk_gated_delta_rule("ChunkGatedDeltaRuleSTest.b1_nqk2_nv4_s1m")


if __name__ == "__main__":
    import datetime
    start = datetime.datetime.now()
    print("start:", start)
    test_b2_nqk2_nv4_s4k()
    end = datetime.datetime.now()
    duration = end - start
    print("end:", end)
    print("duration:", duration)
