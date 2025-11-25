# -----------------------------------------------------------------------------------------------------------
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
import pypto
import torch
import torch_npu
import os
from pathlib import Path
from examples.deepseek_v32_exp.lightning_indexer_prolog_quant import (
    IndexerPrologQuantInput, IndexerPrologQuantOutput, IndexerPrologQuantAttr, IndexerPrologQuantConfigs,
    lightning_indexer_prolog_quant)
import pytest


def lighting_indexer_prolog_quant_dyn(inputs: IndexerPrologQuantInput, outputs: IndexerPrologQuantOutput,
                                      attrs: IndexerPrologQuantAttr, configs: IndexerPrologQuantConfigs):
    input_tensors = [
        inputs.x,
        inputs.q_norm,
        inputs.q_norm_scale,
        inputs.w_qb,
        inputs.w_qb_scale,
        inputs.wk,
        inputs.w_proj,
        inputs.ln_gamma_k,
        inputs.ln_beta_k,
        inputs.cos_idx_rope,
        inputs.sin_idx_rope,
        inputs.hadamard_q,
        inputs.hadamard_k,
        inputs.k_cache,
        inputs.k_cache_scale,
        inputs.k_cache_index,
    ]
    output_tensors = [outputs.q_int8, outputs.q_scale, outputs.k_int8, outputs.k_scale, outputs.weights]
    lightning_indexer_prolog_quant(input_tensors, output_tensors, attrs, configs)
    pypto.runtime._device_synchronize()


def tensor_from_bin(data_path, file_name, shape, dtype):
    num_elem = 1
    for s in shape:
        num_elem *= s
    to_load_size = num_elem * dtype.itemsize
    file_size = os.path.getsize(os.path.join(data_path, file_name))
    assert file_size == to_load_size, f"file size {file_size} != to load size {to_load_size}, {file_name}"
    return torch.from_file(os.path.join(data_path, file_name), size=num_elem, dtype=dtype).reshape(shape)


def gen_zero_tensor(t):
    return torch.zeros_like(t).npu()


def gen_data(case_name):
    import sys
    golden_dir_path = str(
        Path(Path(__file__), "../../../../../", "framework/tests/st/operator/src/test_deepseek_v3.2_exp").resolve())
    sys.path.append(golden_dir_path)
    golden_dir_path = str(
        Path(Path(__file__), "../../../../../", "framework/tests/cmake/scripts").resolve())
    sys.path.append(golden_dir_path)

    from gen_quant_lightning_indexer_prolog import gen_dims, gen_indexer_prolog_inputs, indexer_prolog
    if case_name.startswith("QuantLightningIndexerPrologSTest.b4_s1_2_s2_64k"):
        params = {
            "b": 4,
            "s1": 2,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b8_s1_2_s2_64k"):
        params = {
            "b": 8,
            "s1": 2,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b1_s1_4k_s2_64k"):
        params = {
            "b": 1,
            "s1": 1024 * 4,
            "s2": 1024 * 64
        }
    elif case_name.startswith("QuantLightningIndexerPrologSTest.b2_s1_4k_s2_64k"):
        params = {
            "b": 2,
            "s1": 1024 * 4,
            "s2": 1024 * 64
        }
    else:
        raise Exception(f"Can't get func to gen golden, Case({case_name})")

    seed = 0
    # PyTorch 随机数生成器
    torch.manual_seed(seed)
    dims = gen_dims(params)
    inputs = gen_indexer_prolog_inputs(dims, torch.bfloat16)
    outputs = indexer_prolog(inputs, dims)
    return dims, inputs, outputs


def do_test_lighting_indexer_prolog_quant(case_name):
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    print(f"=== run test case: {case_name} ===")

    dims, inputs_data, golden_data = gen_data(case_name)

    t = dims["t"]
    h = dims["h"]
    q_lora_rank = dims["q_lora_rank"]
    idx_head_dim = dims["idx_head_dim"]
    head_num = dims["idx_n_heads"]
    rope_head_dim = dims["rope_head_dim"]

    torch_npu.npu.config.allow_internal_format = True

    inputs = IndexerPrologQuantInput(
        x=inputs_data["token_x"].npu().reshape(t, h),
        q_norm=inputs_data["q_norm"].npu().reshape(t, q_lora_rank),
        q_norm_scale=inputs_data["q_norm_scale"].npu().reshape(t, 1),
        w_qb=torch_npu.npu_format_cast(inputs_data["w_idx_qb_nz"].npu().contiguous(), torch_npu.Format.FRACTAL_NZ),
        w_qb_scale=inputs_data["w_idx_qb_scale"].npu(),
        wk=torch_npu.npu_format_cast(inputs_data["w_idx_k_nz"].npu().contiguous(), torch_npu.Format.FRACTAL_NZ),
        w_proj=torch_npu.npu_format_cast(
            inputs_data["weights_proj_nz"].npu().contiguous(), torch_npu.Format.FRACTAL_NZ),
        ln_gamma_k=inputs_data["layer_norm_gamma"].npu(),
        ln_beta_k=inputs_data["layer_norm_beta"].npu(),
        cos_idx_rope=inputs_data["cos_idx_rope"].npu().reshape(t, rope_head_dim),
        sin_idx_rope=inputs_data["sin_idx_rope"].npu().reshape(t, rope_head_dim),
        hadamard_q=inputs_data["hadamard_q"].npu(),
        hadamard_k=inputs_data["hadamard_k"].npu(),
        k_cache=inputs_data["idx_k_cache"].npu(),
        k_cache_scale=inputs_data["idx_k_scale_cache"].npu(),
        k_cache_index=inputs_data["idx_k_cache_index"].npu().reshape(t)
    )

    q_int8_golden = golden_data["query"].reshape(t, head_num, idx_head_dim)
    q_scale_golden = golden_data["query_scale"].reshape(t, head_num, 1)
    k_cache_golden = golden_data["idx_k_cache_out"]
    k_cache_scale_golden = golden_data["idx_k_scale_cache_out"]
    weights_golden = golden_data["weights"].reshape(t, head_num)

    outputs = IndexerPrologQuantOutput(
        q_int8=gen_zero_tensor(q_int8_golden),
        q_scale=gen_zero_tensor(q_scale_golden),
        k_int8=inputs.k_cache,
        k_scale=inputs.k_cache_scale,
        weights=gen_zero_tensor(weights_golden)
    )

    # ---- Attrs ----
    attrs = IndexerPrologQuantAttr(
        eps=1e-6,
        layerout_query="TND",
        layerout_key="PA_BSND",
    )

    configs = IndexerPrologQuantConfigs(
        q_linear=[16, 16, 512, 512, 128, 128],
        q_hd=[32, 32, 128, 128, 128, 128],
        k_linear=[16, 16, 512, 512, 64, 64],
        w_linear=[16, 16, 1024, 1024, 32, 32],
        unroll_list=[32, 16, 8, 4, 2, 1],
        l1_reuse_param={1: 4},
        copy_in_threshold=2 * 1024 * 1024,
        cycle_upper_bound=8192,
        block_size=128
    )

    lighting_indexer_prolog_quant_dyn(inputs, outputs, attrs, configs)

    if "b2_s1_4k_s2_64k" in case_name:
        error_count_threshold = 8
    else:
        error_count_threshold = 0
    compare(outputs.q_int8.cpu(), q_int8_golden, "q_int8", 1, 0, error_count_threshold)
    compare(outputs.q_scale.cpu(), q_scale_golden, "q_scale", 0.0001, 0, error_count_threshold)
    compare(outputs.k_int8.cpu(), k_cache_golden, "k_int8", 1, 0, 0)
    compare(outputs.k_scale.cpu(), k_cache_scale_golden, "k_scale", 0.0001, 0, 0)
    compare(outputs.weights.cpu(), weights_golden, "weights", 0.0001, 0., 0)

    print(f"=== {case_name}: PASS ===")

    pypto.runtime._device_fini()


def compare(t: torch.Tensor, t_ref: torch.Tensor, name, atol, rtol, error_count_threshold=0):
    assert t.shape == t_ref.shape
    assert t.dtype == t_ref.dtype
    assert t.device == t_ref.device
    diff_mask1 = (t - t_ref).abs() > atol
    diff_mask2 = (t - t_ref).abs() > rtol * t_ref.abs()
    diff_mask = diff_mask1 & diff_mask2
    error_count = diff_mask.sum().item()
    max_diff, max_pos = torch.max((t - t_ref).abs().flatten(), dim=0)
    max_pos = torch.unravel_index(max_pos, t.shape)
    max_pos = tuple(idx.item() for idx in max_pos)

    assert error_count <= error_count_threshold, \
        (f"compare fail: {name}, max diff: {max_diff} at {max_pos}, "
         f"error_count: {error_count}, error_count_threshold: {error_count_threshold}")


@pytest.mark.skip(reason="similar to test_b8_s1_2_s2_64k")
def test_b4_s1_2_s2_64k():
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b4_s1_2_s2_64k")


def test_b8_s1_2_s2_64k():
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b8_s1_2_s2_64k")


@pytest.mark.skip(reason="large test case")
def test_b1_s1_4k_s2_64k():
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b1_s1_4k_s2_64k")


@pytest.mark.skip(reason="large test case")
def test_b2_s1_4k_s2_64k():
    do_test_lighting_indexer_prolog_quant("QuantLightningIndexerPrologSTest.b2_s1_4k_s2_64k")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
        level=logging.INFO
    )
    test_b8_s1_2_s2_64k()
    test_b2_s1_4k_s2_64k()
