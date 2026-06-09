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
import os
import json
import pypto
import numpy as np
import torch
from pypto.cost_model import _cost_model_run_subgraph_line


def safe_json_load(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data, None
    except FileNotFoundError:
        return None, "File not found"
    except json.JSONDecodeError as e:
        return None, f"Invalid json format: {e}"
    except PermissionError:
        return None, "Permission Error"
    except Exception as e:
        return None, f"Load json fail, unknown error: {e}"


def softmax_core(input_tensor: pypto.tensor) -> pypto.tensor:
    row_max = pypto.amax(input_tensor, dim=-1, keepdim=True)
    sub = pypto.sub(input_tensor, row_max)
    exp = pypto.exp(sub)
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    return pypto.div(exp, esum)


@pypto.frontend.jit(
    runtime_options={"run_mode": 1}
)
def softmax(input_tensor: pypto.Tensor(), output_tensor: pypto.Tensor()):
    tensor_shape = input_tensor.shape
    b = tensor_shape[0]
    n1, n2, dim = tensor_shape[1:]
    tile_b = 1
    b_loop = b // tile_b

    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in range(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        input_view = input_tensor[b_offset:b_offset_end, :n1, :n2, :dim]
        softmax_out = softmax_core(input_view)
        pypto.assemble(softmax_out, [b_offset, 0, 0, 0], output_tensor)


def test_costmodel_subgraph_line_basic():
    shape = (32, 32, 1, 256)
    input_data = torch.rand(shape, dtype=torch.float32)
    output_data = torch.zeros(shape, dtype=torch.float32)

    softmax(input_data, output_data)

    result = _cost_model_run_subgraph_line(
        inputs=[input_data],
        outputs=[output_data],
        p_sg_id=0,
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("status") == "success", f"status is {result.get('status')}, error: {result.get('error_msg')}"
    assert "p_sg_id" in result
    assert result["p_sg_id"] == 0
    assert "subgraph_total_cycles" in result
    assert isinstance(result["subgraph_total_cycles"], int)
    assert "functions" in result
    assert isinstance(result["functions"], list)
    assert "output_dir" in result


def test_costmodel_subgraph_line_multi_subgraphs():
    shape = (32, 32, 1, 256)
    input_data = torch.rand(shape, dtype=torch.float32)
    output_data = torch.zeros(shape, dtype=torch.float32)

    softmax(input_data, output_data)

    tested_sg_count = 0
    for p_sg_id in range(10):
        result = _cost_model_run_subgraph_line(
            inputs=[input_data],
            outputs=[output_data],
            p_sg_id=p_sg_id,
        )
        assert isinstance(result, dict)

        if result.get("status") != "success":
            continue

        tested_sg_count += 1
        assert result["p_sg_id"] == p_sg_id
        assert isinstance(result["subgraph_total_cycles"], int)
        assert isinstance(result["functions"], list)

        for func in result["functions"]:
            assert "hash" in func
            assert "cycles" in func
            assert isinstance(func["cycles"], int)

    assert tested_sg_count > 0, "Expected at least one subgraph to be available"


def test_costmodel_subgraph_line_swimlane():
    shape = (32, 32, 1, 256)
    input_data = torch.rand(shape, dtype=torch.float32)
    output_data = torch.zeros(shape, dtype=torch.float32)

    softmax(input_data, output_data)

    result = _cost_model_run_subgraph_line(
        inputs=[input_data],
        outputs=[output_data],
        p_sg_id=0,
    )

    assert result.get("status") == "success", f"status is {result.get('status')}, error: {result.get('error_msg')}"

    output_dir = result.get("output_dir", "")
    assert output_dir, "output_dir should not be empty"

    merged_swimlane_path = os.path.join(output_dir, "merged_swimlane.json")
    merged_swimlane, error = safe_json_load(merged_swimlane_path)
    assert not error, f"safe_json_load({merged_swimlane_path}): {error}"
    assert merged_swimlane is not None


def test_costmodel_subgraph_line_invalid_sg_id():
    shape = (32, 32, 1, 256)
    input_data = torch.rand(shape, dtype=torch.float32)
    output_data = torch.zeros(shape, dtype=torch.float32)

    softmax(input_data, output_data)

    result = _cost_model_run_subgraph_line(
        inputs=[input_data],
        outputs=[output_data],
        p_sg_id=999,
    )

    assert result["p_sg_id"] == 999
    assert result["subgraph_total_cycles"] == 18446744073709551615
