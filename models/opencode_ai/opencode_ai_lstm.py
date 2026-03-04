#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
OpenCode AI LSTM Module

This module implements LSTM (Long Short-Term Memory) for OpenCode AI.
It is a high-performance tensor computation function based on the PyPTO framework,
primarily used for sequence modeling and natural language processing tasks.

Main Functions:
    - opencode_ai_lstm: Main LSTM kernel function
    - opencode_ai_lstm_main: JIT compiled kernel for LSTM computation
    - opencode_ai_lstm_compute: Base inference function for LSTM computation
"""
import os
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from opencode_ai_lstm_impl import LstmConfig, LstmTileConfig, opencode_ai_lstm_compute
from opencode_ai_lstm_golden import opencode_ai_lstm_compute as golden_compute


def check_args(
    states_4d,
    z4_4d,
    prev_cell,
    w_cell,
    b_cell,
    w_state,
    b_state,
    h_out,
    c_out
):
    """Check input arguments for OpenCode AI LSTM."""
    assert states_4d.dim() == 2
    assert states_4d.shape[1] == 16384
    assert states_4d.dtype == torch.float16

    assert z4_4d.dim() == 2
    assert z4_4d.shape[1] == 16384
    assert z4_4d.dtype == torch.float16

    assert prev_cell.dim() == 2
    assert prev_cell.shape[1] == 4096
    assert prev_cell.dtype == torch.float16

    assert w_cell.dim() == 1
    assert w_cell.shape[0] == 4096
    assert w_cell.dtype == torch.float16

    assert b_cell.dim() == 1
    assert b_cell.shape[0] == 4096
    assert b_cell.dtype == torch.float16

    assert w_state.dim() == 1
    assert w_state.shape[0] == 4096
    assert w_state.dtype == torch.float16

    assert b_state.dim() == 1
    assert b_state.shape[0] == 4096
    assert b_state.dtype == torch.float16

    assert h_out.dim() == 2
    assert h_out.shape[1] == 4096
    assert h_out.dtype == torch.float16

    assert c_out.dim() == 2
    assert c_out.shape[1] == 4096
    assert c_out.dtype == torch.float16


def main():
    """Main entry point for OpenCode AI LSTM."""
    test_opencode_ai_lstm()


def gen_input(
    batch_size: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device_id: int
) -> tuple:
    """Generate random input data for LSTM."""
    torch.manual_seed(42)
    hidden_dim_4 = hidden_dim * 4

    states_4d = torch.randn([batch_size, hidden_dim_4], dtype=dtype, device=f'npu:{device_id}')
    z4_4d = torch.randn([batch_size, hidden_dim_4], dtype=dtype, device=f'npu:{device_id}')
    prev_cell = torch.randn([batch_size, hidden_dim], dtype=dtype, device=f'npu:{device_id}')

    w_cell = torch.randn([hidden_dim], dtype=dtype, device=f'npu:{device_id}')
    b_cell = torch.randn([hidden_dim], dtype=dtype, device=f'npu:{device_id}')
    w_state = torch.randn([hidden_dim], dtype=dtype, device=f'npu:{device_id}')
    b_state = torch.randn([hidden_dim], dtype=dtype, device=f'npu:{device_id}')

    h_out = torch.zeros([batch_size, hidden_dim], dtype=dtype, device=f'npu:{device_id}')
    c_out = torch.zeros([batch_size, hidden_dim], dtype=dtype, device=f'npu:{device_id}')

    return states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out


def opencode_ai_lstm_main(states_4d_shape, z4_4d_shape, prev_cell_shape, w_cell_shape, b_cell_shape, w_state_shape, b_state_shape, h_out_shape, c_out_shape):
    """Main function for OpenCode AI LSTM JIT compilation."""
    states_4d_shape = (pypto.frontend.dynamic("states_4d_shape"), states_4d_shape[1])
    z4_4d_shape = (pypto.frontend.dynamic("states_4d_shape"), z4_4d_shape[1])
    prev_cell_shape = (pypto.frontend.dynamic("states_4d_shape"), prev_cell_shape[1])
    h_out_shape = (pypto.frontend.dynamic("states_4d_shape"), h_out_shape[1])
    c_out_shape = (pypto.frontend.dynamic("states_4d_shape"), c_out_shape[1])

    config = LstmConfig(alpha=0.1, eps_cell=1e-6, eps_state=1e-6)
    tile_config = LstmTileConfig()

    @pypto.frontend.jit(
        runtime_options={"device_sched_mode": 1,
                         "stitch_cfgcache_size": 2700000},
    )
    def kernel(
        states_4d: pypto.tensor(states_4d_shape, pypto.DT_FP16),
        z4_4d: pypto.tensor(z4_4d_shape, pypto.DT_FP16),
        prev_cell: pypto.tensor(prev_cell_shape, pypto.DT_FP16),
        w_cell: pypto.tensor(w_cell_shape, pypto.DT_FP16),
        b_cell: pypto.tensor(b_cell_shape, pypto.DT_FP16),
        w_state: pypto.tensor(w_state_shape, pypto.DT_FP16),
        b_state: pypto.tensor(b_state_shape, pypto.DT_FP16),
        h_out: pypto.tensor(h_out_shape, pypto.DT_FP16),
        c_out: pypto.tensor(c_out_shape, pypto.DT_FP16)
    ):
        opencode_ai_lstm_compute(
            states_4d=states_4d,
            z4_4d=z4_4d,
            prev_cell=prev_cell,
            w_cell=w_cell,
            b_cell=b_cell,
            w_state=w_state,
            b_state=b_state,
            config=config,
            tile_config=tile_config,
            h_out=h_out,
            c_out=c_out
        )

    return kernel


@allow_in_graph
def opencode_ai_lstm(
    states_4d: torch.Tensor,
    z4_4d: torch.Tensor,
    prev_cell: torch.Tensor,
    w_cell: torch.Tensor,
    b_cell: torch.Tensor,
    w_state: torch.Tensor,
    b_state: torch.Tensor,
    h_out: torch.Tensor,
    c_out: torch.Tensor
) -> None:
    """
    OpenCode AI LSTM computation function.

    This function computes LSTM output using high-performance PyPTO operations.
    It implements the standard LSTM architecture with RMSNorm and GELU activations.

    Args:
        states_4d: Input states [batch_size, hidden_dim * 4]
        z4_4d: Input z4 [batch_size, hidden_dim * 4]
        prev_cell: Previous cell state [batch_size, hidden_dim]
        w_cell: Cell weight [hidden_dim]
        b_cell: Cell bias [hidden_dim]
        w_state: State weight [hidden_dim]
        b_state: State bias [hidden_dim]
        h_out: Output hidden state [batch_size, hidden_dim]
        c_out: Output cell state [batch_size, hidden_dim]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if not isinstance(states_4d, FakeTensor):
        check_args(states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out)

    shapes = [states_4d.shape, z4_4d.shape, prev_cell.shape, w_cell.shape, b_cell.shape, w_state.shape, b_state.shape, h_out.shape, c_out.shape]
    inputs = [states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out]
    opencode_ai_lstm_main(*shapes)(*inputs)


def test_opencode_ai_lstm() -> None:
    """Test function for OpenCode AI LSTM."""
    x_dtype = torch.float16
    batch_size = 32
    hidden_dim = 4096
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # Generate inputs
    states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out = \
        gen_input(batch_size, hidden_dim, x_dtype, device_id)

    # Run kernel
    opencode_ai_lstm(states_4d, z4_4d, prev_cell, w_cell, b_cell, w_state, b_state, h_out, c_out)

    # Run golden
    golden_h, golden_c = golden_compute(
        states_4d.cpu(), z4_4d.cpu(), prev_cell.cpu(),
        alpha=0.1, eps_cell=1e-6, eps_state=1e-6,
        w_cell=w_cell.cpu(), b_cell=b_cell.cpu(), w_state=w_state.cpu(), b_state=b_state.cpu()
    )

    # Compare results
    assert_allclose(np.array(h_out.cpu().flatten().tolist()), np.array(golden_h.cpu().flatten().tolist()),
                    rtol=0.001, atol=5e-3)
    assert_allclose(np.array(c_out.cpu().flatten().tolist()), np.array(golden_c.cpu().flatten().tolist()),
                    rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    main()
