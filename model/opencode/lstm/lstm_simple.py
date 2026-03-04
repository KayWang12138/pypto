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
"""
Simplified LSTM Operator Implementation for PyPTO

This is a simplified LSTM cell implementation for demonstration purposes.
"""

import os
import logging
import torch
from numpy.testing import assert_allclose
import pypto

BATCH_SIZE = 4
HIDDEN_SIZE = 128
INPUT_SIZE = 64


def lstm_golden(x, h_prev, c_prev, w_xh, w_hh, b):
    """
    Simplified LSTM golden reference.
    """
    # Linear transformations
    gates_x = torch.matmul(x, w_xh) + b
    gates_h = torch.matmul(h_prev, w_hh)
    gates = gates_x + gates_h
    
    # Split into 4 gates
    chunk_size = gates.shape[-1] // 4
    i_t, f_t, g_t, o_t = torch.split(gates, chunk_size, dim=-1)
    
    # Apply activations
    i_t = torch.sigmoid(i_t)
    f_t = torch.sigmoid(f_t)
    g_t = torch.tanh(g_t)
    o_t = torch.sigmoid(o_t)
    
    # Cell state update
    c_t = f_t * c_prev + i_t * g_t
    
    # Hidden state update
    h_t = o_t * torch.tanh(c_t)
    
    return h_t, c_t


def lstm_compute(x, h_prev, c_prev, w_xh, w_hh, b, h_out, c_out):
    """
    Simplified LSTM computation for PyPTO.
    """
    # Set tile shapes
    pypto.set_cube_tile_shapes([BATCH_SIZE, BATCH_SIZE], [HIDDEN_SIZE, HIDDEN_SIZE], [HIDDEN_SIZE, HIDDEN_SIZE])
    
    # Linear transformations
    gates_x = pypto.matmul(x, w_xh, pypto.DT_FP32)
    gates_h = pypto.matmul(h_prev, w_hh, pypto.DT_FP32)
    gates = pypto.add(gates_x, pypto.add(gates_h, b))
    
    # Split into 4 gates using view
    chunk_size = HIDDEN_SIZE
    i_t = pypto.view(gates, [BATCH_SIZE, chunk_size], [0, 0])
    f_t = pypto.view(gates, [BATCH_SIZE, chunk_size], [0, chunk_size * 1])
    g_t = pypto.view(gates, [BATCH_SIZE, chunk_size], [0, chunk_size * 2])
    o_t = pypto.view(gates, [BATCH_SIZE, chunk_size], [0, chunk_size * 3])
    
    # Apply activations
    i_t = pypto.sigmoid(i_t)
    f_t = = pypto.sigmoid(f_t)
    # Use sigmoid(2*x)-1 to approximate tanh
    g_t_scaled = pypto.mul(g_t, 2.0)
    g_t = pypto.sub(pypto.sigmoid(g_t_scaled), 1.0)
    o_t = pypto.sigmoid(o_t)
    
    # Cell state update
    c_t = pypto.add(pypto.mul(f_t, c_prev), pypto.mul(i_t, g_t))
    
    # Hidden state update
    c_t_scaled = pypto.mul(c_t, 2.0)
    c_tanh_approx = pypto.sub(pypto.sigmoid(c_t_scaled), 1.0)
    h_t = pypto.mul(o_t, c_tanh_approx)
    
    # Write outputs
    h_out[:] = h_t
    c_out[:] = c_t


def lstm_op(x, h_prev, c_prev, w_xh, w_hh, b, run_mode: str = "npu"):
    x_shape, h_shape, c_shape = x.shape, h_prev.shape, c_prev.shape
    w_xh_shape, w_hh_shape = w_xh.shape, w_hh.shape
    b_shape = b.shape
    
    if run_mode == "npu":
        mode = pypto.RunMode.NPU
    elif run_mode == "sim":
        mode = pypto.RunMode.SIM
    else:
        raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'npu' or 'sim'")
    
    @pypto.frontend.jit(runtime_options={"run_mode": mode})
    def lstm_kernel(
        x: pypto.Tensor(x_shape, pypto.DT_FP32),
        h_prev: pypto.Tensor(h_shape, pypto.DT_FP32),
        c_prev: pypto.Tensor(c_shape, pypto.DT_FP32),
        w_xh: pypto.Tensor(w_xh_shape, pypto.DT_FP32),
        w_hh: pypto.Tensor(w_hh_shape, pypto.DT_FP32),
        b: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> tuple:
        h_out = pypto.zeros(h_shape, pypto.DT_FP32)
        c_out = pypto.zeros(c_shape, pypto.DT_FP32)
        lstm_compute(x, h_prev, c_prev, w_xh, w_hh, b, h_out, c_out)
        return h_out, c_out
    
    return lstm_kernel(x, h_prev, c_prev, w_xh, w_hh, b)


def prepare_test_data(device):
    """Prepare test data."""
    x = torch.randn(BATCH_SIZE, INPUT_SIZE, dtype=torch.float32, device=device)
    h_prev = torch.randn(BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32, device=device)
    c_prev = torch.randn(BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32, device=device)
    
    w_xh = torch.randn(INPUT_SIZE, HIDDEN_SIZE * 4, dtype=torch.float32, device=device)
    w_hh = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE * 4, dtype=torch.float32, device=device)
    b = torch.randn(HIDDEN_SIZE * 4, dtype=torch.float32, device=device)
    
    return x, h_prev, c_prev, w_xh, w_hh, b


def run_precision_test(device_id, run_mode):
    """Run correctness verification."""
    logging.info("\n" + "=" * 40)
    logging.info("Running [Precision Test]")
    logging.info("=" * 40)
    
    device = f'npu:{device_id}' if run_mode == "npu" else 'cpu'
    
    # Prepare data
    x, h_prev, c_prev, w_xh, w_hh, b = prepare_test_data(device)
    
    # Run NPU Kernel
    h_out, c_out = lstm_op(x, h_prev, c_prev, w_xh, w_hh, b, run_mode)
    
    # Run Golden
    golden_h, golden_c = lstm_golden(x, h_prev, c_prev, w_xh, w_hh, b)
    
    # Compare
    diff_h = (h_out - golden_h).abs().max().item()
    diff_c = (c_out - golden_c).abs().max().item()
    logging.info(f"Max Diff Hidden: {diff_h:.6f}")
    logging.info(f"Max Diff Cell:   {diff_c:.6f}")
    
    try:
        assert_allclose(h_out.cpu().numpy(), golden_h.cpu().numpy(), rtol=1e-2, atol=1e-2)
        assert_allclose(c_out.cpu().numpy(), golden_c.cpu().numpy(), rtol=1e-2, atol=1e-2)
        logging.info(">> Precision Test PASSED!")
    except AssertionError as e:
        logging.error(">> Precision Test FAILED!")
        raise e


def get_device_id():
    """Get and validate TILE_FWK_DEVICE_ID from environment variable."""
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set TILE_FWK_DEVICE_ID environment variable.")
        return None
    
    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        logging.error(f"ERROR: TILE_FWK_DEVICE_ID must be an integer")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LSTM PyPTO Example")
    parser.add_argument('--run_mode', type=str, default="npu", choices=["npu", "sim"])
    args = parser.parse_args()
    
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
    else:
        device_id = 0
    
    # Run Test
    run_precision_test(device_id, args.run_mode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
