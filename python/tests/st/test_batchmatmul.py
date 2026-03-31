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
"""
from dataclasses import dataclass, field
from typing import Optional
import os
import numpy as np
import torch
import torch_npu
import pypto
import pytest
from numpy.testing import assert_allclose
import torch.nn.functional as F
from typing import Optional, List, Tuple


FP32 = pypto.DT_FP32
FP16 = pypto.DT_FP16
INT32 = pypto.DT_INT32
INT8 = pypto.DT_INT8
UINT64 = pypto.DT_UINT64
UINT32 = pypto.DT_UINT32


@dataclass
class ShapeConfig:
    ori_shape: list
    m_tile_shape: list
    k_tile_shape: list
    n_tile_shape: list
    view_shape: list
    in_dtype: pypto.DataType
    out_dtype: pypto.DataType
    a_trans: bool = False
    b_trans: bool = False
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    gm_acc: bool = False


@dataclass
class ExtendParams:
    bias_shape: list = field(default_factory=list)
    bias_dtype: np.dtype = None
    scale_shape: list = field(default_factory=list)
    scale_dtype: np.dtype = None
    scale: int = None
    relu_type: int = None


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
)
def batchmatmul_3d_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    shape_info: ShapeConfig,
):
    m = shape_info.ori_shape[1]
    n = shape_info.ori_shape[3]
    m_view = shape_info.view_shape[0]
    n_view = shape_info.view_shape[1]
    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    pypto.set_cube_tile_shapes(shape_info.m_tile_shape, shape_info.k_tile_shape, shape_info.n_tile_shape,
                                enable_split_k=shape_info.gm_acc)
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            if shape_info.a_trans:
                a_view = a_tensor[:, :, m_idx * m_view: m_idx * m_view + m_view]
            else:
                a_view = a_tensor[:, m_idx * m_view: m_idx * m_view + m_view, :]
            if shape_info.b_trans:
                b_view = b_tensor[:, n_idx * n_view: n_idx * n_view + n_view, :]
            else:
                b_view = b_tensor[:, :, n_idx * n_view: n_idx * n_view + n_view]
            out_view = pypto.matmul(a_view, b_view, a_trans=shape_info.a_trans, b_trans=shape_info.b_trans,
                                    out_dtype=shape_info.out_dtype)
            out_tensor[:, 
                        m_idx * m_view: m_idx * m_view + m_view, 
                        n_idx * n_view: n_idx * n_view + n_view] = out_view


@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 0, "compile_debug_mode": 0}
)
def batchmatmul_4d_kernel(
    a_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    b_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    out_tensor: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    shape_info: ShapeConfig,
):
    m = shape_info.ori_shape[2]
    n = shape_info.ori_shape[4]
    m_view = shape_info.view_shape[0]
    n_view = shape_info.view_shape[1]
    m_loop = (m + m_view - 1) // m_view
    n_loop = (n + n_view - 1) // n_view
    pypto.set_cube_tile_shapes(shape_info.m_tile_shape, shape_info.k_tile_shape, shape_info.n_tile_shape, 
                                enable_split_k=shape_info.gm_acc)
    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_L0_mIdx", idx_name="m_idx"):
        for n_idx in pypto.loop(0, n_loop, 1, name="LOOP_L0_nIdx", idx_name="n_idx"):
            if shape_info.a_trans:
                a_view = a_tensor[:, :, :, m_idx * m_view: m_idx * m_view + m_view]
            else:
                a_view = a_tensor[:, :, m_idx * m_view: m_idx * m_view + m_view, :]
            if shape_info.b_trans:
                b_view = b_tensor[:, :, n_idx * n_view: n_idx * n_view + n_view, :]
            else:
                b_view = b_tensor[:, :, :, n_idx * n_view: n_idx * n_view + n_view]
            out_view = pypto.matmul(a_view, b_view, a_trans=shape_info.a_trans, b_trans=shape_info.b_trans,
                                    out_dtype=shape_info.out_dtype)
            out_tensor[:, :, 
                        m_idx * m_view: m_idx * m_view + m_view, 
                        n_idx * n_view: n_idx * n_view + n_view] = out_view


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="unused test case")
def test_bmm3d_with_mn_split():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_npu.npu.config.allow_internal_format = True
    b = 3
    m = 128
    k = 256
    n = 512
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 64
    n_view = 128
    shape_info = ShapeConfig([b, m, k, n], [tile_m, tile_m], [tile_k, 2 * tile_k], [tile_n, tile_n], [m_view, n_view],
                                FP16, FP32, False, False, False, False, False, False)
    a_tensor = torch.rand([b, m, k], dtype=torch.float16, device=f'npu:{device_id}')
    b_tensor = torch.rand([b, k, n], dtype=torch.float16, device=f'npu:{device_id}')
    out_tensor = torch.zeros([b, m, n], dtype=torch.float32, device=f'npu:{device_id}')
    golden = torch.matmul(a_tensor.to(torch.float32), b_tensor.to(torch.float32))
    batchmatmul_3d_kernel(a_tensor, b_tensor, out_tensor, shape_info)
    assert torch.allclose(out_tensor.cpu().to(torch.float32), golden.cpu().to(torch.float32), atol=1e-3, rtol=1e-3)


@pytest.mark.soc("950", "910")
@pytest.mark.skip(reason="unused test case")
def test_bmm4d_with_mn_split():
    device_id = os.environ.get('TILE_FWK_DEVICE_ID', 0)
    torch_npu.npu.config.allow_internal_format = True
    b1 = 3
    b2 = 2
    m = 128
    k = 256
    n = 512
    tile_m = 64
    tile_k = 64
    tile_n = 64
    m_view = 64
    n_view = 128
    shape_info = ShapeConfig([b1, b2, m, k, n], [tile_m, tile_m], [tile_k, 2 * tile_k], [tile_n, tile_n], 
                            [m_view, n_view], FP16, FP32, False, False, False, False, False, False)
    a_tensor = torch.rand([b1, b2, m, k], dtype=torch.float16, device=f'npu:{device_id}')
    b_tensor = torch.rand([b1, b2, k, n], dtype=torch.float16, device=f'npu:{device_id}')
    out_tensor = torch.zeros([b1, b2, m, n], dtype=torch.float32, device=f'npu:{device_id}')
    golden = torch.matmul(a_tensor.to(torch.float32), b_tensor.to(torch.float32))
    batchmatmul_4d_kernel(a_tensor, b_tensor, out_tensor, shape_info)
    assert torch.allclose(out_tensor.cpu().to(torch.float32), golden.cpu().to(torch.float32), atol=1e-3, rtol=1e-3)


import time
from typing import Dict, Any, List


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _build_shape_info(case: Dict[str, Any]) -> ShapeConfig:
    """
    按官方 test_batchmatmul.py 的 ShapeConfig 方式组装参数。
    """
    ndim = case["ndim"]
    if ndim == 3:
        ori_shape = [case["b"], case["m"], case["k"], case["n"]]
    elif ndim == 4:
        ori_shape = [case["b1"], case["b2"], case["m"], case["k"], case["n"]]
    else:
        raise ValueError(f"unsupported ndim={ndim}, only 3/4 are supported")

    return ShapeConfig(
        ori_shape=ori_shape,
        m_tile_shape=case.get("m_tile_shape", [64, 64]),
        k_tile_shape=case.get("k_tile_shape", [64, 128]),
        n_tile_shape=case.get("n_tile_shape", [64, 64]),
        view_shape=case.get("view_shape", [64, 128]),
        in_dtype=FP16,
        out_dtype=FP32,
        a_trans=case.get("a_trans", False),
        b_trans=case.get("b_trans", False),
        a_format_nz=False,
        b_format_nz=False,
        c_format_nz=False,
        gm_acc=case.get("gm_acc", False),
    )


def _gen_inputs_for_case(case: Dict[str, Any], device: str):
    """
    生成每轮输入，保证 PyPTO 和 Torch 走同一份数据。
    """
    if case["ndim"] == 3:
        a = torch.rand([case["b"], case["m"], case["k"]], dtype=torch.float16, device=device)
        b = torch.rand([case["b"], case["k"], case["n"]], dtype=torch.float16, device=device)
        out = torch.zeros([case["b"], case["m"], case["n"]], dtype=torch.float32, device=device)
    else:
        a = torch.rand([case["b1"], case["b2"], case["m"], case["k"]], dtype=torch.float16, device=device)
        b = torch.rand([case["b1"], case["b2"], case["k"], case["n"]], dtype=torch.float16, device=device)
        out = torch.zeros([case["b1"], case["b2"], case["m"], case["n"]], dtype=torch.float32, device=device)
    return a, b, out


def _run_one_case_multi_round(
    case: Dict[str, Any],
    device: str,
    num_rounds: int,
    warmup_rounds: int,
    rtol: float,
    atol: float,
):
    """
    单个 shape case 的多轮测试（warmup + 正式轮次），比较 PyPTO 与 Torch。
    """
    shape_info = _build_shape_info(case)
    total_rounds = warmup_rounds + num_rounds

    pypto_times, pypto_mems = [], []
    torch_times, torch_mems = [], []

    case_name = case["name"]
    print("\n" + "-" * 90)
    print(f"[CASE] {case_name} | warmup={warmup_rounds}, rounds={num_rounds}, device={device}")
    print(f"       ndim={case['ndim']}, m={case['m']}, k={case['k']}, n={case['n']}")
    print("-" * 90)

    for r in range(total_rounds):
        torch.manual_seed(2026 + r)
        a_tensor, b_tensor, out_tensor = _gen_inputs_for_case(case, device)

        # PyPTO
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        if case["ndim"] == 3:
            batchmatmul_3d_kernel(a_tensor, b_tensor, out_tensor, shape_info)
        else:
            batchmatmul_4d_kernel(a_tensor, b_tensor, out_tensor, shape_info)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        pypto_t = t1 - t0
        pypto_m = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        # Torch 参考实现（与官方一致：转 fp32 再 matmul）
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        golden = torch.matmul(a_tensor.to(torch.float32), b_tensor.to(torch.float32))
        torch.npu.synchronize()
        t1 = time.perf_counter()
        torch_t = t1 - t0
        torch_m = torch.npu.max_memory_allocated(device) / (1024 ** 2)

        # 正确性
        assert torch.allclose(
            out_tensor.detach().cpu().to(torch.float32),
            golden.detach().cpu().to(torch.float32),
            atol=atol,
            rtol=rtol,
        ), f"[{case_name}] round={r+1} output mismatch"

        # 仅统计正式测试轮次
        if r >= warmup_rounds:
            pypto_times.append(pypto_t)
            pypto_mems.append(pypto_m)
            torch_times.append(torch_t)
            torch_mems.append(torch_m)
            print(
                f"Round {r+1:02d} | "
                f"PyPTO: {pypto_t*1000:.2f} ms / {pypto_m:.1f} MB | "
                f"Torch: {torch_t*1000:.2f} ms / {torch_m:.1f} MB"
            )

        del a_tensor, b_tensor, out_tensor, golden
        torch.npu.empty_cache()

    avg_pypto_t = _mean(pypto_times)
    avg_pypto_m = _mean(pypto_mems)
    avg_torch_t = _mean(torch_times)
    avg_torch_m = _mean(torch_mems)

    print(f"[CASE-SUMMARY] {case_name}")
    print(f"  Torch: {avg_torch_t*1000:.2f} ms | {avg_torch_m:.1f} MB")
    print(f"  PyPTO: {avg_pypto_t*1000:.2f} ms | {avg_pypto_m:.1f} MB")
    if avg_pypto_t > 0:
        print(f"  Speedup(Torch/PyPTO): {avg_torch_t / avg_pypto_t:.2f}x")

    return {
        "case_name": case_name,
        "avg_pypto_t": avg_pypto_t,
        "avg_pypto_m": avg_pypto_m,
        "avg_torch_t": avg_torch_t,
        "avg_torch_m": avg_torch_m,
    }


@pytest.mark.soc("950", "910")
def test_batchmatmul_pypto_vs_torch_multi_size_perf():
    """
    多 shape、多轮（warmup + 正式）比较 PyPTO batchmatmul 与 Torch 实现。
    可通过环境变量调轮次：
      - BMM_WARMUP_ROUNDS (默认 3)
      - BMM_NUM_ROUNDS    (默认 10)
    """
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    torch_npu.npu.config.allow_internal_format = True
    device = f"npu:{device_id}"

    warmup_rounds = int(os.environ.get("BMM_WARMUP_ROUNDS", 3))
    num_rounds = int(os.environ.get("BMM_NUM_ROUNDS", 10))
    rtol = float(os.environ.get("BMM_RTOL", 1e-3))
    atol = float(os.environ.get("BMM_ATOL", 1e-3))

    # 尽量选规则尺寸，降低 tile/view 边界影响，便于稳定对比
    cases = [
        # 3D BMM
        {"name": "3d_b1_m128_k256_n512", "ndim": 3, "b": 1, "m": 128, "k": 256, "n": 512},
        {"name": "3d_b4_m256_k256_n256", "ndim": 3, "b": 4, "m": 256, "k": 256, "n": 256},
        {"name": "3d_b8_m128_k512_n256", "ndim": 3, "b": 8, "m": 128, "k": 512, "n": 256},
        {"name": "3d_b1_m6144_k6144_n6144", "ndim": 3, "b": 1, "m": 6144, "k": 6144, "n": 6144},
        # 4D BMM
        {"name": "4d_b1_2_m128_k256_n512", "ndim": 4, "b1": 1, "b2": 2, "m": 128, "k": 256, "n": 512},
        {"name": "4d_b2_2_m256_k256_n256", "ndim": 4, "b1": 2, "b2": 2, "m": 256, "k": 256, "n": 256},
        {"name": "4d_b2_4_m128_k512_n256", "ndim": 4, "b1": 2, "b2": 4, "m": 128, "k": 512, "n": 256},
        {"name": "4d_b1_1_m6144_k6144_n6144", "ndim": 4, "b1": 1, "b2": 1, "m": 6144, "k": 6144, "n": 6144},
    ]

    print(
        f"\n🚀 BatchMatMul PyPTO vs Torch 多 size 多轮测试 "
        f"(warmup={warmup_rounds}, rounds={num_rounds}) on {device}"
    )

    all_stats = []
    for case in cases:
        stats = _run_one_case_multi_round(
            case=case,
            device=device,
            num_rounds=num_rounds,
            warmup_rounds=warmup_rounds,
            rtol=rtol,
            atol=atol,
        )
        all_stats.append(stats)

    # 全局汇总（按 case 的均值再平均）
    avg_pypto_t = _mean([x["avg_pypto_t"] for x in all_stats])
    avg_torch_t = _mean([x["avg_torch_t"] for x in all_stats])
    avg_pypto_m = _mean([x["avg_pypto_m"] for x in all_stats])
    avg_torch_m = _mean([x["avg_torch_m"] for x in all_stats])

    print("\n" + "=" * 90)
    print("📈 [GLOBAL SUMMARY] BatchMatMul PyPTO vs Torch (across multi-size cases)")
    print(f"Torch: {avg_torch_t*1000:.2f} ms | {avg_torch_m:.1f} MB")
    print(f"PyPTO: {avg_pypto_t*1000:.2f} ms | {avg_pypto_m:.1f} MB")
    if avg_pypto_t > 0:
        print(f"Speedup(PyPTO speed / Torch speed): {avg_torch_t / avg_pypto_t:.2f}x")
    print("=" * 90)