#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------
import os
import argparse
import torch
import pypto
import numpy as np
from numpy.testing import assert_allclose

FP32_ALIGN = 8
CHANNEL_TILE = 16
DEFAULT_TEST_SHAPES = [
    (2400, 88, 1, 1),
    (400, 8, 1, 1),
    (800, 40, 2, 1),
    (900, 45, 1, 1),
]


def get_device_id():
    """获取 NPU 设备 ID"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("警告: 未设置 TILE_FWK_DEVICE_ID 环境变量，请确保已配置 NPU 环境。")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        return None


def align_up(value, align):
    return ((value + align - 1) // align) * align


def select_bn_reduce_tiles(shape):
    n, c, h, w = shape
    hw = h * w
    tile_hw = max(FP32_ALIGN, align_up(hw, FP32_ALIGN))

    if n >= 512:
        target_flat_tile = 512
    else:
        target_flat_tile = 256

    tile_n = max(32, target_flat_tile // tile_hw)
    if tile_n >= 128:
        tile_n = 128
    elif tile_n >= 64:
        tile_n = 64
    else:
        tile_n = 32

    tile_reduce = tile_n * tile_hw
    out_tile_c = 64 if c > 64 else 16
    return tile_n, CHANNEL_TILE, tile_hw, tile_reduce, out_tile_c


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "stitch_function_num_initial": 128,
        "stitch_function_outcast_memory": 1024,
        "stitch_function_inner_memory": 1024,
    },
    debug_options=dict(compile_debug_mode=1, runtime_debug_mode=1),
)
def bn_reduce_kernel(
    x: pypto.Tensor([], pypto.DT_FP32),
    sum_out: pypto.Tensor([], pypto.DT_FP32),
    sq_sum_out: pypto.Tensor([], pypto.DT_FP32),
    tile_n=128,
    tile_c=16,
    tile_hw=8,
    tile_reduce=1024,
    out_tile_c=64,
):
    n, c, h, w = x.shape
    reduce_axis_size = n * h * w

    v1 = pypto.reshape(x, [n, c, h * w], inplace=True)

    pypto.set_vec_tile_shapes(tile_n, tile_c, tile_hw)
    v1_sq = v1 * v1

    v2 = pypto.transpose(v1, 1, 2)
    v2_sq = pypto.transpose(v1_sq, 1, 2)

    v3 = pypto.reshape(v2, [reduce_axis_size, c], inplace=True)
    v3_sq = pypto.reshape(v2_sq, [reduce_axis_size, c], inplace=True)

    pypto.set_vec_tile_shapes(tile_reduce, tile_c)
    sum_raw = pypto.sum(v3, dim=0, keepdim=True)
    sq_sum_raw = pypto.sum(v3_sq, dim=0, keepdim=True)

    pypto.set_vec_tile_shapes(1, out_tile_c, 1, 8)
    final_sum = pypto.reshape(sum_raw, [1, c, 1, 1], inplace=True)
    final_sq_sum = pypto.reshape(sq_sum_raw, [1, c, 1, 1], inplace=True)

    sum_out.move(final_sum)
    sq_sum_out.move(final_sq_sum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", type=str, default="npu", choices=["npu"])
    parser.parse_args()

    device_id = get_device_id()
    if device_id is not None:
        import torch_npu

        torch.npu.set_device(device_id)

    device = f"npu:{device_id}"

    test_shapes = DEFAULT_TEST_SHAPES

    failed_cases = []

    for i, shape in enumerate(test_shapes):
        try:
            tile_n, tile_c, tile_hw, tile_reduce, out_tile_c = select_bn_reduce_tiles(shape)

            # 1. 准备输入数据 (使用均分分布，避免 FP32 平方在大 Shape 时因极值产生 Inf/NaN)
            x_torch = torch.rand(shape, dtype=torch.float32, device=device)

            # 2. 准备输出 tensor
            c = shape[1]
            output_shape = (1, c, 1, 1)
            sum_out = torch.empty(output_shape, dtype=torch.float32, device=device)
            sq_sum_out = torch.empty(output_shape, dtype=torch.float32, device=device)

            # 3. 执行自定义 Kernel
            bn_reduce_kernel(
                x_torch,
                sum_out,
                sq_sum_out,
                tile_n=tile_n,
                tile_c=tile_c,
                tile_hw=tile_hw,
                tile_reduce=tile_reduce,
                out_tile_c=out_tile_c,
            )

            # 4. 计算 Golden Reference
            golden_sum = torch.sum(x_torch, dim=(0, 2, 3), keepdim=True)
            golden_sq_sum = torch.sum(x_torch * x_torch, dim=(0, 2, 3), keepdim=True)

            # 5. 精度校验
            actual_sum_np = sum_out.cpu().numpy()
            golden_sum_np = golden_sum.cpu().numpy()
            sum_diff = np.max(np.abs(actual_sum_np - golden_sum_np))

            actual_sq_sum_np = sq_sum_out.cpu().numpy()
            golden_sq_sum_np = golden_sq_sum.cpu().numpy()
            sq_sum_diff = np.max(np.abs(actual_sq_sum_np - golden_sq_sum_np))

            assert_allclose(actual_sum_np, golden_sum_np, rtol=1e-3, atol=1e-3)
            assert_allclose(actual_sq_sum_np, golden_sq_sum_np, rtol=1e-3, atol=1e-3)
            print(
                f"Case {i + 1}: sum_diff={sum_diff:.6e}, "
                f"sq_sum_diff={sq_sum_diff:.6e}"
            )

        except Exception as e:
            print(f"  ✗ Case {i + 1} Failed: {e}")
            failed_cases.append((i + 1, shape))

    print(f"\n{'=' * 20} 测试总结 {'=' * 20}")
    if not failed_cases:
        print("全部测试通过。")
    else:
        print(f"⚠️ 存在 {len(failed_cases)} 个失败用例:")
        for idx, shp in failed_cases:
            print(f"   - Case {idx}: {shp}")


if __name__ == "__main__":
    main()
