# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Simplified frontend runtime example for plm.store(..., relu_pre_mode=...) on A5 PTO."""

import torch
import torch_npu

import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm


def _make_q(device: str) -> torch.Tensor:
    base = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    return (base.remainder(9) - 4.0).to(device)


def _make_k(device: str) -> torch.Tensor:
    base = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    return ((base * 3).remainder(7) - 3.0).to(device)


@fe.kernel
def store_relu_kernel(
    q: pl.Tensor[[64, 64], pl.FP32],
    k: pl.Tensor[[64, 64], pl.FP32],
    raw_out: pl.Tensor[[64, 64], pl.FP32],
    relu_out: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.section_cube():
        mat_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
        q_mat = plm.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = plm.make_tile(mat_type, addr=0x4000, size=16384)

        left_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Left, blayout=2, slayout=1)
        q_left = plm.make_tile(left_type, addr=0x0000, size=16384)

        right_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2)
        k_right = plm.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024)
        acc = plm.make_tile(acc_type, addr=0x0000, size=16384)

        plm.load(q_mat, q, [0, 0])
        plm.load(k_mat, k, [0, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        plm.move(q_left, q_mat)
        plm.move(k_right, k_mat)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        plm.matmul(acc, q_left, k_right)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

        plm.store(raw_out, acc, [0, 0])
        plm.store(relu_out, acc, [0, 0], relu_pre_mode="normal_relu")
        pl.system.bar_all()
        plm.printf("STORE_RELU raw_out[0:8,0:8]\\n")
        plm.dump_tensor(raw_out, offsets=[0, 0], shapes=[8, 8])
        plm.printf("STORE_RELU relu_out[0:8,0:8]\\n")
        plm.dump_tensor(relu_out, offsets=[0, 0], shapes=[8, 8])

    return relu_out


@fe.jit()
def test_store_relu():
    compiled_lib = fe.compile(store_relu_kernel, arch="a5", codegen_mode="pto")
    if compiled_lib is None:
        raise RuntimeError("compile failed for store_relu_kernel")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        print(f"Current device is {device_name}, skip.")
        return

    q = _make_q(device)
    k = _make_k(device)
    raw_out = torch.zeros((64, 64), device=device, dtype=torch.float32)
    relu_out = torch.zeros((64, 64), device=device, dtype=torch.float32)

    fe.launch(None, 1, compiled_lib, q, k, raw_out, relu_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    relu_ref = torch.relu(raw_ref)

    print("***********raw acc->gm output (top-left 8x8)***********")
    print(raw_out[:8, :8])
    print("***********relu store output (top-left 8x8)***********")
    print(relu_out[:8, :8])
    print("***********golden raw output (top-left 8x8)***********")
    print(raw_ref[:8, :8])
    print("***********golden relu output (top-left 8x8)***********")
    print(relu_ref[:8, :8])

    torch.testing.assert_close(raw_out, raw_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(relu_out, relu_ref, rtol=1e-2, atol=1e-2)
    print("result equal!")


if __name__ == "__main__":
    test_store_relu()
    print("\nAll tests passed!")
