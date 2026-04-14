# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime example for plm.store(..., fp_tile=...) on A5 CCE."""

import struct

import torch
import torch_npu

import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm


FP_SCALE_VALUE = 2.0
FP_SCALE_SIGNED_INT8_FLAG = 1 << 46
FP_SCALE_BITS = FP_SCALE_SIGNED_INT8_FLAG | 0x40000000  # signed-int8 + float32 bit-pattern for 2.0f


def _make_q(device: str) -> torch.Tensor:
    row = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32).repeat(8)
    return row.unsqueeze(0).repeat(64, 1)


def _make_k(device: str) -> torch.Tensor:
    return torch.eye(64, device=device, dtype=torch.float32)


def _make_fp_params(device: str) -> torch.Tensor:
    return torch.full((1, 64), FP_SCALE_BITS, device=device, dtype=torch.int64)


def _decode_fp_param(value: int) -> tuple[float, int, int]:
    raw = int(value) & ((1 << 64) - 1)
    scale_bits = raw & 0xFFFFFFFF
    scale = struct.unpack("!f", struct.pack("!I", scale_bits))[0]
    signed_flag = (raw >> 46) & 0x1
    return scale, signed_flag, raw


@fe.kernel(auto_sync=False)
def store_fp_cce_kernel(
    q: pl.Tensor[[64, 64], pl.FP32],
    k: pl.Tensor[[64, 64], pl.FP32],
    fp_params: pl.Tensor[[1, 64], pl.INT64],
    raw_out: pl.Tensor[[64, 64], pl.FP32],
    fp_out: pl.Tensor[[64, 64], pl.INT8],
) -> pl.Tensor[[64, 64], pl.INT8]:
    with pl.section_cube():
        mat_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
        q_mat = plm.make_tile(mat_type, addr=0x0000, size=16384)
        k_mat = plm.make_tile(mat_type, addr=0x4000, size=16384)

        fp_mat_type = plm.TileType(
            shape=[1, 64],
            dtype=pl.INT64,
            target_memory=pl.MemorySpace.Mat,
            blayout=1,
            slayout=0,
        )
        fp_mat = plm.make_tile(fp_mat_type, addr=0x8000, size=512)

        left_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Left, blayout=2, slayout=1)
        q_left = plm.make_tile(left_type, addr=0x0000, size=16384)

        right_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2)
        k_right = plm.make_tile(right_type, addr=0x0000, size=16384)

        acc_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024)
        acc = plm.make_tile(acc_type, addr=0x0000, size=16384)

        fp_type = plm.TileType(shape=[1, 64], dtype=pl.INT64, target_memory=pl.MemorySpace.Scaling)
        fp_tile = plm.make_tile(fp_type, addr=0x0000, size=512)

        plm.load(q_mat, q, [0, 0])
        plm.load(k_mat, k, [0, 0])
        plm.load(fp_mat, fp_params, [0, 0])

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
        plm.move(fp_tile, fp_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.FIX, event_id=1)
        plm.store(fp_out, acc, [0, 0], fp_tile=fp_tile)
        pl.system.bar_all()

    return fp_out


@fe.jit()
def test_store_fp_cce():
    compiled_lib = fe.compile(store_fp_cce_kernel, arch="a5", codegen_mode="cce")
    if compiled_lib is None:
        raise RuntimeError("compile failed for store_fp_cce_kernel")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        print(f"Current device is {device_name}, skip.")
        return

    q = _make_q(device)
    k = _make_k(device)
    fp_params = _make_fp_params(device)
    raw_out = torch.zeros((64, 64), device=device, dtype=torch.float32)
    fp_out = torch.zeros((64, 64), device=device, dtype=torch.int8)

    fe.launch(None, 1, compiled_lib, q, k, fp_params, raw_out, fp_out)
    torch.npu.synchronize()

    raw_ref = torch.matmul(q, k)
    expected_fp = torch.clamp(torch.round(raw_ref * FP_SCALE_VALUE), -128, 127).to(torch.int8)
    decoded_fp = [_decode_fp_param(v) for v in fp_params[0, :8].tolist()]

    print("***********cce q input (top-left 8x8)***********")
    print(q[:8, :8])
    print("***********cce fp tile source bits (top-left 1x8)***********")
    print(fp_params[:, :8])
    print("***********cce decoded fp tile payload (scale, signed_int8_flag, raw_u64)***********")
    print(decoded_fp)
    print("***********cce raw acc->gm output (top-left 8x8)***********")
    print(raw_out[:8, :8])
    print("***********cce expected store_fp int8 output***********")
    print(expected_fp[:8, :8])
    print("***********cce store_fp int8 output (top-left 8x8)***********")
    print(fp_out[:8, :8])
    print("***********cce store_fp summary***********")
    print({"fp_params_dtype": str(fp_params.dtype), "fp_out_dtype": str(fp_out.dtype)})

    torch.testing.assert_close(raw_out, raw_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fp_out.to(torch.int32), expected_fp.to(torch.int32), rtol=0, atol=0)
    print("cce raw store and store_fp outputs both match expectations.")


if __name__ == "__main__":
    test_store_fp_cce()
    print("\nAll tests passed!")
