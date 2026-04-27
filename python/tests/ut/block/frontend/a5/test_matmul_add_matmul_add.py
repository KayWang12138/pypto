# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static fused matmul + add test with manual synchronization."""

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm


@fe.kernel
def matmul_add_matmul_add(
    q: pl.Tensor[[64, 64], pl.FP32],
    k: pl.Tensor[[64, 64], pl.FP32],
    v: pl.Tensor[[64, 64], pl.FP32],
    x1: pl.Tensor[[64, 64], pl.FP32],
    x2: pl.Tensor[[64, 64], pl.FP32],
    out: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    tile_p_vec = plm.TileType(shape=[32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    mm1_res = plm.make_tile(tile_p_vec, addr=0x0000, size=8192)

    tile_p_vec = plm.TileType(shape=[32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    mm2_res = plm.make_tile(tile_p_vec, addr=0x2000, size=8192)

    tile_v1_mat = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
    v1_mat = plm.make_tile(tile_v1_mat, addr=0x10000, size=16384)

    with pl.section_cube():
        tile_q_mat = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
        q_mat = plm.make_tile(tile_q_mat, addr=0x0000, size=16384)

        tile_k_mat = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
        k_mat = plm.make_tile(tile_k_mat, addr=0x4000, size=16384)
        v_mat = plm.make_tile(tile_k_mat, addr=0x8000, size=16384)

        tile_q_left = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Left, blayout=2, slayout=1)
        q_left = plm.make_tile(tile_q_left, addr=0x0000, size=16384)

        tile_k_right = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2)
        k_right = plm.make_tile(tile_k_right, addr=0x0000, size=16384)
        v_right = plm.make_tile(tile_k_right, addr=0x4000, size=16384)  # kv addr not conflict, no need sync

        tile_c1_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024)
        tile_c1 = plm.make_tile(tile_c1_type, addr=0x0000, size=16384)
        tile_c2 = plm.make_tile(tile_c1_type, addr=0x8000, size=16384)

        plm.load(q_mat, q, [0, 0])
        plm.load(k_mat, k, [0, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        plm.move(q_left, q_mat)
        plm.move(k_right, k_mat)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        plm.matmul(tile_c1, q_left, k_right)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.move(mm1_res, tile_c1, acc_to_vec_mode="dual_split_m")  # ACC -> UB
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

        plm.load(v_mat, v, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        plm.move(v_right, v_mat)

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2)
        plm.move(q_left, v1_mat)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        plm.matmul(tile_c2, q_left, v_right)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.move(mm2_res, tile_c2, acc_to_vec_mode="dual_split_m")  # ACC -> UB
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)

    with pl.section_vector():
        sub_core = pl.block.get_subblock_idx()
        sub_index = pl.block.index_cast(sub_core)
        tile_type_x1 = plm.TileType(shape=[32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        tile_x1 = plm.make_tile(tile_type_x1, addr=0x4000, size=8192)
        tile_x2 = plm.make_tile(tile_type_x1, addr=0x6000, size=8192)

        tile_type_out = plm.TileType(shape=[32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        tile_out = plm.make_tile(tile_type_out, addr=0x8000, size=8192)

        tile_type_nz = plm.TileType(shape=[33, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[32, 64], blayout=2, slayout=1)
        tile_nz = plm.make_tile(tile_type_nz, addr=0xA000, size=8448)  # NZ no bank conflict

        off = sub_index * 32
        plm.load(tile_x1, x1, [off, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)

        plm.add(tile_out, mm1_res, tile_x1)
        plm.move(tile_nz, tile_out)  # ND2NZ

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        plm.insert(v1_mat, tile_nz, offset=off * 32)  # UB2L1 NZ2NZ

        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

        plm.load(tile_x2, x2, [off, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=1)

        plm.add(tile_out, mm2_res, tile_x2)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)

        plm.store(out, tile_out, [off, 0])

    return out


@fe.jit()
def test_matmul_add_matmul_add():
    compiled_lib = fe.compile(matmul_add_matmul_add, arch="a5", codegen_mode="cce")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)

    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        print(f"Currrent device is not Ascend950, skip.")
        return
    shape = [64, 64]
    torch.manual_seed(0)
    dtype = torch.float32

    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    x1 = torch.randn(shape, device=device, dtype=dtype)
    x2 = torch.randn(shape, device=device, dtype=dtype)
    out = torch.zeros(shape, device=device, dtype=dtype)

    fe.launch(None, 1, compiled_lib, q, k, v, x1, x2, out)
    torch.npu.synchronize()

    c1 = torch.matmul(q, k)
    v1 = c1 + x1
    c2 = torch.matmul(v1, v)
    out_ref = c2 + x2

    print("***********npu output***********")
    print(out.shape, out.dtype)
    print(out)
    print("***********golden output***********")
    print(out_ref.shape, out_ref.dtype)
    print(out_ref)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("result equal!")


if __name__ == "__main__":
    test_matmul_add_matmul_add()
    print("\nAll tests passed!")
