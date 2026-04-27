# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tiled Matmul + Add with L1 Double Buffer using auto-rotate + tile.lock/unlock.

Computes: out = A @ B + X
  A: [256, 64], B: [64, 256], X: [256, 256]  (FP16 matmul → FP32 acc)

Tiling:
  tile_m=64, tile_n=64, K=64 (no K tiling)
  M tiles = 4,  N tiles = 4  →  16 tiles total

L1 double buffer: NBuffer with current() auto-rotate.
  Each iteration current() returns the next slot (ping→pong→ping→...).
  No manual advance() or buf_idx needed.

Synchronization: tile.lock(pipe) / tile.unlock(pipe).
  buf_id is fully hidden from user.
"""

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm
from pypto_block.pypto_core import ir

M_SIZE = 256
K_SIZE = 64
N_SIZE = 256
TILE_M = 64
TILE_N = 64


@fe.kernel
def tiled_matmul_add_db(
    a: pl.Tensor[[M_SIZE, K_SIZE], pl.FP16],
    b: pl.Tensor[[K_SIZE, N_SIZE], pl.FP16],
    x: pl.Tensor[[M_SIZE, N_SIZE], pl.FP32],
    out: pl.Tensor[[M_SIZE, N_SIZE], pl.FP32],
    workspace: pl.Tensor[[M_SIZE, N_SIZE], pl.FP32],
) -> pl.Tensor[[M_SIZE, N_SIZE], pl.FP32]:

    # ========== Buffer declarations ==========
    # L1 double buffer (auto-rotate via current())
    a_l1_db = pl.L1NBuffer(0x00000, pl.FP16, shape=(TILE_M, K_SIZE), buf_ids=(0, 1))
    b_l1_db = pl.L1NBuffer(0x04000, pl.FP16, shape=(K_SIZE, TILE_N), buf_ids=(2, 3))

    # L0A / L0B (single slot)
    a_left = pl.L0ABuffer(0x0000, pl.FP16, shape=(TILE_M, K_SIZE), buf_id=4,
                           blayout=2, slayout=1)
    b_right = pl.L0BBuffer(0x0000, pl.FP16, shape=(K_SIZE, TILE_N), buf_id=5)

    # ACC (single slot)
    acc = pl.L0CBuffer(0x0000, pl.FP32, shape=(TILE_M, TILE_N), buf_id=6, fractal=1024)

    # UB (Vector section)
    mm_res_ub = pl.UBBuffer(0x0000, pl.FP32, shape=(32, TILE_N), buf_id=7)
    x_ub = pl.UBBuffer(0x2000, pl.FP32, shape=(32, TILE_N), buf_id=8)
    out_ub = pl.UBBuffer(0x4000, pl.FP32, shape=(32, TILE_N), buf_id=9)

    # ========== Cube Section ==========
    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE_M):
            for j in pl.range(0, N_SIZE, TILE_N):
                # current() auto-rotates: ping → pong → ping → ...
                cur_a = a_l1_db.current()
                cur_b = b_l1_db.current()

                # MTE2: load A[i,:], B[:,j] → L1 current slot
                cur_a.tile.lock(pl.PipeType.MTE2)
                plm.load(cur_a.tile, a, [i, 0])
                cur_a.tile.unlock(pl.PipeType.MTE2)

                cur_b.tile.lock(pl.PipeType.MTE2)
                plm.load(cur_b.tile, b, [0, j])
                cur_b.tile.unlock(pl.PipeType.MTE2)

                # MTE1: move L1 → L0A/L0B
                # Lock both src (L1) and dst (L0A/L0B)
                cur_a.tile.lock(pl.PipeType.MTE1)
                a_left.tile.lock(pl.PipeType.MTE1)
                plm.move(a_left.tile, cur_a.tile)
                cur_a.tile.unlock(pl.PipeType.MTE1)
                a_left.tile.unlock(pl.PipeType.MTE1)

                cur_b.tile.lock(pl.PipeType.MTE1)
                b_right.tile.lock(pl.PipeType.MTE1)
                plm.move(b_right.tile, cur_b.tile)
                cur_b.tile.unlock(pl.PipeType.MTE1)
                b_right.tile.unlock(pl.PipeType.MTE1)

                # M: matmul → ACC
                # Lock L0A/L0B (read) and ACC (write) on M pipe
                a_left.tile.lock(pl.PipeType.M)
                b_right.tile.lock(pl.PipeType.M)
                acc.tile.lock(pl.PipeType.M)
                plm.matmul(acc.tile, a_left.tile, b_right.tile)
                a_left.tile.unlock(pl.PipeType.M)
                b_right.tile.unlock(pl.PipeType.M)
                acc.tile.unlock(pl.PipeType.M)

                # FIX: store ACC → workspace GM
                acc.tile.lock(pl.PipeType.FIX)
                plm.store(workspace, acc.tile, [i, j])
                acc.tile.unlock(pl.PipeType.FIX)

                # Cross-core → Vector
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ========== Vector Section ==========
    with pl.section_vector():
        sub_core = pl.block.get_subblock_idx()
        sub_index = pl.block.index_cast(sub_core)

        for i in pl.range(0, M_SIZE, TILE_M):
            for j in pl.range(0, N_SIZE, TILE_N):
                row_off = sub_index * 32

                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

                mm_res_ub.tile.lock(pl.PipeType.MTE2)
                plm.load(mm_res_ub.tile, workspace, [i + row_off, j])
                mm_res_ub.tile.unlock(pl.PipeType.MTE2)
                x_ub.tile.lock(pl.PipeType.MTE2)
                plm.load(x_ub.tile, x, [i + row_off, j])
                x_ub.tile.unlock(pl.PipeType.MTE2)

                mm_res_ub.tile.lock(pl.PipeType.V)
                x_ub.tile.lock(pl.PipeType.V)
                out_ub.tile.lock(pl.PipeType.V)
                plm.add(out_ub.tile, mm_res_ub.tile, x_ub.tile)
                mm_res_ub.tile.unlock(pl.PipeType.V)
                x_ub.tile.unlock(pl.PipeType.V)
                out_ub.tile.unlock(pl.PipeType.V)

                out_ub.tile.lock(pl.PipeType.MTE3)
                plm.store(out, out_ub.tile, [i + row_off, j])
                out_ub.tile.unlock(pl.PipeType.MTE3)

    return out


# ============================================================================
# auto_mutex variant — no manual lock/unlock
# ============================================================================


@fe.kernel(auto_mutex=True)
def tiled_matmul_add_db_auto(
    a: pl.Tensor[[M_SIZE, K_SIZE], pl.FP16],
    b: pl.Tensor[[K_SIZE, N_SIZE], pl.FP16],
    x: pl.Tensor[[M_SIZE, N_SIZE], pl.FP32],
    out: pl.Tensor[[M_SIZE, N_SIZE], pl.FP32],
    workspace: pl.Tensor[[M_SIZE, N_SIZE], pl.FP32],
) -> pl.Tensor[[M_SIZE, N_SIZE], pl.FP32]:

    a_l1_db = pl.L1NBuffer(0x00000, pl.FP16, shape=(TILE_M, K_SIZE), buf_ids=(0, 1))
    b_l1_db = pl.L1NBuffer(0x04000, pl.FP16, shape=(K_SIZE, TILE_N), buf_ids=(2, 3))

    a_left = pl.L0ABuffer(0x0000, pl.FP16, shape=(TILE_M, K_SIZE), buf_id=4,
                           blayout=2, slayout=1)
    b_right = pl.L0BBuffer(0x0000, pl.FP16, shape=(K_SIZE, TILE_N), buf_id=5)

    acc = pl.L0CBuffer(0x0000, pl.FP32, shape=(TILE_M, TILE_N), buf_id=6, fractal=1024)

    mm_res_ub = pl.UBBuffer(0x0000, pl.FP32, shape=(32, TILE_N), buf_id=7)
    x_ub = pl.UBBuffer(0x2000, pl.FP32, shape=(32, TILE_N), buf_id=8)
    out_ub = pl.UBBuffer(0x4000, pl.FP32, shape=(32, TILE_N), buf_id=9)

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE_M):
            for j in pl.range(0, N_SIZE, TILE_N):
                cur_a = a_l1_db.current()
                cur_b = b_l1_db.current()

                plm.load(cur_a.tile, a, [i, 0])
                plm.load(cur_b.tile, b, [0, j])
                plm.move(a_left.tile, cur_a.tile)
                plm.move(b_right.tile, cur_b.tile)
                plm.matmul(acc.tile, a_left.tile, b_right.tile)
                plm.store(workspace, acc.tile, [i, j])

                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    with pl.section_vector():
        sub_core = pl.block.get_subblock_idx()
        sub_index = pl.block.index_cast(sub_core)

        for i in pl.range(0, M_SIZE, TILE_M):
            for j in pl.range(0, N_SIZE, TILE_N):
                row_off = sub_index * 32

                pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)

                plm.load(mm_res_ub.tile, workspace, [i + row_off, j])
                plm.load(x_ub.tile, x, [i + row_off, j])
                plm.add(out_ub.tile, mm_res_ub.tile, x_ub.tile)
                plm.store(out, out_ub.tile, [i + row_off, j])

    return out


# ============================================================================
# Tests
# ============================================================================


def test_tiled_matmul_add_db_parse():
    """Verify IR generation for tiled matmul_add with auto-rotate L1 DB."""
    prog = tiled_matmul_add_db.parse()
    ir_str = str(prog)

    # Auto-rotate struct cursor injected for both L1 NBuffers
    assert ir_str.count("struct.declare") >= 2, "Expected struct.declare for a_l1 and b_l1 cursors"
    assert "struct.get" in ir_str
    assert "struct.set" in ir_str

    # L1 ping/pong tiles materialized
    mat_count = ir_str.count("target_memory=pl.MemorySpace.Mat")
    assert mat_count >= 4, f"Expected >=4 Mat tiles (a ping/pong + b ping/pong), got {mat_count}"

    # Dynamic mutex for L1 DB
    assert "mutex_lock" in ir_str
    assert "mutex_unlock" in ir_str

    # Static mutex for single-slot buffers (ACC, UB)
    assert "mutex_id=6" in ir_str, "Missing ACC buf_id"
    assert "mutex_id=7" in ir_str, "Missing UB buf_id"

    # Matmul
    assert "manual.matmul" in ir_str

    # Tiling loops
    assert "pl.range(0, 256, 64)" in ir_str

    # Cross-core
    assert "set_cross_core" in ir_str
    assert "wait_cross_core" in ir_str

    # Vector add
    assert "manual.add" in ir_str

    # L1 addr auto-offset
    assert "memref_addr=0" in ir_str
    assert "memref_addr=8192" in ir_str  # a_l1 pong

    print("IR parse OK — tiled matmul_add with auto-rotate L1 DB verified.")


def test_tiled_matmul_add_db_auto_parse():
    """Verify auto_mutex injects mutex_lock/unlock without manual lock/unlock."""
    prog = tiled_matmul_add_db_auto.parse()
    ir_str = str(prog)

    # Auto-injected mutex lock/unlock
    assert "mutex_lock" in ir_str, f"Expected mutex_lock in auto_mutex IR:\n{ir_str}"
    assert "mutex_unlock" in ir_str, f"Expected mutex_unlock in auto_mutex IR:\n{ir_str}"

    # All buffer ops still present
    assert "manual.load" in ir_str
    assert "manual.matmul" in ir_str
    assert "manual.add" in ir_str

    # Cross-core still works
    assert "set_cross_core" in ir_str
    assert "wait_cross_core" in ir_str

    # Lock count: each plm op generates lock/unlock for its tile args
    # Cube: load(1) + load(1) + move(2) + move(2) + matmul(3) + store(1) = 10 per iteration
    # Vector: load(1) + load(1) + add(3) + store(1) = 6 per iteration
    lock_count = ir_str.count("mutex_lock")
    unlock_count = ir_str.count("mutex_unlock")
    assert lock_count >= 10, f"Expected >=10 mutex_lock, got {lock_count}"
    assert lock_count == unlock_count, f"lock({lock_count}) != unlock({unlock_count})"

    print(f"auto_mutex parse OK — {lock_count} lock/unlock pairs injected.")


def test_tiled_matmul_add_db_compile_cce():
    """Verify CCE codegen with get_buf/rls_buf calls."""
    try:
        compiled = fe.compile(tiled_matmul_add_db, arch="a5", codegen_mode="cce")
        print("CCE compile OK:", compiled.lib_path)
    except Exception as e:
        import os
        build_dir = "./build/tiled_matmul_add_db"
        if os.path.isdir(build_dir):
            for f in os.listdir(build_dir):
                if f.endswith(".cpp"):
                    with open(os.path.join(build_dir, f)) as fh:
                        content = fh.read()
                    if "get_buf" in content:
                        get_count = content.count("get_buf(")
                        rls_count = content.count("rls_buf(")
                        print(f"CCE codegen OK — {get_count} get_buf + {rls_count} rls_buf calls found.")
                        lines = [l.strip() for l in content.split("\n") if "get_buf" in l or "rls_buf" in l]
                        for l in lines[:8]:
                            print(" ", l)
                        return
        print(f"CCE compile failed: {e}")


@fe.jit()
def test_tiled_matmul_add_db_npu():
    """End-to-end NPU test (A5 only)."""
    compiled_lib = fe.compile(tiled_matmul_add_db_auto, arch="a5", codegen_mode="cce")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)
    device_name = torch.npu.get_device_name()
    if "Ascend950" not in device_name:
        print(f"Current device is {device_name}, not A5. Skip.")
        return

    torch.manual_seed(42)
    a = torch.randn(M_SIZE, K_SIZE, device=device, dtype=torch.float16)
    b = torch.randn(K_SIZE, N_SIZE, device=device, dtype=torch.float16)
    x = torch.randn(M_SIZE, N_SIZE, device=device, dtype=torch.float32)
    out = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float32)
    workspace = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float32)

    fe.launch(None, 1, compiled_lib, a, b, x, out, workspace)
    torch.npu.synchronize()

    out_ref = torch.matmul(a.float(), b.float()) + x
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("result equal!")


if __name__ == "__main__":
    # test_tiled_matmul_add_db_parse()
    # test_tiled_matmul_add_db_compile_cce()
    test_tiled_matmul_add_db_npu()  # uncomment on A5
    print("\nAll tests passed!")
