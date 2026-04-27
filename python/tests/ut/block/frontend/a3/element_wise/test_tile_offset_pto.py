"""Test tile[offset] element offset syntax in PTO codegen pipeline.

Uses codegen_mode="pto" (default) to exercise PTOCodegen::VisitExpr_(TileOffsetExpr),
which lowers `tile[offset]` to:

    %off_i64 = arith.index_cast %offset : index to i64
    %byte_off = arith.muli %off_i64, %elem_bytes : i64
    %new_addr = arith.addi %base_addr, %byte_off : i64
    %tmp = pto.alloc_tile addr = %new_addr : !pto.tile_buf<...>

Kernel description:
  Input shape  [4, 64] FP32.
  Output shape [4, 64] FP32.
  Four [1, 64] FP32 Vec tiles back a single big tile with element offset access.
  Per-row compute (row index `r` ∈ [0, 4)):
    t0  = x[r]           (load)
    t1  = y[r]           (load)
    sum = t0 + t1        (plm.add)   → uses tile[r*64] as out
    dif = t1 - t0        (plm.sub)   → uses tile[r*64] as out/in
    prod = sum * dif     (plm.mul)
    scaled = prod * 0.5  (plm.muls)
    z[r] = scaled        (store)

Algebraically z[r] = (y² - x²) * 0.5; the reference uses this identity directly.
"""

import math

import torch
import torch_npu
import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm


ROWS = 4
COLS = 64
ROW_SIZE = COLS * 4         # bytes per [1, COLS] FP32 row
BUF_SIZE = ROWS * ROW_SIZE  # bytes for the full [1, ROWS * COLS] tile
SCALE = 0.5


@fe.kernel
def tile_offset_pto_kernel(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Tensor[[ROWS, COLS], pl.FP32],
    z: pl.Tensor[[ROWS, COLS], pl.FP32],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    tile_type = plm.TileType(shape=[1, COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)

    # Four independent big tiles, each covering ROWS logical rows via element offset.
    buf_x = plm.make_tile(tile_type, addr=0x0000, size=BUF_SIZE)
    buf_y = plm.make_tile(tile_type, addr=0x0400, size=BUF_SIZE)
    buf_sum = plm.make_tile(tile_type, addr=0x0800, size=BUF_SIZE)
    buf_diff = plm.make_tile(tile_type, addr=0x0c00, size=BUF_SIZE)

    with pl.section_vector():
        for r in pl.range(0, ROWS):
            off = r * COLS

            # ----- load phase -----
            plm.load(buf_x[off], x, [r, 0])
            plm.load(buf_y[off], y, [r, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

            # ----- compute phase: use tile[offset] on both ins() and outs() -----
            plm.add(buf_sum[off], buf_x[off], buf_y[off])   # sum  = x + y
            plm.sub(buf_diff[off], buf_y[off], buf_x[off])  # diff = y - x
            pl.system.bar_v()
            plm.mul(buf_sum[off], buf_sum[off], buf_diff[off])  # sum = sum * diff  (y^2 - x^2)
            pl.system.bar_v()
            plm.muls(buf_sum[off], buf_sum[off], SCALE)         # sum *= 0.5

            # ----- store phase -----
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            plm.store(z, buf_sum[off], [r, 0])
    return z


def tile_offset_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference: z = (y^2 - x^2) * SCALE, computed exactly as the kernel does."""
    t_sum = x + y
    t_diff = y - x
    prod = t_sum * t_diff
    return prod * SCALE


@fe.jit()
def test_tile_offset_pto():
    compiled = fe.compile(tile_offset_pto_kernel, arch="a3")
    print("compiled:", compiled.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)
    torch.manual_seed(42)

    x = torch.randn((ROWS, COLS), device=device, dtype=torch.float32)
    y = torch.randn((ROWS, COLS), device=device, dtype=torch.float32)
    z = torch.zeros((ROWS, COLS), device=device, dtype=torch.float32)

    fe.launch(None, 1, compiled, x, y, z)
    torch.npu.synchronize()

    z_ref = tile_offset_ref(x, y)
    diff = (z - z_ref).abs().max().item()
    print(f"max|diff| = {diff:.6g}")
    torch.testing.assert_close(z, z_ref, rtol=1e-5, atol=1e-5)
    print("PASS: tile[offset] PTO pipeline (add/sub/mul/muls)")


if __name__ == "__main__":
    test_tile_offset_pto()
