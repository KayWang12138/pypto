# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for BufferManager (Buffer/NBuffer) and pl.mutex.lock/unlock.

Covers:
    * NBuffer address calculation (auto-offset per slot)
    * BufferSlot.current()/advance() cursor semantics
    * pl.mutex.lock/unlock IR generation (static mutex_id)
    * NBuffer used as closure variable with tiles passed into kernel
    * NBuffer declared inside kernel body (pl.NBuffer(...) special parsing)
    * PTO codegen output contains pto.get_buf / pto.rls_buf
"""

import pytest

import pypto_block.frontend as fe
import pypto_block.language as pl
import pypto_block.language.op.manual as plm
from pypto_block.pypto_core import ir


# ---------------------------------------------------------------------------
# Pure-Python tests for Buffer / NBuffer (no IR context needed)
# ---------------------------------------------------------------------------


class TestNBufferAddressing:
    """NBuffer should auto-compute slot addresses from base_addr + i*slot_size."""

    def test_default_slot_size(self):
        nbuf = pl.NBuffer(
            shape=[128, 128],
            dtype=pl.FP16,
            memory=pl.MemorySpace.Mat,
            base_addr=0x00000,
            buf_ids=(0, 1),
            blayout=2,
            slayout=1,
        )
        # 128 * 128 * 2 bytes = 32768 per slot
        assert nbuf.slot_size == 32768
        assert nbuf.num_slots == 2
        assert nbuf.buf_ids == (0, 1)
        assert nbuf.slots[0].addr == 0x00000
        assert nbuf.slots[1].addr == 0x08000

    def test_explicit_size_override(self):
        nbuf = pl.NBuffer(
            shape=[64, 64],
            dtype=pl.FP16,
            memory=pl.MemorySpace.Vec,
            base_addr=0x1000,
            buf_ids=(4, 5, 6),
            size=16384,  # custom (not the 8192 default)
        )
        assert nbuf.slot_size == 16384
        assert nbuf.num_slots == 3
        assert [s.addr for s in nbuf.slots] == [0x1000, 0x5000, 0x9000]

    def test_buf_id_range_check(self):
        with pytest.raises(ValueError, match="0, 31"):
            pl.NBuffer(
                shape=[64], dtype=pl.FP16, memory=pl.MemorySpace.Vec,
                base_addr=0, buf_ids=(0, 32),
            )

    def test_buf_id_uniqueness(self):
        with pytest.raises(ValueError, match="unique"):
            pl.NBuffer(
                shape=[64], dtype=pl.FP16, memory=pl.MemorySpace.Vec,
                base_addr=0, buf_ids=(1, 1),
            )

    def test_cursor_advance(self):
        nbuf = pl.NBuffer(
            shape=[64], dtype=pl.FP16, memory=pl.MemorySpace.Vec,
            base_addr=0, buf_ids=(10, 11),
        )
        s0 = nbuf.current()
        assert s0.buf_id == 10
        nbuf.advance()
        s1 = nbuf.current()
        assert s1.buf_id == 11
        nbuf.advance()
        s2 = nbuf.current()
        assert s2.buf_id == 10  # wraps
        nbuf.reset()
        s3 = nbuf.current()
        assert s3.buf_id == 10


class TestBufferSingleSlot:
    def test_single_buffer_props(self):
        buf = pl.Buffer(
            shape=[64, 64],
            dtype=pl.FP32,
            memory=pl.MemorySpace.Vec,
            addr=0x0400,
            buf_id=3,
        )
        assert buf.addr == 0x0400
        assert buf.buf_id == 3
        # 64 * 64 * 4 = 16384
        assert buf.size == 16384
        assert buf.dtype == pl.FP32
        assert buf.memory == pl.MemorySpace.Vec

    def test_single_buffer_buf_id_range(self):
        with pytest.raises(ValueError, match="0, 31"):
            pl.Buffer(
                shape=[64], dtype=pl.FP16, memory=pl.MemorySpace.Vec,
                addr=0, buf_id=-1,
            )


# ---------------------------------------------------------------------------
# Kernel-level tests — parse + codegen
# ---------------------------------------------------------------------------


def _ir_to_str(prog: ir.Program) -> str:
    return str(prog)


class TestMutexLockIR:
    """pl.mutex.lock/unlock should produce system.mutex_{lock,unlock} IR calls."""

    def test_static_mutex_lock_unlock(self):
        @fe.kernel(auto_sync=False)
        def k(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            t = plm.make_tile(tt, addr=0, size=128)
            pl.mutex.lock(pl.PipeType.MTE2, 0)
            plm.load(t, x, [0])
            pl.mutex.unlock(pl.PipeType.MTE2, 0)
            return x

        ir_str = _ir_to_str(k.parse())
        assert "system.mutex_lock" in ir_str, (
            f"Expected system.mutex_lock in IR:\n{ir_str}"
        )
        assert "system.mutex_unlock" in ir_str, (
            f"Expected system.mutex_unlock in IR:\n{ir_str}"
        )

    def test_keyword_form(self):
        @fe.kernel(auto_sync=False)
        def k(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            t = plm.make_tile(tt, addr=0, size=128)
            pl.mutex.lock(pipe=pl.PipeType.MTE2, mutex_id=1)
            plm.load(t, x, [0])
            pl.mutex.unlock(pipe=pl.PipeType.MTE2, mutex_id=1)
            return x

        ir_str = _ir_to_str(k.parse())
        assert "system.mutex_lock" in ir_str
        assert "system.mutex_unlock" in ir_str


class TestNBufferWithKernel:
    """NBuffer declared outside kernel used as closure variable."""

    def test_nbuffer_closure_tiles(self):
        """Slots' tiles accessible as closure vars inside kernel."""
        nbuf = pl.NBuffer(
            shape=[128], dtype=pl.FP16,
            memory=pl.MemorySpace.Vec,
            base_addr=0, buf_ids=(0, 1),
        )
        tile_ping = nbuf.slots[0].tile
        tile_pong = nbuf.slots[1].tile

        @fe.kernel(auto_sync=False)
        def k(x: pl.Tensor[[128], pl.FP16]) -> pl.Tensor[[128], pl.FP16]:
            plm.load(tile_ping, x, [0])
            plm.load(tile_pong, x, [0])
            return x

        ir_str = _ir_to_str(k.parse())
        assert ir_str.count("block.make_tile") >= 2, ir_str

    def test_nbuffer_with_mutex(self):
        """NBuffer slots used with pl.mutex.lock/unlock."""
        nbuf = pl.NBuffer(
            shape=[64], dtype=pl.FP16,
            memory=pl.MemorySpace.Vec,
            base_addr=0, buf_ids=(2, 3),
        )
        tile0 = nbuf.slots[0].tile
        bid0 = nbuf.buf_ids[0]

        @fe.kernel(auto_sync=False)
        def k(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
            pl.mutex.lock(pl.PipeType.MTE2, bid0)
            plm.load(tile0, x, [0])
            pl.mutex.unlock(pl.PipeType.MTE2, bid0)
            return x

        ir_str = _ir_to_str(k.parse())
        assert "system.mutex_lock" in ir_str
        assert "system.mutex_unlock" in ir_str
        assert "mutex_id=2" in ir_str, f"Expected mutex_id=2, got:\n{ir_str}"

    def test_nbuffer_inside_kernel(self):
        """NBuffer declared inside kernel body (parser special-casing)."""

        @fe.kernel(auto_sync=False)
        def k(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
            nbuf = pl.NBuffer(
                shape=[64], dtype=pl.FP16,
                memory=pl.MemorySpace.Vec,
                base_addr=0, buf_ids=(4, 5),
            )
            t = nbuf.slots[0].tile
            pl.mutex.lock(pl.PipeType.MTE2, 4)
            plm.load(t, x, [0])
            pl.mutex.unlock(pl.PipeType.MTE2, 4)
            return x

        ir_str = _ir_to_str(k.parse())
        assert "system.mutex_lock" in ir_str
        assert "block.make_tile" in ir_str

    def test_nbuffer_address_auto_offset_in_ir(self):
        """Verify that slot 0/1 have different memref_addr in generated IR."""
        nbuf = pl.NBuffer(
            shape=[128, 128], dtype=pl.FP16,
            memory=pl.MemorySpace.Mat,
            base_addr=0, buf_ids=(0, 1),
            blayout=2, slayout=1,
        )
        tile_ping = nbuf.slots[0].tile
        tile_pong = nbuf.slots[1].tile

        @fe.kernel(auto_sync=False)
        def k(
            a: pl.Tensor[[128, 128], pl.FP16],
        ) -> pl.Tensor[[128, 128], pl.FP16]:
            plm.load(tile_ping, a, [0, 0])
            plm.load(tile_pong, a, [0, 0])
            return a

        ir_str = _ir_to_str(k.parse())
        assert "memref_addr=0" in ir_str
        assert "memref_addr=32768" in ir_str


class TestMutexCodegen:
    """Verify the PTO backend lowers system.mutex_lock/unlock to pto.get_buf/rls_buf."""

    def test_pto_get_rls_buf_emitted(self):
        @fe.kernel(auto_sync=False)
        def k(
            x: pl.Tensor[[64], pl.FP16],
            out: pl.Tensor[[64], pl.FP16],
        ) -> pl.Tensor[[64], pl.FP16]:
            tt = plm.TileType(shape=[64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
            t = plm.make_tile(tt, addr=0, size=128)
            pl.mutex.lock(pl.PipeType.MTE2, 0)
            plm.load(t, x, [0])
            pl.mutex.unlock(pl.PipeType.MTE2, 0)
            plm.store(out, t, [0])
            return out

        try:
            compiled = fe.compile(k, arch="a5")
        except Exception as exc:
            pytest.skip(f"PTO compile unavailable in this environment: {exc}")
        mlir_str = getattr(compiled, "pto_ir", None) or getattr(compiled, "ir_str", None) or ""
        if not mlir_str:
            pytest.skip("Compiled object does not expose PTO MLIR text")
        assert "pto.get_buf" in mlir_str, mlir_str
        assert "pto.rls_buf" in mlir_str, mlir_str
        assert "TLOAD" in mlir_str, mlir_str


class TestCanndslStyleAPI:
    """canndsl-style memory-specific subclasses and current()/advance() pattern."""

    def test_l1nbuffer_construction(self):
        db = pl.L1NBuffer(0, pl.FP16, shape=(32, 32), buf_ids=(0, 1))
        assert db.num_slots == 2
        assert db.slot_size == 32 * 32 * 2  # 2048
        assert db.slots[0].addr == 0
        assert db.slots[1].addr == 2048

    def test_all_subclass_memory_spaces(self):
        from pypto_block.pypto_core.ir import MemorySpace
        assert pl.UBNBuffer(0, pl.FP16, shape=(8,), buf_ids=(0, 1)).slots[0].memory == MemorySpace.Vec
        assert pl.L1NBuffer(0, pl.FP16, shape=(8,), buf_ids=(0, 1)).slots[0].memory == MemorySpace.Mat
        assert pl.L0ANBuffer(0, pl.FP16, shape=(8,), buf_ids=(0, 1)).slots[0].memory == MemorySpace.Left
        assert pl.L0BNBuffer(0, pl.FP16, shape=(8,), buf_ids=(0, 1)).slots[0].memory == MemorySpace.Right
        assert pl.L0CNBuffer(0, pl.FP32, shape=(8,), buf_ids=(0, 1)).slots[0].memory == MemorySpace.Acc

    def test_current_advance_tensor_alias(self):
        db = pl.UBNBuffer(0, pl.FP32, shape=(64,), buf_ids=(10, 11))
        s0 = db.current()
        assert s0.buf_id == 10
        assert s0.tensor is s0.tile  # canndsl compat alias
        db.advance()
        assert db.current().buf_id == 11
        db.advance()
        assert db.current().buf_id == 10  # wraps

    def test_canndsl_style_in_kernel(self):
        """L1NBuffer declared in kernel body — current() auto-rotates via IR cursor."""
        @fe.kernel(auto_sync=False)
        def k(gm_q: pl.Tensor[[32, 32], pl.FP16]) -> pl.Tensor[[32, 32], pl.FP16]:
            q_l1_db = pl.L1NBuffer(0, pl.FP16, shape=(32, 32), buf_ids=(0, 1))
            # Each current() auto-rotates: first → slot0, second → slot1
            cur = q_l1_db.current()
            cur.lock(pl.PipeType.MTE2)
            plm.load(cur.tile, gm_q, [0, 0])
            cur.unlock(pl.PipeType.MTE2)
            cur2 = q_l1_db.current()
            cur2.lock(pl.PipeType.MTE2)
            plm.load(cur2.tile, gm_q, [0, 0])
            cur2.unlock(pl.PipeType.MTE2)
            return gm_q

        ir_str = _ir_to_str(k.parse())
        # Dynamic mutex via struct cursor — uses _dyn variant
        assert "mutex_lock" in ir_str
        assert "mutex_unlock" in ir_str
        # Both tiles materialized (ping + pong in the tile tuple)
        assert ir_str.count("block.make_tile") >= 2
        # struct cursor injected
        assert "struct.declare" in ir_str
        assert "struct.set" in ir_str

    def test_fa_style_multi_buffer(self):
        """Multiple NBuffers across different memory spaces — FA-like pattern."""
        q_l1_db = pl.L1NBuffer(0, pl.FP16, shape=(32, 32), buf_ids=(0, 1))
        k_l1_db = pl.L1NBuffer(4 * 1024, pl.FP16, shape=(32, 32), buf_ids=(2, 3))
        ub_db = pl.UBNBuffer(0, pl.FP32, shape=(32, 32), buf_ids=(4, 5))

        q_tile = q_l1_db.slots[0].tile
        k_tile = k_l1_db.slots[0].tile
        ub_tile = ub_db.slots[0].tile

        @fe.kernel(auto_sync=False)
        def k(
            gm_q: pl.Tensor[[32, 32], pl.FP16],
            gm_k: pl.Tensor[[32, 32], pl.FP16],
        ) -> pl.Tensor[[32, 32], pl.FP16]:
            pl.mutex.lock(pl.PipeType.MTE2, 0)
            plm.load(q_tile, gm_q, [0, 0])
            pl.mutex.unlock(pl.PipeType.MTE2, 0)
            pl.mutex.lock(pl.PipeType.MTE2, 2)
            plm.load(k_tile, gm_k, [0, 0])
            pl.mutex.unlock(pl.PipeType.MTE2, 2)
            return gm_q

        ir_str = _ir_to_str(k.parse())
        assert ir_str.count("system.mutex_lock") >= 2
        assert ir_str.count("block.make_tile") >= 2

    def test_loop_auto_rotate(self):
        """For-loop with current() — auto-rotates each iteration via IR cursor.

        In a pl.range loop, each call to ``current()`` generates a dynamic
        tile/buf_id selection based on ``struct.get(cursor) % num_slots``,
        followed by ``struct.set(cursor, cursor+1)``. The cursor is managed
        by the struct loop-threading mechanism.
        """

        @fe.kernel(auto_sync=False)
        def k(gm_q: pl.Tensor[[128, 32], pl.FP16]) -> pl.Tensor[[128, 32], pl.FP16]:
            q_l1_db = pl.L1NBuffer(0, pl.FP16, shape=(32, 32), buf_ids=(0, 1))
            for i in pl.range(0, 4):
                cur = q_l1_db.current()
                cur.lock(pl.PipeType.MTE2)
                plm.load(cur.tile, gm_q, [i * 32, 0])
                cur.unlock(pl.PipeType.MTE2)
            return gm_q

        ir_str = _ir_to_str(k.parse())
        # Both tiles materialized (dynamic selection)
        assert ir_str.count("block.make_tile") == 2, ir_str
        # IR cursor struct
        assert "struct.declare" in ir_str
        assert "struct.get" in ir_str
        assert "struct.set" in ir_str
        # Dynamic mutex
        assert "mutex_lock" in ir_str

    def test_straight_line_auto_rotate(self):
        """Straight-line current() calls — each advances the IR cursor."""
        @fe.kernel(auto_sync=False)
        def k(gm_q: pl.Tensor[[32, 32], pl.FP16]) -> pl.Tensor[[32, 32], pl.FP16]:
            q_l1_db = pl.L1NBuffer(0, pl.FP16, shape=(32, 32), buf_ids=(0, 1))
            # Two current() calls → cursor goes 0→1
            cur0 = q_l1_db.current()
            cur0.lock(pl.PipeType.MTE2)
            plm.load(cur0.tile, gm_q, [0, 0])
            cur0.unlock(pl.PipeType.MTE2)
            cur1 = q_l1_db.current()
            cur1.lock(pl.PipeType.MTE2)
            plm.load(cur1.tile, gm_q, [0, 0])
            cur1.unlock(pl.PipeType.MTE2)
            return gm_q

        ir_str = _ir_to_str(k.parse())
        # Both slots materialized
        assert ir_str.count("block.make_tile") == 2, ir_str
        assert "memref_addr=0" in ir_str
        assert "memref_addr=2048" in ir_str

    def test_single_slot_nbuffer_current(self):
        """Single-slot NBuffer: current() works without struct cursor."""
        @fe.kernel(auto_sync=False)
        def k(gm_q: pl.Tensor[[32, 32], pl.FP16]) -> pl.Tensor[[32, 32], pl.FP16]:
            q_l1 = pl.L1NBuffer(0, pl.FP16, shape=(32, 32), buf_ids=(5,))
            cur = q_l1.current()
            cur.lock(pl.PipeType.MTE2)
            plm.load(cur.tile, gm_q, [0, 0])
            cur.unlock(pl.PipeType.MTE2)
            return gm_q

        ir_str = _ir_to_str(k.parse())
        assert "block.make_tile" in ir_str
        assert "mutex_lock" in ir_str
        assert "mutex_id=5" in ir_str
        # No struct cursor for single-slot
        assert "struct.declare" not in ir_str

    def test_previous_in_kernel(self):
        """previous() returns the slot before current cursor position."""
        @fe.kernel(auto_sync=False)
        def k(gm_q: pl.Tensor[[64, 32], pl.FP16]) -> pl.Tensor[[64, 32], pl.FP16]:
            q_l1_db = pl.L1NBuffer(0, pl.FP16, shape=(32, 32), buf_ids=(0, 1))
            # First current() → slot 0, advances cursor to 1
            cur = q_l1_db.current()
            cur.lock(pl.PipeType.MTE2)
            plm.load(cur.tile, gm_q, [0, 0])
            cur.unlock(pl.PipeType.MTE2)
            # previous() → slot 0 (cursor-1 = 0)
            prev = q_l1_db.previous()
            prev.lock(pl.PipeType.MTE1)
            plm.load(prev.tile, gm_q, [0, 0])
            prev.unlock(pl.PipeType.MTE1)
            return gm_q

        ir_str = _ir_to_str(k.parse())
        # Both tiles materialized
        assert ir_str.count("block.make_tile") >= 2, ir_str
        # Both lock/unlock present
        assert ir_str.count("mutex_lock") >= 2
        assert ir_str.count("mutex_unlock") >= 2
        # previous() should NOT advance cursor (no extra struct.set)
        # current() emits one struct.set; previous() emits zero
        set_count = ir_str.count("struct.set")
        assert set_count == 1, f"Expected 1 struct.set (from current only), got {set_count}"


class TestAutoMutex:
    """auto_mutex=True should inject mutex_lock/unlock around buffer tile ops."""

    def test_auto_mutex_single_buffer(self):
        """Single-slot Buffer: load auto-injects lock/unlock on MTE2."""
        buf = pl.UBBuffer(0, pl.FP16, shape=(64,), buf_id=3)

        @fe.kernel(auto_sync=False, auto_mutex=True)
        def k(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
            plm.load(buf.tile, x, [0])
            plm.store(x, buf.tile, [0])
            return x

        ir_str = _ir_to_str(k.parse())
        # load → MTE2 lock/unlock
        assert ir_str.count("mutex_lock") >= 2, ir_str
        assert ir_str.count("mutex_unlock") >= 2, ir_str
        assert "mutex_id=3" in ir_str, ir_str

    def test_auto_mutex_nbuffer_loop(self):
        """NBuffer in loop with auto_mutex: auto-rotate + auto lock/unlock."""
        @fe.kernel(auto_sync=False, auto_mutex=True)
        def k(gm_q: pl.Tensor[[128, 32], pl.FP16]) -> pl.Tensor[[128, 32], pl.FP16]:
            q_l1_db = pl.L1NBuffer(0, pl.FP16, shape=(32, 32), buf_ids=(0, 1))
            for i in pl.range(0, 4):
                cur = q_l1_db.current()
                plm.load(cur.tile, gm_q, [i * 32, 0])
            return gm_q

        ir_str = _ir_to_str(k.parse())
        # Auto-injected mutex lock/unlock
        assert "mutex_lock" in ir_str, ir_str
        assert "mutex_unlock" in ir_str, ir_str
        # Both tiles materialized (dynamic selection)
        assert ir_str.count("block.make_tile") == 2, ir_str

    def test_auto_mutex_multi_tile_op(self):
        """Op with multiple buffer tiles: all get lock/unlock."""
        buf_a = pl.UBBuffer(0, pl.FP16, shape=(64,), buf_id=0)
        buf_b = pl.UBBuffer(0x100, pl.FP16, shape=(64,), buf_id=1)
        buf_c = pl.UBBuffer(0x200, pl.FP16, shape=(64,), buf_id=2)

        @fe.kernel(auto_sync=False, auto_mutex=True)
        def k(x: pl.Tensor[[64], pl.FP16]) -> pl.Tensor[[64], pl.FP16]:
            plm.load(buf_a.tile, x, [0])
            plm.load(buf_b.tile, x, [0])
            plm.add(buf_c.tile, buf_a.tile, buf_b.tile)
            return x

        ir_str = _ir_to_str(k.parse())
        # add op: 3 tiles → 3 lock + 3 unlock
        # load ops: 1 tile each → 1 lock + 1 unlock each
        assert ir_str.count("mutex_lock") >= 5, f"Expected >=5 mutex_lock, got:\n{ir_str}"
        assert ir_str.count("mutex_unlock") >= 5, f"Expected >=5 mutex_unlock, got:\n{ir_str}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
