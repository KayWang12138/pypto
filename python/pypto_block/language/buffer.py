# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Declarative Buffer / NBuffer descriptors backed by A5 Mutex (buffer_id).

These classes provide an ergonomic, canndsl-style declaration of single-slot
and multi-slot tile buffers. Address offsets across slots are computed
automatically; each slot carries a dedicated MutexID for synchronization.

Typical usage::

    import pypto_block.language as pl
    import pypto_block.language.op.manual as plm

    a_db = pl.L1NBuffer(0x00000, pl.FP16, shape=(128, 128), buf_ids=(0, 1))

    # auto-rotate cursor with buffer-level lock/unlock (recommended)
    cur = a_db.current()
    cur.lock(pl.PipeType.MTE2)
    plm.load(cur.tile, a, [i, k])
    cur.unlock(pl.PipeType.MTE2)

    # dynamic index via tile tuple — reuses existing tile_buf[buf_idx]
    tile_buf = a_db.tiles           # (ping, pong)
    buf_idx = (k // 128) % 2
    plm.load(tile_buf[buf_idx], a, [i, k])

``Buffer`` is the single-slot variant. Both are pure Python — IR tile nodes
are created lazily the first time a slot's ``.tile`` is accessed, so these
descriptors can be declared inside a ``@fe.kernel`` body without leaking
state between compilations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Optional, Sequence

from pypto_block.pypto_core import DataType
from pypto_block.pypto_core.ir import MemorySpace

import pypto_block.language.op.manual as plm


__all__ = [
    "Buffer", "NBuffer", "BufferSlot", "_TileRef", "TileSpec",
    "UBBuffer", "L1Buffer", "L0ABuffer", "L0BBuffer", "L0CBuffer",
    "UBNBuffer", "L1NBuffer", "L0ANBuffer", "L0BNBuffer", "L0CNBuffer",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _product(shape: Sequence[int]) -> int:
    return reduce(lambda a, b: a * b, shape, 1)


def _dtype_bytes(dtype: DataType) -> int:
    """Return the element width of ``dtype`` in bytes (rounded up for sub-byte).

    Uses ``DataType.get_bit()`` from the C++ binding when available, falling
    back to a conservative table for common types.
    """
    if hasattr(dtype, "get_bit"):
        bits = int(dtype.get_bit())
        return max(1, (bits + 7) // 8)
    _width = {
        DataType.BOOL: 1,
        DataType.INT8: 1,
        DataType.UINT8: 1,
        DataType.INT16: 2,
        DataType.UINT16: 2,
        DataType.FP16: 2,
        DataType.BF16: 2,
        DataType.INT32: 4,
        DataType.UINT32: 4,
        DataType.FP32: 4,
        DataType.INT64: 8,
        DataType.UINT64: 8,
    }
    if dtype in _width:
        return _width[dtype]
    raise ValueError(f"Cannot determine element size for dtype={dtype!r}")


def _default_slot_size(shape: Sequence[int], dtype: DataType) -> int:
    """Default slot byte size: product(shape) * sizeof(dtype)."""
    return _product(shape) * _dtype_bytes(dtype)


# ---------------------------------------------------------------------------
# TileSpec — shape/dtype/layout attrs without address info
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    """Tile type attributes (no address, shared across Buffer/NBuffer slots)."""

    shape: Sequence[int]
    dtype: DataType
    memory: MemorySpace
    blayout: Optional[int] = None
    slayout: Optional[int] = None
    fractal: Optional[int] = None
    pad: Optional[int] = None
    compact: Optional[int] = None
    valid_shape: Optional[Sequence[int]] = None

    def make_tile_type(self) -> "plm.TileType":
        """Construct a fresh ``plm.TileType`` from this spec.

        Each call returns a new instance — layout defaulting in
        :class:`plm.TileType.__post_init__` mutates the object, so we cannot
        safely cache one across slots.
        """
        return plm.TileType(
            shape=list(self.shape),
            dtype=self.dtype,
            target_memory=self.memory,
            valid_shape=list(self.valid_shape) if self.valid_shape is not None else None,
            blayout=self.blayout,
            slayout=self.slayout,
            fractal=self.fractal,
            pad=self.pad,
            compact=self.compact,
        )


# ---------------------------------------------------------------------------
# _TileRef — tile with bound buf_id, supports .lock() / .unlock()
# ---------------------------------------------------------------------------


class _TileRef:
    """Tile wrapper with bound ``buf_id`` for Mutex lock/unlock.

    Returned by ``Buffer.tile`` and ``BufferSlot.tile``. Wraps the
    underlying plm Tile IR node (passed to ``plm.load``, ``plm.move``,
    etc. via ``unwrap()``).

    Prefer calling ``lock()`` / ``unlock()`` on the parent ``BufferSlot``
    or ``Buffer`` rather than on this object directly::

        cur = nbuf.current()
        cur.lock(pl.PipeType.MTE2)       # buffer-level sync
        plm.load(cur.tile, tensor, [...]) # tile data access
        cur.unlock(pl.PipeType.MTE2)
    """

    def __init__(self, tile, buf_id, buf_id_values=None, memory=None, var_name=None):
        self._tile = tile        # plm.Tile (IR Call node)
        self._buf_id = buf_id    # int, IR Expr, or None (no Mutex)
        self._buf_id_values = buf_id_values  # tuple of ints or None
        self._memory = memory    # MemorySpace — for auto_mutex pipe inference
        self._ir_var = None      # cached IR Var from _ensure_tile_has_var
        self._var_name = var_name  # Python source name for readable codegen

    @property
    def has_buf_id(self) -> bool:
        return self._buf_id is not None

    def lock(self, pipe):
        """Acquire Mutex for the bound ``buf_id`` on ``pipe``. No-op if no buf_id."""
        if self._buf_id is None:
            return None
        from pypto_block.ir.op.system_ops import mutex_lock
        return mutex_lock(pipe, self._buf_id, buf_id_values=self._buf_id_values)

    def unlock(self, pipe):
        """Release Mutex for the bound ``buf_id`` on ``pipe``. No-op if no buf_id."""
        if self._buf_id is None:
            return None
        from pypto_block.ir.op.system_ops import mutex_unlock
        return mutex_unlock(pipe, self._buf_id, buf_id_values=self._buf_id_values)

    def unwrap(self):
        """Extract the underlying IR Expr for parser/codegen consumption."""
        if hasattr(self._tile, "unwrap"):
            return self._tile.unwrap()
        return self._tile

    @property
    def buf_id(self):
        return self._buf_id

    def __repr__(self):
        return f"_TileRef(buf_id={self._buf_id})"


# ---------------------------------------------------------------------------
# BufferSlot — (tile, buf_id) view returned by NBuffer.current()
# ---------------------------------------------------------------------------


class BufferSlot:
    """Lightweight view of a single NBuffer slot.

    Synchronization is done at the buffer level, data access via ``.tile``::

        cur = nbuf.current()
        cur.lock(pl.PipeType.MTE2)
        plm.load(cur.tile, tensor, [i, 0])
        cur.unlock(pl.PipeType.MTE2)

    Attributes:
        tile: Underlying :class:`_TileRef` — pass to ``plm.load`` / ``plm.move`` etc.
        buf_id: The MutexID (convenience access).
        tensor: Alias for ``tile`` (canndsl compatibility).
    """

    __slots__ = ("_tile", "_buf_id")

    def __init__(self, tile, buf_id):
        if not isinstance(tile, _TileRef):
            tile = _TileRef(tile, buf_id)
        self._tile = tile
        self._buf_id = buf_id

    def lock(self, pipe):
        """Acquire Mutex for this buffer slot on ``pipe``."""
        return self._tile.lock(pipe)

    def unlock(self, pipe):
        """Release Mutex for this buffer slot on ``pipe``."""
        return self._tile.unlock(pipe)

    @property
    def tile(self) -> _TileRef:
        return self._tile

    @property
    def tensor(self) -> _TileRef:
        """Alias for ``tile`` — canndsl compatibility."""
        return self._tile

    @property
    def buf_id(self):
        return self._buf_id


# ---------------------------------------------------------------------------
# Buffer — single-slot descriptor
# ---------------------------------------------------------------------------


class Buffer:
    """Single-slot buffer descriptor bound to a ``buf_id``.

    The underlying ``plm`` tile is created lazily on first access to
    :pyattr:`tile`, so a Buffer can be declared anywhere inside a kernel body.

    Args:
        shape: Tile shape.
        dtype: Element data type.
        memory: Target memory space (Vec / Mat / Left / Right / Acc).
        addr: Byte offset within ``memory``.
        buf_id: MutexID for this buffer (0-31), or None for no Mutex sync.
        size: Byte size of the buffer (defaults to ``product(shape) * sizeof(dtype)``).
        blayout/slayout/fractal/pad/compact/valid_shape: Optional tile attributes
            forwarded to :class:`plm.TileType`.
    """

    def __init__(
        self,
        shape: Sequence[int],
        dtype: DataType,
        memory: MemorySpace,
        addr: int,
        buf_id: Optional[int] = None,
        *,
        size: Optional[int] = None,
        blayout: Optional[int] = None,
        slayout: Optional[int] = None,
        fractal: Optional[int] = None,
        pad: Optional[int] = None,
        compact: Optional[int] = None,
        valid_shape: Optional[Sequence[int]] = None,
    ):
        if buf_id is not None:
            if not isinstance(buf_id, int) or buf_id < 0 or buf_id > 31:
                raise ValueError(
                    f"Buffer.buf_id must be an int in [0, 31] (Mutex ISASI range), got {buf_id!r}"
                )
        self._spec = TileSpec(
            shape=tuple(shape),
            dtype=dtype,
            memory=memory,
            blayout=blayout,
            slayout=slayout,
            fractal=fractal,
            pad=pad,
            compact=compact,
            valid_shape=tuple(valid_shape) if valid_shape is not None else None,
        )
        self._addr = int(addr)
        self._buf_id = int(buf_id) if buf_id is not None else None
        self._size = int(size) if size is not None else _default_slot_size(shape, dtype)
        self._tile = None  # materialized on first `.tile` access
        self._var_name: str | None = None  # set by parser from Python LHS name

    # --- introspection ------------------------------------------------------

    @property
    def spec(self) -> TileSpec:
        return self._spec

    @property
    def addr(self) -> int:
        return self._addr

    @property
    def buf_id(self) -> int | None:
        return self._buf_id

    @property
    def size(self) -> int:
        return self._size

    @property
    def memory(self) -> MemorySpace:
        return self._spec.memory

    @property
    def dtype(self) -> DataType:
        return self._spec.dtype

    @property
    def shape(self) -> Sequence[int]:
        return self._spec.shape

    # --- tile materialization ----------------------------------------------

    def lock(self, pipe):
        """Acquire Mutex for this buffer on ``pipe``."""
        return self.tile.lock(pipe)

    def unlock(self, pipe):
        """Release Mutex for this buffer on ``pipe``."""
        return self.tile.unlock(pipe)

    @property
    def tile(self) -> _TileRef:
        """Return a :class:`_TileRef` for this buffer, creating the underlying plm tile on first use."""
        if self._tile is None:
            raw_tile = plm.make_tile(self._spec.make_tile_type(), addr=self._addr, size=self._size)
            self._tile = _TileRef(raw_tile, self._buf_id, memory=self._spec.memory,
                                  var_name=self._var_name)
        return self._tile

    def set_validshape(self, row, col) -> None:
        """Apply ``plm.set_validshape`` to this buffer's tile."""
        plm.set_validshape(self.tile.unwrap(), row, col)


# ---------------------------------------------------------------------------
# NBuffer — N-slot descriptor with cursor
# ---------------------------------------------------------------------------


class NBuffer:
    """N-slot buffer descriptor for multi-buffer (ping-pong, triple, …) patterns.

    Each slot has its own address, automatically offset by
    ``slot_index * slot_size`` from ``base_addr``. Optionally, each slot can
    be bound to a ``buf_id`` for Mutex synchronization.

    Args:
        shape: Per-slot tile shape.
        dtype: Element data type.
        memory: Target memory space.
        base_addr: Byte offset of slot 0.
        buf_ids: Tuple of MutexIDs, one per slot. Determines num_slots.
            Mutually exclusive with ``num_slots`` (provide one or the other).
        num_slots: Number of buffer slots when no Mutex sync is needed.
            Mutually exclusive with ``buf_ids``.
        size: Per-slot byte size (defaults to ``product(shape) * sizeof(dtype)``).
        blayout/slayout/fractal/pad/compact/valid_shape: Optional tile attributes.

    Example::

        # With Mutex sync (buf_ids determines slot count):
        a_db = pl.NBuffer(shape=[128, 128], dtype=pl.FP16,
                          memory=pl.MemorySpace.Mat, base_addr=0,
                          buf_ids=(0, 1))

        # Without Mutex sync (num_slots determines slot count):
        tmp_db = pl.NBuffer(shape=[64, 128], dtype=pl.FP32,
                            memory=pl.MemorySpace.Vec, base_addr=0x1000,
                            num_slots=2)
    """

    def __init__(
        self,
        shape: Sequence[int],
        dtype: DataType,
        memory: MemorySpace,
        base_addr: int,
        buf_ids: Optional[Sequence[int]] = None,
        *,
        num_slots: Optional[int] = None,
        size: Optional[int] = None,
        blayout: Optional[int] = None,
        slayout: Optional[int] = None,
        fractal: Optional[int] = None,
        pad: Optional[int] = None,
        compact: Optional[int] = None,
        valid_shape: Optional[Sequence[int]] = None,
    ):
        # Resolve slot count from buf_ids and/or num_slots
        if buf_ids is not None and num_slots is not None:
            buf_ids = tuple(buf_ids)
            if len(buf_ids) != num_slots:
                raise ValueError(
                    f"buf_ids length ({len(buf_ids)}) must match num_slots ({num_slots})"
                )
        elif buf_ids is not None:
            buf_ids = tuple(buf_ids)
            num_slots = len(buf_ids)
        elif num_slots is not None:
            if num_slots < 1:
                raise ValueError(f"num_slots must be >= 1, got {num_slots}")
            buf_ids = None
        else:
            raise ValueError("NBuffer requires either buf_ids or num_slots")

        if buf_ids is not None:
            if len(buf_ids) == 0:
                raise ValueError("NBuffer requires at least one buf_id")
            for bid in buf_ids:
                if not isinstance(bid, int) or bid < 0 or bid > 31:
                    raise ValueError(
                        f"NBuffer.buf_ids entries must be ints in [0, 31], got {bid!r}"
                    )
            if len(set(buf_ids)) != len(buf_ids):
                raise ValueError(f"NBuffer.buf_ids must be unique, got {buf_ids}")

        self._buf_ids = buf_ids  # None when no Mutex sync
        self._num_slots = num_slots
        self._base_addr = int(base_addr)
        self._slot_size = int(size) if size is not None else _default_slot_size(shape, dtype)
        self._cursor = 0

        self._slots: list[Buffer] = []
        for i in range(num_slots):
            bid = buf_ids[i] if buf_ids is not None else None
            self._slots.append(
                Buffer(
                    shape=shape,
                    dtype=dtype,
                    memory=memory,
                    addr=self._base_addr + i * self._slot_size,
                    buf_id=bid,
                    size=self._slot_size,
                    blayout=blayout,
                    slayout=slayout,
                    fractal=fractal,
                    pad=pad,
                    compact=compact,
                    valid_shape=valid_shape,
                )
            )

    # --- introspection ------------------------------------------------------

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def buf_ids(self) -> tuple[int, ...] | None:
        return self._buf_ids

    @property
    def has_buf_ids(self) -> bool:
        return self._buf_ids is not None

    @property
    def base_addr(self) -> int:
        return self._base_addr

    @property
    def slot_size(self) -> int:
        return self._slot_size

    @property
    def slots(self) -> tuple[Buffer, ...]:
        return tuple(self._slots)

    @property
    def tiles(self) -> tuple:
        """Tuple of the underlying tile expressions, one per slot.

        Suitable for use with the existing ``tile_buf[buf_idx]`` dynamic-index
        pattern — the AST parser recognizes tile-valued tuples and expands
        subscript access into an if-else chain.
        """
        return tuple(slot.tile for slot in self._slots)

    # --- cursor management (static, Python-level) --------------------------

    def current(self) -> BufferSlot:
        """Return the slot at the current Python-level cursor.

        Since the cursor is Python-level, this pattern works best when the
        loop is Python-unrolled or when paired with :meth:`advance` inside a
        statically sized construct. For dynamic loops, use :pyattr:`tiles`
        with a subscript access instead.
        """
        slot = self._slots[self._cursor % self.num_slots]
        return BufferSlot(tile=slot.tile, buf_id=slot.buf_id)

    def previous(self) -> BufferSlot:
        """Return the slot at the previous cursor position (cursor - 1).

        Typical double-buffer pipeline pattern::

            q_l1_db.advance()
            cur  = q_l1_db.current()    # new slot → load new data
            prev = q_l1_db.previous()   # old slot → compute with old data
        """
        idx = (self._cursor - 1) % self.num_slots
        slot = self._slots[idx]
        return BufferSlot(tile=slot.tile, buf_id=slot.buf_id)

    def advance(self) -> None:
        """Advance the Python-level cursor by one slot (wraps modulo num_slots)."""
        self._cursor = (self._cursor + 1) % self.num_slots

    def reset(self) -> None:
        """Reset the Python-level cursor to slot 0."""
        self._cursor = 0

    # --- convenience --------------------------------------------------------

    def __getitem__(self, idx: int):
        """NBuffer[i] returns the i-th slot's tile (static int index only).

        For dynamic (IR Expr) indexing, use ``nbuf.tiles[buf_idx]`` directly.
        """
        if not isinstance(idx, int):
            raise TypeError(
                f"NBuffer indexing requires a static int, got {type(idx).__name__}. "
                "Use nbuf.tiles[buf_idx] for dynamic indexing."
            )
        return self._slots[idx].tile

    def __len__(self) -> int:
        return self.num_slots

    def set_validshape(self, row, col) -> None:
        """Apply ``plm.set_validshape`` to every slot."""
        for slot in self._slots:
            slot.set_validshape(row, col)


# ---------------------------------------------------------------------------
# Memory-specific NBuffer subclasses (canndsl-style convenience)
# ---------------------------------------------------------------------------
# Each subclass pre-fills ``memory`` and default ``blayout/slayout`` per the
# hardware requirements documented in .claude/hardware.md. Signature:
#
#     L1NBuffer(base_addr, dtype, shape=(...), buf_ids=(...), **overrides)
#
# This matches canndsl's calling convention:
#
#     q_l1_db = L1NBuffer(0, Float32, shape=(32, 32), buf_ids=(0, 1))
# ---------------------------------------------------------------------------


class UBNBuffer(NBuffer):
    """Multi-slot buffer in UB (Vec) memory."""

    def __init__(self, base_addr, dtype, *, shape, buf_ids=None, num_slots=None, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Vec,
                         base_addr=base_addr, buf_ids=buf_ids, num_slots=num_slots, **kw)


class L1NBuffer(NBuffer):
    """Multi-slot buffer in L1 (Mat) memory. Default layout: NZ (blayout=2, slayout=1)."""

    def __init__(self, base_addr, dtype, *, shape, buf_ids=None, num_slots=None, blayout=2, slayout=1, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Mat,
                         base_addr=base_addr, buf_ids=buf_ids, num_slots=num_slots,
                         blayout=blayout, slayout=slayout, **kw)


class L0ANBuffer(NBuffer):
    """Multi-slot buffer in L0A (Left) memory. Default layout: NZ (blayout=2, slayout=1)."""

    def __init__(self, base_addr, dtype, *, shape, buf_ids=None, num_slots=None, blayout=2, slayout=1, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Left,
                         base_addr=base_addr, buf_ids=buf_ids, num_slots=num_slots,
                         blayout=blayout, slayout=slayout, **kw)


class L0BNBuffer(NBuffer):
    """Multi-slot buffer in L0B (Right) memory. Default layout: (blayout=1, slayout=2)."""

    def __init__(self, base_addr, dtype, *, shape, buf_ids=None, num_slots=None, blayout=1, slayout=2, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Right,
                         base_addr=base_addr, buf_ids=buf_ids, num_slots=num_slots,
                         blayout=blayout, slayout=slayout, **kw)


class L0CNBuffer(NBuffer):
    """Multi-slot buffer in L0C (Acc) memory. Default layout: NZ (blayout=2, slayout=1)."""

    def __init__(self, base_addr, dtype, *, shape, buf_ids=None, num_slots=None, blayout=2, slayout=1, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Acc,
                         base_addr=base_addr, buf_ids=buf_ids, num_slots=num_slots,
                         blayout=blayout, slayout=slayout, **kw)


# --- Single-slot memory-specific subclasses ---


class UBBuffer(Buffer):
    """Single buffer in UB (Vec) memory."""

    def __init__(self, base_addr, dtype, *, shape, buf_id=None, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Vec,
                         addr=base_addr, buf_id=buf_id, **kw)


class L1Buffer(Buffer):
    """Single buffer in L1 (Mat) memory."""

    def __init__(self, base_addr, dtype, *, shape, buf_id=None, blayout=2, slayout=1, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Mat,
                         addr=base_addr, buf_id=buf_id,
                         blayout=blayout, slayout=slayout, **kw)


class L0ABuffer(Buffer):
    """Single buffer in L0A (Left) memory. Default layout: NZ (blayout=2, slayout=1)."""

    def __init__(self, base_addr, dtype, *, shape, buf_id=None, blayout=2, slayout=1, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Left,
                         addr=base_addr, buf_id=buf_id,
                         blayout=blayout, slayout=slayout, **kw)


class L0BBuffer(Buffer):
    """Single buffer in L0B (Right) memory."""

    def __init__(self, base_addr, dtype, *, shape, buf_id=None, blayout=1, slayout=2, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Right,
                         addr=base_addr, buf_id=buf_id,
                         blayout=blayout, slayout=slayout, **kw)


class L0CBuffer(Buffer):
    """Single buffer in L0C (Acc) memory."""

    def __init__(self, base_addr, dtype, *, shape, buf_id=None, blayout=2, slayout=1, **kw):
        super().__init__(shape=shape, dtype=dtype, memory=MemorySpace.Acc,
                         addr=base_addr, buf_id=buf_id,
                         blayout=blayout, slayout=slayout, **kw)
