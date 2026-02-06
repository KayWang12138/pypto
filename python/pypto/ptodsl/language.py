#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""

import contextvars
from mlir.ir import F32Type, IndexType
from mlir.dialects import arith, pto

_builder = contextvars.ContextVar("_builder", default=None)


def _get():
    b = _builder.get()
    if b is None:
        raise RuntimeError("Not inside ir_builder() context")
    return b


# --- Type helpers (string -> attr) ---
def _blayout(s):
    return pto.BLayoutAttr.get(getattr(pto.BLayout, s))


def _slayout(s):
    return pto.SLayoutAttr.get(getattr(pto.SLayout, s))


def _pad(s):
    return pto.PadValueAttr.get(getattr(pto.PadValue, s))


def _addr_space(s):
    return pto.AddressSpaceAttr.get(getattr(pto.AddressSpace, s))


# --- Lazy attrs: pto.float32, pto.PIPE_MTE2, etc. resolved via __getattr__ (module attrs don't use __get__) ---
def __getattr__(name):
    if name == "float32":
        return F32Type.get()
    if name in ("PIPE_MTE2", "PIPE_V", "PIPE_MTE3", "EVENT_ID0"):
        return _get().attrs[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Types ---
def PtrType(dtype):
    return pto.PtrType.get(dtype)


def TensorType(rank=2, dtype=None):
    return pto.TensorViewType.get(rank=rank, element_type=dtype)


def SubTensorType(shape, dtype):
    return pto.TileViewType.get(shape=shape, element_type=dtype)


def TileBufConfig(blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"):
    bl = _blayout(blayout)
    sl = _slayout(slayout)
    pd = _pad(pad)
    return pto.TileBufConfigAttr.get(
        blayout=bl, slayout=sl, s_fractal_size=s_fractal_size, pad=pd
    )


def TileBufType(shape, dtype, memory_space="UB", config=None, valid_shape=None):
    if valid_shape is None:
        valid_shape = shape
    ub = _addr_space(memory_space)
    return pto.TileBufType.get(
        shape=shape,
        element_type=dtype,
        memory_space=ub,
        valid_shape=valid_shape,
        config=config,
    )


# --- Index constant ---
def const(value):
    return arith.ConstantOp(IndexType.get(), value).result


# --- Ops ---
def as_tensor(tensor_type, ptr, shape, strides):
    return pto.MakeTensorViewOp(tensor_type, ptr=ptr, shape=shape, strides=strides).result


def slice_view(subtensor_type, source, offsets, sizes):
    # return pto.PartitionViewOp(
    #     subtensor_type, source=source, offsets=offsets, sizes=sizes
    # ).result
    return pto.SubViewOp(subtensor_type, source=source, offsets=offsets, sizes=sizes).result


def alloc_tile(tile_type):
    return pto.AllocTileOp(tile_type).result


def load(source_view, tile_buf):
    pto.TLoadOp(None, source_view, tile_buf)


def _pipe_attr(name):
    """Map string pipe name to MLIR pipe attribute. e.g. 'MTE2' -> PIPE_MTE2."""
    key = f"PIPE_{name}" if not name.startswith("PIPE_") else name
    return _get().attrs[key]


def _event_attr(event_id):
    """Map event_id (int 0..7) to MLIR event attribute. e.g. 0 -> EVENT_ID0."""
    if not 0 <= event_id <= 7:
        raise ValueError(f"event_id must be 0..7, got {event_id}")
    return _get().attrs[f"EVENT_ID{event_id}"]


def set_flag(src_pipe: str, dst_pipe: str, *, event_id: int):
    pto.SetFlagOp(_pipe_attr(src_pipe), _pipe_attr(dst_pipe), _event_attr(event_id))


def wait_flag(src_pipe: str, dst_pipe: str, *, event_id: int):
    pto.WaitFlagOp(_pipe_attr(src_pipe), _pipe_attr(dst_pipe), _event_attr(event_id))


def relu(input_tile, output_tile):
    pto.TReluOp(input_tile, output_tile)


def store(tile_buf, dest_view):
    pto.TStoreOp(None, tile_buf, dest_view)


# Expose _builder for __init__.py (set/reset from ir_builder)
class _BuilderVar:
    def set(self, ctx):
        return _builder.set(ctx)

    def get(self):
        return _builder.get()

    def reset(self, token):
        _builder.reset(token)


_builder_var = _BuilderVar()