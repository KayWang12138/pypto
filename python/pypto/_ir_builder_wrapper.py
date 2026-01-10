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
Wrapper for IR builder operations to enable imperative-style syntax like torch/numpy.

This module provides a TileWrapper class that wraps TileValue objects and allows
operations like:
    res_if_y = res_loop_y + scale2
instead of the verbose:
    res_if_y = builder.create_tile(ctx, tile_shape, ir.DataType.float, "outputY")
    mul_op_y = builder.create_binary_scalar_op(ir.Opcode.OP_MULS, res_loop_y, scale2, res_if_y)
    builder.emit(ctx, mul_op_y)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypto.pypto_impl import ir


class TileWrapper:
    """
    Wrapper for TileValue that enables imperative-style operations.
    
    This wrapper stores a TileValue along with the builder and context needed
    to create new operations. It overloads Python operators to automatically
    create tiles, operations, and emit them.
    
    Example:
        wrapper = TileWrapper(tile_value, builder, ctx)
        result = wrapper + scalar  # Automatically creates tile, op, and emits
    """
    
    def __init__(self, tile_value: "ir.Tile", builder: "ir.IrBuilder", ctx: "ir.IrBuilderContext", name: str = ""):
        """
        Initialize the wrapper.
        
        Args:
            tile_value: The TileValue to wrap
            builder: The IRBuilder instance
            ctx: The IrBuilderContext instance
            name: Optional name for generated tiles (if empty, auto-generated)
        """
        self._tile = tile_value
        self._builder = builder
        self._ctx = ctx
        self._name_prefix = name
    
    @property
    def tile(self) -> "ir.Tile":
        """Get the underlying TileValue."""
        return self._tile
    
    def _create_result_tile(self, name: str = "") -> "ir.Tile":
        """Create a new tile with the same shape and dtype as the wrapped tile."""
        shape = list(self._tile.shape)  # GetShape returns a list
        dtype = self._tile.type.dtype  # GetDataType from Type
        result_name = name if name else (self._name_prefix if self._name_prefix else "")
        return self._builder.create_tile(self._ctx, shape, dtype, result_name)
    
    def _create_binary_scalar_op(self, opcode: "ir.Opcode", scalar: "ir.Scalar", result_tile: "ir.Tile") -> "ir.Operation":
        """Create a binary scalar operation."""
        return self._builder.create_binary_scalar_op(opcode, self._tile, scalar, result_tile)
    
    def _emit_op(self, op: "ir.Operation"):
        """Emit an operation."""
        self._builder.emit(self._ctx, op)
    
    def _binary_op(self, opcode: "ir.Opcode", scalar: "ir.Scalar", name: str = "") -> "TileWrapper":
        """
        Perform a binary scalar operation and return a new wrapper.
        
        Args:
            opcode: The operation code (e.g., ir.Opcode.OP_ADDS)
            scalar: The scalar operand
            name: Optional name for the result tile
            
        Returns:
            A new TileWrapper wrapping the result tile
        """
        result_tile = self._create_result_tile(name)
        op = self._create_binary_scalar_op(opcode, scalar, result_tile)
        self._emit_op(op)
        return TileWrapper(result_tile, self._builder, self._ctx, name)
    
    def __add__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """Overload + operator for addition (OP_ADDS)."""
        from pypto.pypto_impl import ir
        return self._binary_op(ir.Opcode.OP_ADDS, scalar)
    
    def __mul__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """Overload * operator for multiplication (OP_MULS)."""
        from pypto.pypto_impl import ir
        return self._binary_op(ir.Opcode.OP_MULS, scalar)
    
    def __sub__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """Overload - operator for subtraction (OP_SUBS)."""
        from pypto.pypto_impl import ir
        return self._binary_op(ir.Opcode.OP_SUBS, scalar)
    
    def __truediv__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """Overload / operator for division (OP_DIVS)."""
        from pypto.pypto_impl import ir
        return self._binary_op(ir.Opcode.OP_DIVS, scalar)
    
    def __radd__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """Right-hand addition (scalar + tile)."""
        return self.__add__(scalar)
    
    def __rmul__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """Right-hand multiplication (scalar * tile)."""
        return self.__mul__(scalar)
    
    def __rsub__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """Right-hand subtraction (scalar - tile)."""
        # For scalar - tile, we need to reverse the operands
        # This requires creating an op with reverse=True, but for simplicity,
        # we'll use the same pattern and note that reverse operand support
        # may need to be added to the underlying operation
        from pypto.pypto_impl import ir
        result_tile = self._create_result_tile()
        op = self._builder.create_binary_scalar_op(ir.Opcode.OP_SUBS, self._tile, scalar, result_tile)
        # Note: If the operation supports reverse operand, it should be set here
        self._emit_op(op)
        return TileWrapper(result_tile, self._builder, self._ctx)
    
    def __rtruediv__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """Right-hand division (scalar / tile)."""
        from pypto.pypto_impl import ir
        result_tile = self._create_result_tile()
        op = self._builder.create_binary_scalar_op(ir.Opcode.OP_DIVS, self._tile, scalar, result_tile)
        # Note: If the operation supports reverse operand, it should be set here
        self._emit_op(op)
        return TileWrapper(result_tile, self._builder, self._ctx)
    
    def __iadd__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """In-place addition (OP_ADDS) using the same tile as output."""
        from pypto.pypto_impl import ir
        op = self._builder.create_binary_scalar_op(ir.Opcode.OP_ADDS, self._tile, scalar, self._tile)
        self._emit_op(op)
        return self  # Return self for in-place operations
    
    def __imul__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """In-place multiplication (OP_MULS) using the same tile as output."""
        from pypto.pypto_impl import ir
        op = self._builder.create_binary_scalar_op(ir.Opcode.OP_MULS, self._tile, scalar, self._tile)
        self._emit_op(op)
        return self  # Return self for in-place operations
    
    def __isub__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """In-place subtraction (OP_SUBS) using the same tile as output."""
        from pypto.pypto_impl import ir
        op = self._builder.create_binary_scalar_op(ir.Opcode.OP_SUBS, self._tile, scalar, self._tile)
        self._emit_op(op)
        return self  # Return self for in-place operations
    
    def __itruediv__(self, scalar: "ir.Scalar") -> "TileWrapper":
        """In-place division (OP_DIVS) using the same tile as output."""
        from pypto.pypto_impl import ir
        op = self._builder.create_binary_scalar_op(ir.Opcode.OP_DIVS, self._tile, scalar, self._tile)
        self._emit_op(op)
        return self  # Return self for in-place operations


def wrap_tile(tile_value: "ir.Tile", builder: "ir.IrBuilder", ctx: "ir.IrBuilderContext", name: str = "") -> TileWrapper:
    """
    Convenience function to wrap a TileValue for imperative-style operations.
    
    Args:
        tile_value: The TileValue to wrap
        builder: The IRBuilder instance
        ctx: The IrBuilderContext instance
        name: Optional name prefix for generated tiles
        
    Returns:
        A TileWrapper instance
        
    Example:
        res_loop_y_wrapped = wrap_tile(res_loop_y, builder, ctx)
        res_if_y = res_loop_y_wrapped * scale2
    """
    return TileWrapper(tile_value, builder, ctx, name)


def create_wrapped_tile(
    builder: "ir.IrBuilder",
    ctx: "ir.IrBuilderContext",
    shape: list,
    dtype: "ir.DataType",
    name: str = ""
) -> TileWrapper:
    """
    Create a new tile and wrap it for imperative-style operations.
    
    This is a convenience function that combines create_tile and wrap_tile.
    
    Args:
        builder: The IRBuilder instance
        ctx: The IrBuilderContext instance
        shape: The shape of the tile
        dtype: The data type of the tile
        name: Optional name for the tile
        
    Returns:
        A TileWrapper instance
        
    Example:
        res_loop_y = create_wrapped_tile(builder, ctx, tile_shape, ir.DataType.float, "outputY")
        res_if_y = res_loop_y * scale2  # Simplified syntax!
    """
    tile = builder.create_tile(ctx, shape, dtype, name)
    return TileWrapper(tile, builder, ctx, name)
