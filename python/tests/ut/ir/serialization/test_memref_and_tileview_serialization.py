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
"""Tests for serialization of MemRef and TileView.

This file tests the serialization and deserialization of:
- MemRef (memory references) - memory allocation information
- TileView (tile view information) - tiling and layout information

These are critical components for tensor memory management and optimization.

Source: Merged from test_memref_serialization.py and test_tileview_serialization.py
"""

import pytest
from pypto import ir
from pypto.ir import DataType, MemorySpace


class TestMemRefSerialization:
    """Test serialization of MemRef nodes."""

    def test_memref_basic_serialization(self):
        """Test basic MemRef serialization with DDR memory space."""
        span = ir.Span.unknown()

        # Create address expression
        addr = ir.ConstInt(0, DataType.INT64, span)

        # Create MemRef with DDR memory space
        memref = ir.MemRef(
            memory_space=MemorySpace.DDR,
            addr=addr,
            size=1024,
            id=1,
            span=span
        )

        # Serialize
        data = ir.serialize(memref)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored, ir.MemRef)
        assert restored.memory_space_ == MemorySpace.DDR
        assert isinstance(restored.addr_, ir.ConstInt)
        assert restored.addr_.value == 0
        assert restored.size_ == 1024
        assert restored.id_ == 1
        ir.assert_structural_equal(memref, restored, enable_auto_mapping=True)

    def test_memref_all_memory_spaces(self):
        """Test MemRef serialization with all MemorySpace enum values."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)

        # Test all memory space types
        memory_spaces = [
            MemorySpace.DDR,
            MemorySpace.UB,
            MemorySpace.L1,
            MemorySpace.L0A,
            MemorySpace.L0B,
            MemorySpace.L0C,
        ]

        for idx, mem_space in enumerate(memory_spaces):
            memref = ir.MemRef(
                memory_space=mem_space,
                addr=addr,
                size=2048,
                id=idx + 1,
                span=span
            )

            # Serialize and deserialize
            data = ir.serialize(memref)
            restored = ir.deserialize(data)

            # Verify memory space is preserved
            assert restored.memory_space_ == mem_space, \
                f"MemorySpace {mem_space} should be preserved"
            assert restored.size_ == 2048
            assert restored.id_ == idx + 1
            ir.assert_structural_equal(memref, restored, enable_auto_mapping=True)

    def test_memref_with_expression_addr(self):
        """Test MemRef with complex address expression."""
        span = ir.Span.unknown()

        # Create complex address: base + offset
        base = ir.ConstInt(1000, DataType.INT64, span)
        offset = ir.ConstInt(256, DataType.INT64, span)
        addr = ir.Add(base, offset, DataType.INT64, span)

        # Create MemRef
        memref = ir.MemRef(
            memory_space=MemorySpace.UB,
            addr=addr,
            size=512,
            id=42,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(memref)
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored, ir.MemRef)
        assert restored.memory_space_ == MemorySpace.UB
        assert isinstance(restored.addr_, ir.Add)
        assert isinstance(restored.addr_.left, ir.ConstInt)
        assert restored.addr_.left.value == 1000
        assert isinstance(restored.addr_.right, ir.ConstInt)
        assert restored.addr_.right.value == 256
        assert restored.size_ == 512
        assert restored.id_ == 42
        ir.assert_structural_equal(memref, restored, enable_auto_mapping=True)

    def test_memref_large_size(self):
        """Test MemRef with large size value (uint64_t)."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)

        # Use a large size value (close to uint64_t max)
        large_size = 2**40  # 1TB

        memref = ir.MemRef(
            memory_space=MemorySpace.L1,
            addr=addr,
            size=large_size,
            id=999,
            span=span
        )

        # Serialize and deserialize
        data = ir.serialize(memref)
        restored = ir.deserialize(data)

        # Verify large size is preserved
        assert restored.size_ == large_size
        assert restored.id_ == 999
        ir.assert_structural_equal(memref, restored, enable_auto_mapping=True)


class TestTensorTypeWithMemRef:
    """Test serialization of TensorType with MemRef."""

    def test_tensor_type_with_memref(self):
        """Test TensorType with MemRef serialization."""
        span = ir.Span.unknown()

        # Create shape
        dim = ir.ConstInt(100, DataType.INT64, span)

        # Create MemRef
        addr = ir.ConstInt(0, DataType.INT64, span)
        memref = ir.MemRef(
            memory_space=MemorySpace.UB,
            addr=addr,
            size=400,  # 100 * 4 bytes (FP32)
            id=1,
            span=span
        )

        # Create TensorType with MemRef
        tensor_type = ir.TensorType([dim], DataType.FP32, memref)
        var = ir.Var("tensor_with_memref", tensor_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored, ir.Var)
        assert restored.name == "tensor_with_memref"
        assert isinstance(restored.type, ir.TensorType)
        assert restored.type.dtype == DataType.FP32

        # Verify MemRef is preserved
        assert restored.type.memref is not None
        assert isinstance(restored.type.memref, ir.MemRef)
        assert restored.type.memref.memory_space_ == MemorySpace.UB
        assert restored.type.memref.size_ == 400
        assert restored.type.memref.id_ == 1

        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)

    def test_tensor_type_without_memref(self):
        """Test TensorType without MemRef (memref is optional)."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(100, DataType.INT64, span)

        # Create TensorType without MemRef
        tensor_type = ir.TensorType([dim], DataType.FP32)
        var = ir.Var("tensor_no_memref", tensor_type, span)

        # Serialize and deserialize
        data = ir.serialize(var)
        restored = ir.deserialize(data)

        # Verify memref is None
        assert isinstance(restored.type, ir.TensorType)
        assert restored.type.memref is None
        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)

    def test_tensor_type_with_memref_different_spaces(self):
        """Test TensorType with MemRef in different memory spaces."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(256, DataType.INT64, span)
        addr = ir.ConstInt(0, DataType.INT64, span)

        memory_spaces = [MemorySpace.DDR, MemorySpace.UB, MemorySpace.L0A]

        for idx, mem_space in enumerate(memory_spaces):
            memref = ir.MemRef(
                memory_space=mem_space,
                addr=addr,
                size=1024,
                id=idx + 10,
                span=span
            )

            tensor_type = ir.TensorType([dim], DataType.INT32, memref)
            var = ir.Var(f"tensor_{mem_space.name}", tensor_type, span)

            # Serialize and deserialize
            data = ir.serialize(var)
            restored = ir.deserialize(data)

            # Verify
            assert restored.type.memref.memory_space_ == mem_space
            ir.assert_structural_equal(var, restored, enable_auto_mapping=True)


class TestTileTypeWithMemRef:
    """Test serialization of TileType with MemRef (without TileView)."""

    def test_tile_type_with_memref(self):
        """Test TileType with MemRef serialization."""
        span = ir.Span.unknown()

        # Create shape
        dim = ir.ConstInt(16, DataType.INT64, span)

        # Create MemRef
        addr = ir.ConstInt(0, DataType.INT64, span)
        memref = ir.MemRef(
            memory_space=MemorySpace.L0A,
            addr=addr,
            size=512,
            id=5,
            span=span
        )

        # Create TileType with MemRef
        tile_type = ir.TileType([dim], DataType.FP16, memref=memref)
        var = ir.Var("tile_with_memref", tile_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TileType)
        assert restored.type.dtype == DataType.FP16

        # Verify MemRef
        assert restored.type.memref is not None
        assert isinstance(restored.type.memref, ir.MemRef)
        assert restored.type.memref.memory_space_ == MemorySpace.L0A
        assert restored.type.memref.size_ == 512
        assert restored.type.memref.id_ == 5

        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)

    def test_tile_type_without_memref(self):
        """Test TileType without MemRef."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(16, DataType.INT64, span)

        # Create TileType without MemRef
        tile_type = ir.TileType([dim], DataType.FP16)
        var = ir.Var("tile_no_memref", tile_type, span)

        # Serialize and deserialize
        data = ir.serialize(var)
        restored = ir.deserialize(data)

        # Verify memref is None
        assert isinstance(restored.type, ir.TileType)
        assert restored.type.memref is None
        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)


class TestTileViewSerialization:
    """Test serialization of TileView - tiling layout information.

    Source: Migrated from test_tileview_serialization.py
    """

    def test_tile_type_with_tileview(self):
        """Test TileType with TileView serialization - basic 1D case."""
        span = ir.Span.unknown()

        # Create shape
        dim = ir.ConstInt(32, DataType.INT64, span)

        # Create MemRef (required for TileView)
        addr = ir.ConstInt(0, DataType.INT64, span)
        memref = ir.MemRef(
            memory_space=MemorySpace.L0A,
            addr=addr,
            size=512,
            id=5,
            span=span
        )

        # Create TileView
        valid_shape_dim = ir.ConstInt(16, DataType.INT64, span)
        stride_dim = ir.ConstInt(1, DataType.INT64, span)
        start_offset = ir.ConstInt(0, DataType.INT64, span)

        tile_view = ir.TileView(
            valid_shape=[valid_shape_dim],
            stride=[stride_dim],
            start_offset=start_offset
        )

        # Create TileType with MemRef and TileView
        tile_type = ir.TileType([dim], DataType.FP32, memref=memref, tile_view=tile_view)
        var = ir.Var("tile_with_view", tile_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)

        # Verify
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TileType)

        # Verify MemRef
        assert restored.type.memref is not None
        assert isinstance(restored.type.memref, ir.MemRef)
        assert restored.type.memref.memory_space_ == MemorySpace.L0A
        assert restored.type.memref.size_ == 512
        assert restored.type.memref.id_ == 5

        # Verify TileView
        assert restored.type.tile_view is not None
        assert len(restored.type.tile_view.valid_shape) == 1
        assert isinstance(restored.type.tile_view.valid_shape[0], ir.ConstInt)
        assert restored.type.tile_view.valid_shape[0].value == 16

        assert len(restored.type.tile_view.stride) == 1
        assert isinstance(restored.type.tile_view.stride[0], ir.ConstInt)
        assert restored.type.tile_view.stride[0].value == 1

        assert isinstance(restored.type.tile_view.start_offset, ir.ConstInt)
        assert restored.type.tile_view.start_offset.value == 0

        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)

    def test_tile_type_with_multi_dim_tileview(self):
        """Test TileType with multi-dimensional TileView - 2D case."""
        span = ir.Span.unknown()

        # Create 2D shape
        dim1 = ir.ConstInt(32, DataType.INT64, span)
        dim2 = ir.ConstInt(32, DataType.INT64, span)

        # Create MemRef (required for TileView)
        addr = ir.ConstInt(0, DataType.INT64, span)
        memref = ir.MemRef(
            memory_space=MemorySpace.L0A,
            addr=addr,
            size=2048,
            id=10,
            span=span
        )

        # Create 2D TileView
        valid_shape = [
            ir.ConstInt(16, DataType.INT64, span),
            ir.ConstInt(16, DataType.INT64, span)
        ]
        stride = [
            ir.ConstInt(32, DataType.INT64, span),  # Row stride
            ir.ConstInt(1, DataType.INT64, span)    # Column stride
        ]
        start_offset = ir.ConstInt(128, DataType.INT64, span)

        tile_view = ir.TileView(
            valid_shape=valid_shape,
            stride=stride,
            start_offset=start_offset
        )

        # Create TileType with MemRef and TileView
        tile_type = ir.TileType([dim1, dim2], DataType.FP16, memref=memref, tile_view=tile_view)
        var = ir.Var("tile_2d_view", tile_type, span)

        # Serialize and deserialize
        data = ir.serialize(var)
        restored = ir.deserialize(data)

        # Verify MemRef
        assert restored.type.memref is not None
        assert restored.type.memref.memory_space_ == MemorySpace.L0A
        assert restored.type.memref.size_ == 2048
        assert restored.type.memref.id_ == 10

        # Verify TileView dimensions
        assert len(restored.type.tile_view.valid_shape) == 2
        assert restored.type.tile_view.valid_shape[0].value == 16
        assert restored.type.tile_view.valid_shape[1].value == 16

        assert len(restored.type.tile_view.stride) == 2
        assert restored.type.tile_view.stride[0].value == 32
        assert restored.type.tile_view.stride[1].value == 1

        assert restored.type.tile_view.start_offset.value == 128

        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)

    def test_tile_type_without_tileview(self):
        """Test TileType without TileView (tile_view is optional)."""
        span = ir.Span.unknown()
        dim = ir.ConstInt(32, DataType.INT64, span)

        # Create TileType without TileView
        tile_type = ir.TileType([dim], DataType.FP32)
        var = ir.Var("tile_no_view", tile_type, span)

        # Serialize and deserialize
        data = ir.serialize(var)
        restored = ir.deserialize(data)

        # Verify tile_view is None
        assert isinstance(restored.type, ir.TileType)
        assert restored.type.tile_view is None
        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)


class TestTileTypeWithMemRefAndTileView:
    """Test serialization of TileType with both MemRef and TileView - complete tile information.

    Source: Merged from test_memref_serialization.py TestMemRefAndTileViewCombined
    """

    def test_tile_type_with_memref_and_tileview(self):
        """Test TileType with both MemRef and TileView - complete configuration."""
        span = ir.Span.unknown()

        # Create shape
        dim = ir.ConstInt(64, DataType.INT64, span)

        # Create MemRef
        addr = ir.ConstInt(0, DataType.INT64, span)
        memref = ir.MemRef(
            memory_space=MemorySpace.L0B,
            addr=addr,
            size=2048,
            id=100,
            span=span
        )

        # Create TileView
        valid_shape_dim = ir.ConstInt(32, DataType.INT64, span)
        stride_dim = ir.ConstInt(1, DataType.INT64, span)
        start_offset = ir.ConstInt(0, DataType.INT64, span)

        tile_view = ir.TileView(
            valid_shape=[valid_shape_dim],
            stride=[stride_dim],
            start_offset=start_offset
        )

        # Create TileType with both MemRef and TileView
        tile_type = ir.TileType([dim], DataType.FP32, memref=memref, tile_view=tile_view)
        var = ir.Var("tile_full", tile_type, span)

        # Serialize
        data = ir.serialize(var)
        assert data is not None

        # Deserialize
        restored = ir.deserialize(data)

        # Verify TileType
        assert isinstance(restored, ir.Var)
        assert isinstance(restored.type, ir.TileType)
        assert restored.type.dtype == DataType.FP32

        # Verify MemRef
        assert restored.type.memref is not None
        assert restored.type.memref.memory_space_ == MemorySpace.L0B
        assert restored.type.memref.size_ == 2048
        assert restored.type.memref.id_ == 100

        # Verify TileView
        assert restored.type.tile_view is not None
        assert len(restored.type.tile_view.valid_shape) == 1
        assert restored.type.tile_view.valid_shape[0].value == 32
        assert restored.type.tile_view.stride[0].value == 1
        assert restored.type.tile_view.start_offset.value == 0

        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)

    def test_complex_tile_with_expression_fields(self):
        """Test TileType with complex expressions in MemRef and TileView fields."""
        span = ir.Span.unknown()

        # Create shape
        dim = ir.ConstInt(128, DataType.INT64, span)

        # Create MemRef with expression addr
        base_addr = ir.ConstInt(4096, DataType.INT64, span)
        offset = ir.ConstInt(256, DataType.INT64, span)
        addr = ir.Add(base_addr, offset, DataType.INT64, span)

        memref = ir.MemRef(
            memory_space=MemorySpace.L0C,
            addr=addr,
            size=8192,
            id=200,
            span=span
        )

        # Create TileView with expression fields
        base_dim = ir.ConstInt(32, DataType.INT64, span)
        extend = ir.ConstInt(32, DataType.INT64, span)
        valid_shape_expr = ir.Add(base_dim, extend, DataType.INT64, span)

        tile_view = ir.TileView(
            valid_shape=[valid_shape_expr],
            stride=[ir.ConstInt(2, DataType.INT64, span)],
            start_offset=ir.ConstInt(16, DataType.INT64, span)
        )

        # Create TileType
        tile_type = ir.TileType([dim], DataType.FP16, memref=memref, tile_view=tile_view)
        var = ir.Var("complex_tile", tile_type, span)

        # Serialize and deserialize
        data = ir.serialize(var)
        restored = ir.deserialize(data)

        # Verify MemRef with expression addr
        assert isinstance(restored.type.memref.addr_, ir.Add)
        assert restored.type.memref.addr_.left.value == 4096
        assert restored.type.memref.addr_.right.value == 256

        # Verify TileView with expression valid_shape
        assert isinstance(restored.type.tile_view.valid_shape[0], ir.Add)
        assert restored.type.tile_view.valid_shape[0].left.value == 32
        assert restored.type.tile_view.valid_shape[0].right.value == 32

        ir.assert_structural_equal(var, restored, enable_auto_mapping=True)


class TestMemRefEdgeCases:
    """Test edge cases and boundary conditions for MemRef."""

    def test_memref_zero_size(self):
        """Test MemRef with zero size - minimal allocation."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)

        memref = ir.MemRef(
            memory_space=MemorySpace.DDR,
            addr=addr,
            size=0,
            id=1,
            span=span
        )

        data = ir.serialize(memref)
        restored = ir.deserialize(data)

        assert restored.size_ == 0
        ir.assert_structural_equal(memref, restored, enable_auto_mapping=True)

    def test_memref_max_id(self):
        """Test MemRef with maximum uint64_t id value - boundary test."""
        span = ir.Span.unknown()
        addr = ir.ConstInt(0, DataType.INT64, span)

        # Maximum uint64_t value
        max_id = 2**64 - 1

        memref = ir.MemRef(
            memory_space=MemorySpace.UB,
            addr=addr,
            size=1024,
            id=max_id,
            span=span
        )

        data = ir.serialize(memref)
        restored = ir.deserialize(data)

        assert restored.id_ == max_id
        ir.assert_structural_equal(memref, restored, enable_auto_mapping=True)
