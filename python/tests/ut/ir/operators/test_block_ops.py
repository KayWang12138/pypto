# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for block operations."""

import pytest
from pypto import ir
from pypto.ir import DataType
from pypto.ir.op import block

class TestBlockElementwiseOps:
    """Test suite for block-level element-wise operators (tile-tile and tile-scalar)."""

    def _make_tile_var(self, name, rows, cols, dtype=DataType.FP32):
        """Helper to create a tile variable with given shape and dtype."""
        span = ir.Span.unknown()
        dim_r = ir.ConstInt(rows, DataType.INT32, span)
        dim_c = ir.ConstInt(cols, DataType.INT32, span)
        tile_type = ir.TileType([dim_r, dim_c], dtype)
        return ir.Var(name, tile_type, span)

    def test_block_add(self):
        """Test block.add operator - element-wise addition of two tiles."""
        tile_a = self._make_tile_var("tile_a", 32, 32)
        tile_b = self._make_tile_var("tile_b", 32, 32)

        call = block.add(tile_a, tile_b)
        print("\n")
        print(ir.python_print(call))        # pl.op.block.add(tile_a, tile_b)
        print(ir.python_print(call.type))   # pl.Tile[[32, 32], pl.FP32]
        assert isinstance(call, ir.Call)
        assert call.op.name == "block.add"
        result_type = call.type
        assert isinstance(result_type, ir.TileType)
        assert result_type.dtype == DataType.FP32

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
