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
    """Test suite for block-level element-wise operators."""

    def test_block_add(self):
        """Test block.add operator - element-wise addition of two tiles."""
        ib = ir.IRBuilder()

        with ib.program("test_block_add_program") as p:
            p.declare_function("test_add")

            with ib.function("test_add", type=ir.FunctionType.InCore) as f:
                # Define input and output tensor parameters
                input_a = f.param("input_a", ir.TensorType([128, 128], DataType.FP32))
                input_b = f.param("input_b", ir.TensorType([128, 128], DataType.FP32))
                output = f.param("output", ir.TensorType([128, 128], DataType.FP32))
                f.return_type(ir.TensorType([128, 128], DataType.FP32))

                # Define tile size and offsets
                tile_height = 32
                tile_width = 32
                row_offset = 0
                col_offset = 0

                # Load tiles from tensors
                tile_a = ib.let("tile_a", block.load(input_a, row_offset, col_offset, tile_height, tile_width))
                tile_b = ib.let("tile_b", block.load(input_b, row_offset, col_offset, tile_height, tile_width))

                # Perform element-wise add
                tile_sum = ib.let("tile_sum", block.add(tile_a, tile_b))

                # Store result back to tensor
                result = ib.let(
                    "result", block.store(tile_sum, row_offset, col_offset, tile_height, tile_width, output)
                )

                # Return result
                ib.return_stmt(result)

            func = f.get_result()
            p.add_function(func)

        program = p.get_result()

        ir_str = str(program)
        assert "block.add" in ir_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
