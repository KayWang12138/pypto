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
"""Tests for IR serialization to and from files.

This file tests file I/O operations for IR serialization and deserialization.
"""

import os
import tempfile

import pytest
from pypto import ir
from pypto.ir import DataType


class TestFileIO:
    """Test serialization to and from files."""

    def test_serialize_to_file_and_back(self):
        """Test serializing to file and deserializing back."""
        span = ir.Span.unknown()
        x = ir.Var("x", ir.ScalarType(DataType.INT64), span)
        y = ir.Var("y", ir.ScalarType(DataType.INT64), span)
        add_expr = ir.Add(x, y, DataType.INT64, span)

        # Create temp file path in temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_ir.msgpack")

            # Serialize to file
            ir.serialize_to_file(add_expr, file_path)

            # Deserialize from file
            restored = ir.deserialize_from_file(file_path)
            assert restored is not None
            assert isinstance(restored, ir.Add)

            # Verify structural equality
            ir.assert_structural_equal(add_expr, restored, enable_auto_mapping = True)

    def test_large_ir_tree_file_io(self):
        """Test serializing large IR tree to file."""
        span = ir.Span.unknown()

        # Create a large statement sequence
        stmts = []
        for i in range(100):
            var = ir.Var(f"var_{i}", ir.ScalarType(DataType.INT64), span)
            const = ir.ConstInt(i, DataType.INT64, span)
            stmt = ir.AssignStmt(var, const, span)
            stmts.append(stmt)

        seq = ir.SeqStmts(stmts, span)

        # Create temp file path in temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "large_ir.msgpack")

            # Serialize to file
            ir.serialize_to_file(seq, file_path)

            # Deserialize from file
            restored = ir.deserialize_from_file(file_path)
            assert restored is not None
            assert isinstance(restored, ir.SeqStmts)
            assert len(restored.stmts) == 100

            # Verify structural equality
            ir.assert_structural_equal(seq, restored, enable_auto_mapping = True)
