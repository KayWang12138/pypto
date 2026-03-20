#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file in the compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Test cases for Dequantize operation
"""
import pypto
import pytest

from test_case_class_vector_operations import DequantizeTestCase


# ============================================================
# 2D Tensor Tests - INT8
# ============================================================

def test_dequantize_symmetric_int8_axis1():
    """Test symmetric dequantization: INT8 -> FP32 with axis=-1"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "0",
        "Dequantize_symmetric_int8_axis1",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_dequantize_symmetric_int8_axis2():
    """Test symmetric dequantization: INT8 -> FP32 with axis=-2"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "1",
        "Dequantize_symmetric_int8_axis2",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -2, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_dequantize_asymmetric_int8_axis1():
    """Test asymmetric dequantization: INT8 -> FP32 with axis=-1"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        },
        {
            "name": "zero_points",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [-10, 10],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "2",
        "Dequantize_asymmetric_int8_axis1",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_dequantize_asymmetric_int8_axis2():
    """Test asymmetric dequantization: INT8 -> FP32 with axis=-2"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        },
        {
            "name": "zero_points",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [-10, 10],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "3",
        "Dequantize_asymmetric_int8_axis2",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -2, "is_asymmetric": True},
    )
    test_case.exec(True)


# ============================================================
# 2D Tensor Tests - INT16
# ============================================================

def test_dequantize_symmetric_int16_axis1():
    """Test symmetric dequantization: INT16 -> FP32 with axis=-1"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int16",
            "data_range": [-32768, 32767],
        },
        {
            "name": "scale",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [0.00001, 0.001],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "4",
        "Dequantize_symmetric_int16_axis1",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_dequantize_symmetric_int16_axis2():
    """Test symmetric dequantization: INT16 -> FP32 with axis=-2"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int16",
            "data_range": [-32768, 32767],
        },
        {
            "name": "scale",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [0.00001, 0.001],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "5",
        "Dequantize_symmetric_int16_axis2",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -2, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_dequantize_asymmetric_int16_axis1():
    """Test asymmetric dequantization: INT16 -> FP32 with axis=-1"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int16",
            "data_range": [-32768, 32767],
        },
        {
            "name": "scale",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [0.00001, 0.001],
        },
        {
            "name": "zero_points",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [-100, 100],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "6",
        "Dequantize_asymmetric_int16_axis1",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_dequantize_asymmetric_int16_axis2():
    """Test asymmetric dequantization: INT16 -> FP32 with axis=-2"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int16",
            "data_range": [-32768, 32767],
        },
        {
            "name": "scale",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [0.00001, 0.001],
        },
        {
            "name": "zero_points",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [-100, 100],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "7",
        "Dequantize_asymmetric_int16_axis2",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -2, "is_asymmetric": True},
    )
    test_case.exec(True)


# ============================================================
# 3D Tensor Tests
# ============================================================

def test_dequantize_3d_symmetric_int8():
    """Test symmetric dequantization with 3D input (INT8)"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (4, 1, 1),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "8",
        "Dequantize_3d_symmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_dequantize_3d_axis2_symmetric_int8():
    """Test symmetric dequantization with 3D input and axis=-2 (INT8)"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "9",
        "Dequantize_3d_axis2_symmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -2, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_dequantize_3d_asymmetric_int8():
    """Test asymmetric dequantization with 3D input (INT8)"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (4, 1, 1),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        },
        {
            "name": "zero_points",
            "shape": (4, 1, 1),
            "dtype": "fp32",
            "data_range": [-10, 10],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "10",
        "Dequantize_3d_asymmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_dequantize_3d_axis2_asymmetric_int8():
    """Test asymmetric dequantization with 3D input and axis=-2 (INT8)"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        },
        {
            "name": "zero_points",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [-10, 10],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "11",
        "Dequantize_3d_axis2_asymmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -2, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_dequantize_3d_symmetric_int16():
    """Test symmetric dequantization with 3D input (INT16)"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int16",
            "data_range": [-32768, 32767],
        },
        {
            "name": "scale",
            "shape": (4, 1, 1),
            "dtype": "fp32",
            "data_range": [0.00001, 0.001],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "12",
        "Dequantize_3d_symmetric_int16",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


# ============================================================
# 4D Tensor Tests
# ============================================================

def test_dequantize_4d_symmetric_int8():
    """Test symmetric dequantization with 4D input (INT8)"""
    original_shape = (2, 2, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (2, 2, 1, 1),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (2, 2, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "13",
        "Dequantize_4d_symmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_dequantize_4d_axis2_symmetric_int8():
    """Test symmetric dequantization with 4D input and axis=-2 (INT8)"""
    original_shape = (2, 2, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (2, 2, 1, 64),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (2, 2, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "14",
        "Dequantize_4d_axis2_symmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -2, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_dequantize_4d_asymmetric_int8():
    """Test asymmetric dequantization with 4D input (INT8)"""
    original_shape = (2, 2, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (2, 2, 1, 1),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        },
        {
            "name": "zero_points",
            "shape": (2, 2, 1, 1),
            "dtype": "fp32",
            "data_range": [-10, 10],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (2, 2, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "15",
        "Dequantize_4d_asymmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_dequantize_4d_axis2_asymmetric_int8():
    """Test asymmetric dequantization with 4D input and axis=-2 (INT8)"""
    original_shape = (2, 2, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (2, 2, 1, 64),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        },
        {
            "name": "zero_points",
            "shape": (2, 2, 1, 64),
            "dtype": "fp32",
            "data_range": [-10, 10],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (2, 2, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "16",
        "Dequantize_4d_axis2_asymmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -2, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_dequantize_4d_symmetric_int16():
    """Test symmetric dequantization with 4D input (INT16)"""
    original_shape = (2, 2, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int16",
            "data_range": [-32768, 32767],
        },
        {
            "name": "scale",
            "shape": (2, 2, 1, 1),
            "dtype": "fp32",
            "data_range": [0.00001, 0.001],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (2, 2, 32, 64)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "17",
        "Dequantize_4d_symmetric_int16",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


# ============================================================
# 5D Tensor Tests
# ============================================================

def test_dequantize_5d_symmetric_int8():
    """Test symmetric dequantization with 5D input (INT8)"""
    original_shape = (2, 2, 2, 16, 32)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "int8",
            "data_range": [-128, 127],
        },
        {
            "name": "scale",
            "shape": (2, 2, 2, 1, 1),
            "dtype": "fp32",
            "data_range": [0.001, 0.1],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "fp32",
        }
    ]
    view_shape = (2, 2, 2, 16, 32)
    tile_shape = (4, 16)
    test_case = DequantizeTestCase(
        "18",
        "Dequantize_5d_symmetric_int8",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "fp32", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)
