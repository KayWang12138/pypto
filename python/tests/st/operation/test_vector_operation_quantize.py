#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Test cases for Quantize operation
"""
import pypto
import pytest

from test_case_class_vector_operations import QuantizeTestCase


def test_quantize_symmetric_int8_axis1():
    """Test symmetric quantization: FP32 -> INT8 with axis=-1"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-127, 127],
        },
        {
            "name": "scale",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "int8",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "0",
        "Quantize_symmetric_int8_axis1",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "int8", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_quantize_symmetric_int8_axis2():
    """Test symmetric quantization: FP32 -> INT8 with axis=-2"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-127, 127],
        },
        {
            "name": "scale",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "int8",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "1",
        "Quantize_symmetric_int8_axis2",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "int8", "axis": -2, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_quantize_asymmetric_uint8_axis1():
    """Test asymmetric quantization: FP32 -> UINT8 with axis=-1"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [0, 255],
        },
        {
            "name": "scale",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        },
        {
            "name": "zero_points",
            "shape": (32, 1),
            "dtype": "fp32",
            "data_range": [-128, 127],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "uint8",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "2",
        "Quantize_asymmetric_uint8_axis1",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "uint8", "axis": -1, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_quantize_asymmetric_uint8_axis2():
    """Test asymmetric quantization: FP32 -> UINT8 with axis=-2"""
    original_shape = (32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [0, 255],
        },
        {
            "name": "scale",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        },
        {
            "name": "zero_points",
            "shape": (1, 64),
            "dtype": "fp32",
            "data_range": [-128, 127],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "uint8",
        }
    ]
    view_shape = (32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "3",
        "Quantize_asymmetric_uint8_axis2",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "uint8", "axis": -2, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_quantize_3d_symmetric():
    """Test symmetric quantization with 3D input"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-127, 127],
        },
        {
            "name": "scale",
            "shape": (4, 1, 1),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "int8",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "4",
        "Quantize_3d_symmetric",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "int8", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_quantize_3d_axis2_symmetric():
    """Test symmetric quantization with 3D input and axis=-2"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-127, 127],
        },
        {
            "name": "scale",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "int8",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "6",
        "Quantize_3d_axis2_symmetric",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "int8", "axis": -2, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_quantize_3d_axis2_asymmetric():
    """Test asymmetric quantization with 3D input and axis=-2"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [0, 255],
        },
        {
            "name": "scale",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        },
        {
            "name": "zero_points",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [-128, 127],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "uint8",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "7",
        "Quantize_3d_axis2_asymmetric",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "uint8", "axis": -2, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_quantize_4d_symmetric():
    """Test symmetric quantization with 4D input"""
    original_shape = (2, 2, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-127, 127],
        },
        {
            "name": "scale",
            "shape": (2, 2, 1, 1),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "int8",
        }
    ]
    view_shape = (2, 2, 32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "5",
        "Quantize_4d_symmetric",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "int8", "axis": -1, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_quantize_3d_axis2_symmetric():
    """Test symmetric quantization with 3D input and axis=-2"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-127, 127],
        },
        {
            "name": "scale",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "int8",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "6",
        "Quantize_3d_axis2_symmetric",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "int8", "axis": -2, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_quantize_3d_axis2_asymmetric():
    """Test asymmetric quantization with 3D input and axis=-2"""
    original_shape = (4, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [0, 255],
        },
        {
            "name": "scale",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        },
        {
            "name": "zero_points",
            "shape": (4, 1, 64),
            "dtype": "fp32",
            "data_range": [-128, 127],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "uint8",
        }
    ]
    view_shape = (4, 32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "7",
        "Quantize_3d_axis2_asymmetric",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "uint8", "axis": -2, "is_asymmetric": True},
    )
    test_case.exec(True)


def test_quantize_4d_axis2_symmetric():
    """Test symmetric quantization with 4D input and axis=-2"""
    original_shape = (2, 2, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [-127, 127],
        },
        {
            "name": "scale",
            "shape": (2, 2, 1, 64),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "int8",
        }
    ]
    view_shape = (2, 2, 32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "8",
        "Quantize_4d_axis2_symmetric",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "int8", "axis": -2, "is_asymmetric": False},
    )
    test_case.exec(True)


def test_quantize_4d_axis2_asymmetric():
    """Test asymmetric quantization with 4D input and axis=-2"""
    original_shape = (2, 2, 32, 64)
    input_tensors = [
        {
            "name": "input",
            "shape": original_shape,
            "dtype": "fp32",
            "data_range": [0, 255],
        },
        {
            "name": "scale",
            "shape": (2, 2, 1, 64),
            "dtype": "fp32",
            "data_range": [0.1, 2.0],
        },
        {
            "name": "zero_points",
            "shape": (2, 2, 1, 64),
            "dtype": "fp32",
            "data_range": [-128, 127],
        }
    ]
    output_tensors = [
        {
            "name": "output",
            "shape": original_shape,
            "dtype": "uint8",
        }
    ]
    view_shape = (2, 2, 32, 64)
    tile_shape = (4, 16)
    test_case = QuantizeTestCase(
        "9",
        "Quantize_4d_axis2_asymmetric",
        input_tensors,
        output_tensors,
        view_shape,
        tile_shape,
        {"otype": "uint8", "axis": -2, "is_asymmetric": True},
    )
    test_case.exec(True)
