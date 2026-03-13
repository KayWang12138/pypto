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
from pathlib import Path
import sys
import pypto
import torch

from pto_test_case_runner import PTOTestCaseRunner, get_pto_dtype_by_name

helper_path: Path = Path(
    Path(__file__).parent.parent.parent.parent.parent,
    "framework/tests/cmake/scripts/helper",
).resolve()
if str(helper_path) not in sys.path:
    sys.path.append(str(helper_path))
from test_case import TestCase
from test_case_tools import get_dtype_by_name, parse_list_str


class AddTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "Add",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "Add", input_tensors, output_tensors, view_shape, tile_shape, params
            ),
        )

    def run_in_dyn_func(self, inputs, _params: dict) -> dict:
        return pypto.add(*inputs)

    def golden_func(self, inputs, _params: dict) -> list:
        return [torch.add(*inputs)]

    def golden_func_params(self) -> dict:
        return {}


class CastTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "Cast",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "Cast", input_tensors, output_tensors, view_shape, tile_shape, params
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        return pypto.cast(
            *inputs, get_pto_dtype_by_name(params["dst_dtype"])
        )

    def golden_func(self, inputs, params: dict) -> list:
        return [inputs[0].to(get_dtype_by_name(params["dst_dtype"], True))]

    def golden_func_params(self) -> dict:
        return {
            "dst_dtype": self._case_desc.output_tensors[0].dtype,
            "mode": self._case_desc.params.get("mode"),
        }


class ExpTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "Exp",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "Exp", input_tensors, output_tensors, view_shape, tile_shape, params
            ),
        )

    def run_in_dyn_func(self, inputs, _params: dict) -> dict:
        return pypto.exp(*inputs)

    def golden_func(self, inputs, _params: dict) -> list:
        return [torch.exp(*inputs)]

    def golden_func_params(self) -> dict:
        return {}


class ScalarAddSTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "ScalarAddS",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "ScalarAddS",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        scalar = inputs[0]
        return pypto.add(*inputs, params.get("scalar"))

    def golden_func(self, inputs, params: dict) -> list:
        return [torch.add(*inputs, params.get("scalar"))]

    def golden_func_params(self) -> dict:
        return {
            "scalar": float(self._case_desc.params.get("scalar")),
            "reverse": bool(self._case_desc.params.get("reverse")),
        }


class ScalarSubSTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "ScalarSubS",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "ScalarSubS",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        return pypto.sub(*inputs, params.get("scalar"))

    def golden_func(self, inputs, params: dict) -> list:
        return [torch.sub(*inputs, params.get("scalar"))]

    def golden_func_params(self) -> dict:
        return {
            "scalar": float(self._case_desc.params.get("scalar")),
            "reverse": bool(self._case_desc.params.get("reverse")),
        }


class ScalarMulSTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "ScalarMulS",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "ScalarMulS",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        return pypto.mul(*inputs, params.get("scalar"))

    def golden_func(self, inputs, params: dict) -> list:
        return [torch.mul(*inputs, params.get("scalar"))]

    def golden_func_params(self) -> dict:
        return {
            "scalar": float(self._case_desc.params.get("scalar")),
            "reverse": bool(self._case_desc.params.get("reverse")),
        }


class ScalarDivSTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "ScalarDivS",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "ScalarDivS",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:

        return pypto.div(*inputs, params.get("scalar"))

    def golden_func(self, inputs, params: dict) -> list:
        return [torch.div(*inputs, params.get("scalar"))]

    def golden_func_params(self) -> dict:
        return {
            "scalar": float(self._case_desc.params.get("scalar")),
            "reverse": bool(self._case_desc.params.get("reverse")),
        }


class PowsTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "Pows",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "Pows",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        return pypto.pow(*inputs, params.get("scalar"))

    def golden_func(self, inputs, params: dict) -> list:
        return [torch.pow(*inputs, params.get("scalar"))]

    def golden_func_params(self) -> dict:
        return {
            "scalar": float(self._case_desc.params.get("scalar")),
        }


class ScalarMaxSTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "ScalarMaxS",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "ScalarMaxS",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        scalar = params.get("scalar")
        return pypto.scalar_maxs(*inputs, scalar, params.get("reverse"))

    def golden_func(self, inputs, params: dict) -> list:
        scalar = torch.full(
            inputs[0].shape, params.get("scalar"), dtype=inputs[0].dtype
        )
        return [torch.max(*inputs, scalar)]

    def golden_func_params(self) -> dict:
        return {
            "scalar": float(self._case_desc.params.get("scalar")),
            "reverse": bool(self._case_desc.params.get("reverse")),
        }


class TransposeTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "Transpose",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "Transpose",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        return pypto.transpose(*inputs, params.get("first_dim"), params.get("second_dim"))

    def golden_func(self, inputs, params: dict) -> list:
        return [
            torch.transpose(*inputs, params.get("first_dim"),
                            params.get("second_dim"))
        ]

    def golden_func_params(self) -> dict:
        return {
            "dims": parse_list_str(self._case_desc.params.get("dims")),
            "first_dim": parse_list_str(self._case_desc.params.get("first_dim"))[0],
            "second_dim": parse_list_str(self._case_desc.params.get("second_dim"))[0],
        }


class TopKTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "TopK",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "TopK", input_tensors, output_tensors, view_shape, tile_shape, params
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        return pypto.topk(*inputs, params.get("count"), params.get("dims"), params.get("islargest"))

    def golden_func(self, inputs, params: dict) -> list:
        return torch.topk(*inputs, params.get("count"), params.get("dims"), params.get("islargest"))

    def golden_func_params(self) -> dict:
        return {
            "dims": parse_list_str(self._case_desc.params.get("dims"))[0],
            "count": int(self._case_desc.params.get("count")),
            "islargest": bool(
                parse_list_str(self._case_desc.params.get("islargest"))[0]
            ),
        }


class LogicalAndTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "LogicalAnd",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "LogicalAnd",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, _params: dict) -> dict:
        return pypto.logical_and(*inputs)

    def golden_func(self, inputs, _params: dict) -> list:
        return [torch.logical_and(*inputs)]

    def golden_func_params(self) -> dict:
        return {}


class CbrtTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "Cbrt",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "Cbrt",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, _params: dict) -> dict:
        return pypto.cbrt(*inputs)

    def golden_func(self, inputs, _params: dict) -> list:
        return [torch._prims.cbrt(*inputs)]

    def golden_func_params(self) -> dict:
        return {}


class ExpandExpDifTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "ExpandExpDif",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "ExpandExpDif",
                input_tensors,
                output_tensors,
                view_shape,
                tile_shape,
                params,
            ),
        )

    def run_in_dyn_func(self, inputs, _params: dict) -> dict:
        return pypto.expand_exp_dif(*inputs)

    def golden_func(self, inputs, _params: dict) -> list:
        dtype_out = inputs[0].dtype
        dtype_in = torch.float32 if dtype_out == torch.bfloat16 else dtype_out
        return [
            torch.exp(torch.sub(inputs[0].to(dtype_in), inputs[1].to(dtype_in))).to(
                dtype_out
            )
        ]

    def golden_func_params(self) -> dict:
        return {}


class QuantizeTestCase(TestCase):
    def __init__(
        self,
        case_index: str,
        case_name: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(
            case_index,
            case_name,
            "Quantize",
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
            PTOTestCaseRunner(
                "Quantize", input_tensors, output_tensors, view_shape, tile_shape, params
            ),
        )

    def run_in_dyn_func(self, inputs, params: dict) -> dict:
        output_dtype = get_pto_dtype_by_name(params["otype"])
        axis = params.get("axis", -1)

        if params.get("is_asymmetric", False):
            return pypto.quantize(inputs[0], inputs[1], output_dtype, axis, inputs[2])
        else:
            return pypto.quantize(inputs[0], inputs[1], output_dtype, axis)

    def golden_func(self, inputs, params: dict) -> list:
        import numpy as np

        input_tensor = inputs[0]
        scale_tensor = inputs[1]
        axis = params.get("axis", -1)

        # Normalize axis to positive index
        if axis < 0:
            axis = len(input_tensor.shape) + axis

        # Broadcast scale to input shape
        if scale_tensor.shape != input_tensor.shape:
            # For axis=-1: scale shape is [..., row, 1]
            # For axis=-2: scale shape is [..., 1, col]
            scale_tensor = scale_tensor.broadcast_to(input_tensor.shape)

        if params.get("is_asymmetric", False):
            zero_points = inputs[2]
            if zero_points.shape != input_tensor.shape:
                zero_points = zero_points.broadcast_to(input_tensor.shape)
            # Perform quantization
            scaled = input_tensor * scale_tensor + zero_points
            output = torch.round(scaled).to(torch.uint8)
        else:
            # Perform quantization
            scaled = input_tensor * scale_tensor
            output = torch.round(scaled).to(torch.int8)

        return [output]

    def golden_func_params(self) -> dict:
        return {
            "otype": self._case_desc.output_tensors[0].dtype,
            "axis": self._case_desc.params.get("axis", -1),
            "is_asymmetric": self._case_desc.params.get("is_asymmetric", False),
        }
        return {}
