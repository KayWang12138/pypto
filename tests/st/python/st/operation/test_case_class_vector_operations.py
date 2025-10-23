#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
""" """
from pathlib import Path
import sys
import pto
import torch

tools_path: Path = Path(Path(__file__).parent, "../../../utils/python")
if str(tools_path) not in sys.path:
    sys.path.append(str(tools_path))
from pto_test_case_runner import PTOTestCaseRunner, get_pto_dtype_by_name
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
        return pto.add(*inputs)

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
        return pto.cast(
            *inputs, get_pto_dtype_by_name(params["dst_dtype"]), params["mode"]
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
        return pto.exp(*inputs)

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
        scalar = pto.element(inputs[0].get_dtype(), params.get("scalar"))
        return pto.scalar_adds(*inputs, scalar, params.get("reverse"))

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
        scalar = pto.element(inputs[0].get_dtype(), params.get("scalar"))
        return pto.scalar_subs(*inputs, scalar, params.get("reverse"))

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
        scalar = pto.element(inputs[0].get_dtype(), params.get("scalar"))
        return pto.scalar_muls(*inputs, scalar, params.get("reverse"))

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
        scalar = pto.element(inputs[0].get_dtype(), params.get("scalar"))
        return pto.scalar_divs(*inputs, scalar, params.get("reverse"))

    def golden_func(self, inputs, params: dict) -> list:
        return [torch.div(*inputs, params.get("scalar"))]

    def golden_func_params(self) -> dict:
        return {
            "scalar": float(self._case_desc.params.get("scalar")),
            "reverse": bool(self._case_desc.params.get("reverse")),
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
        scalar = pto.element(inputs[0].get_dtype(), params.get("scalar"))
        return pto.scalar_maxs(*inputs, scalar, params.get("reverse"))

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
        return pto.transpose(*inputs, params.get("dims"))

    def golden_func(self, inputs, params: dict) -> list:
        return [
            torch.transpose(*inputs, params.get("first_dim"), params.get("second_dim"))
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
        return pto.topk(*inputs, params.get("count"), params.get("dims"))

    def golden_func(self, inputs, params: dict) -> list:
        return torch.topk(*inputs, params.get("count"))

    def golden_func_params(self) -> dict:
        return {
            "dims": parse_list_str(self._case_desc.params.get("dims"))[0],
            "count": int(self._case_desc.params.get("count")),
            "islargest": bool(
                parse_list_str(self._case_desc.params.get("islargest"))[0]
            ),
        }
