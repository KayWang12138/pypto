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
from math import ceil, prod
import numpy as np
import torch
import pto
from test_case_runner import TestCaseRunner
from test_case_tools import get_dtype_by_name


def get_pto_dtype_by_name(name: str):
    str_to_dtype = {
        "int4": pto.DT_INT4,
        "int8": pto.DT_INT8,
        "int16": pto.DT_INT16,
        "int32": pto.DT_INT32,
        "int64": pto.DT_INT64,
        "fp8": pto.DT_FP8,
        "fp16": pto.DT_FP16,
        "fp32": pto.DT_FP32,
        "hf4": pto.DT_HF4,
        "hf8": pto.DT_HF8,
        "uint8": pto.DT_UINT8,
        "uint16": pto.DT_UINT16,
        "uint32": pto.DT_UINT32,
        "uint64": pto.DT_UINT64,
        "bool": pto.DT_BOOL,
        "double": pto.DT_DOUBLE,
        "bf16": pto.DT_BF16,
        "bottom": pto.DT_BOTTOM,
    }
    return str_to_dtype.get(name, pto.DT_FP32)


class PTOTestCaseRunner(TestCaseRunner):
    def __init__(
        self,
        operation: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: tuple,
        tile_shape: tuple,
        params: dict,
    ):
        super().__init__(view_shape, tile_shape, params)
        self._operation = operation
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors

    def gen_loop_range_tuple(self):
        if len(self._input_tensors[0].shape) != len(self._view_shape):
            raise ValueError(
                "The lengths of input tensors and view shape are not same."
            )
        return tuple(
            [
                ceil(self._input_tensors[0].shape[index] / self._view_shape[index])
                for index in list(range(len(self._view_shape)))
            ]
        )

    def input_tensors(self):
        return [
            pto.tensor(
                input_tensor.shape,
                get_pto_dtype_by_name(input_tensor.dtype),
                input_tensor.name,
            )
            for input_tensor in self._input_tensors
        ]

    def input_data(self):
        input_data = []
        for input_tensor in self._input_tensors:
            min_value = input_tensor.data_range.min
            max_value = input_tensor.data_range.max
            data = None
            if min_value != max_value:
                data = np.random.uniform(
                    min_value, max_value, prod(input_tensor.shape)
                ).astype(get_dtype_by_name(input_tensor.dtype))
            else:
                data = np.full(
                    prod(input_tensor.shape),
                    max_value,
                    dtype=get_dtype_by_name(input_tensor.dtype),
                )

            input_data.append(data)
        return input_data

    def output_tensors(self):
        return [
            pto.tensor(
                output_tensor.shape,
                get_pto_dtype_by_name(output_tensor.dtype),
                output_tensor.name,
            )
            for output_tensor in self._output_tensors
        ]

    def output_data(self):
        return [
            np.full(
                prod(output_tensor.shape),
                0,
                dtype=get_dtype_by_name(output_tensor.dtype),
            )
            for output_tensor in self._output_tensors
        ]

    def exec_dyn_func(self, input_tensors: list, output_tensors: list):
        loop_range_tuple = self.gen_loop_range_tuple()
        loop_desc = [
            ['"b0"', '"bIdx"', "bloop"],
            ['"s0"', '"sIdx"', "sloop"],
            ['"h0"', '"hIdx"', "hloop"],
            ['"n0"', '"nIdx"', "nloop"],
        ]
        tab = "    "
        prefix = tab
        dyn_function = "import pto\n"
        dyn_function += "\n"
        dyn_function += f"with pto.function('{self._operation}', input_tensors, output_tensors):\n"
        for index in list(range(len(loop_range_tuple))):
            dyn_function += prefix + (tab * (index + 1))
            dyn_function += (
                f"with pto.loop_function({loop_desc[index][0]}, {loop_desc[index][1]}, "
            )
            dyn_function += f"pto.loop_range({loop_range_tuple[index]})) as {loop_desc[index][2]}:\n"
        prefix = tab * (len(loop_range_tuple) + 1)
        for index in list(range(len(loop_range_tuple))):
            dyn_function += prefix + (tab * (index + 1))
            dyn_function += (
                f"for {loop_desc[index][1][1:-1]} in {loop_desc[index][2]}:\n"
            )
        prefix = tab * 2 * (len(loop_range_tuple) + 1)
        dyn_function += prefix + "input_data = []\n"
        view_offset = [
            f"{loop_desc[index][1][1:-1]} * {self._view_shape[index]}"
            for index in range(len(loop_range_tuple))
        ]
        for index in list(range(len(input_tensors))):
            dyn_function += prefix
            dyn_function += f"input_{index} = pto.view(input_tensors[{index}], {self._view_shape}, ["
            for idx in list(range(len(loop_range_tuple))):
                dyn_function += (
                    f"min(pto.symbolic_scalar({input_tensors[index].shape[idx]}) - "
                )
                dyn_function += f"{loop_desc[idx][1][1:-1]} * {self._view_shape[idx]}, "
                dyn_function += (
                    f"pto.symbolic_scalar({input_tensors[index].shape[idx]})), "
                )
            dyn_function += "], ["
            for offset in view_offset:
                dyn_function += offset + ", "
            dyn_function += "])\n"
            dyn_function += prefix + f"input_data.append(input_{index})\n"
        dyn_function += prefix + f"res = []\n"
        dyn_function += prefix + f"for _index in range(len(output_tensors)):\n"
        dyn_function += prefix + f"    res.append(pto.tensor())\n"
        dyn_function += prefix + f"if len(res) == 1:\n"
        dyn_function += prefix + f"    res[0].move(op_func(input_data, params))\n"
        dyn_function += prefix + f"else:\n"
        dyn_function += (
            prefix + f"    for dst_, src_ in zip(res, op_func(input_data, params)):\n"
        )
        dyn_function += prefix + f"        dst_.move(src_)\n"
        if self._operation == "Transpose":
            (
                view_offset[self._params["first_dim"]],
                view_offset[self._params["second_dim"]],
            ) = (
                view_offset[self._params["second_dim"]],
                view_offset[self._params["first_dim"]],
            )
        dyn_function += prefix + "for dst_, src_ in zip(output_tensors, res):\n"
        dyn_function += prefix + f"    pto.assemble(src_, ["
        for offset in view_offset:
            dyn_function += offset + ", "
        dyn_function += "], dst_)\n"

        dyn_function += prefix + "for input in input_data:\n"
        dyn_function += prefix + "    del input\n"
        dyn_function += prefix + "for tmp in res:\n"
        dyn_function += prefix + "    del tmp\n"
        print(dyn_function)
        pto.set_vec_tile_shapes(*self.tile_shape)
        exec(
            dyn_function,
            {
                "input_tensors": input_tensors,
                "output_tensors": output_tensors,
                "op_func": self._op_func,
                "params": self._params,
            },
        )

    def tear_up(self):
        pto.device_init()

    def tear_down(self):
        pto.device_fini()

    def run_on_device(self, inputs: list) -> list:
        output = self.output_data()
        pto.device_run_once_data_from_host(inputs, output)
        return [
            torch.tensor(
                output[index],
                dtype=get_dtype_by_name(self._output_tensors[index].dtype, True),
            ).reshape(self._output_tensors[index].shape)
            for index in list(range(len(output)))
        ]
