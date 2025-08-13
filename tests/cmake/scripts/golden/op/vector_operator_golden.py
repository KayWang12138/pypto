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
"""vector op 相关用例 Golden 生成逻辑.

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
"""
import sys
import logging
import json
from pathlib import Path
from typing import List

import numpy as np
from bfloat16 import bfloat16
import torch


def get_dtype_by_name(name: str, is_torch: bool = False):
    str_to_dtype = {
        "int8": [np.int8, torch.int8],
        "int16": [np.int16, torch.int16],
        "int32": [np.int32, torch.int32],
        "int64": [np.int64, torch.int64],
        "fp16": [np.float16, torch.float16],
        "fp32": [np.float32, torch.float32],
        "fp64": [np.float64, torch.float64],
        "uint8": [np.uint8, torch.uint8],
        "uint16": [np.uint16, None],
        "uint32": [np.uint32, None],
        "uint64": [np.uint64, None],
        "bool": [np.bool_, torch.bool],
        "double": [np.float64, torch.double],
        "complex64": [np.complex64, torch.complex64],
        "complex128": [np.complex128, torch.complex64],
        "bf16": [bfloat16, torch.bfloat16],
    }
    return str_to_dtype.get(name, [np.float32, torch.float32])[is_torch]


if __name__ == "__main__":
    # 日志级别
    logging.basicConfig(
        format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister
else:
    from golden_register import GoldenRegister


def dump_file(data_pool, data_path, type_str):
    np.array(data_pool).astype(get_dtype_by_name(type_str.lower())).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def gen_op_golden(
    operation: str, golden_func, output_path: Path, case_index: int = None
) -> bool:
    def generate_golden_files(golden_func, output_path: Path, config: dict) -> bool:
        input_tensors = []
        spec_value_map = {
            "nan": np.nan,
            "inf": np.inf,
            "-inf": -np.inf,
            "max": np.finfo(np.float32).max,
            "min": np.finfo(np.float32).min,
        }
        for input_tensor in config["input_tensors"]:
            min = input_tensor["data_range"]["min"]
            max = input_tensor["data_range"]["max"]
            if min != max:
                assert not isinstance(min, str) and not isinstance(
                    min, str
                ), "Data range must be number when the min and max are not same."
                tensor = np.random.uniform(min, max, input_tensor["shape"]).astype(
                    get_dtype_by_name(input_tensor["dtype"])
                )
            else:
                if isinstance(min, str):
                    assert (
                        min in spec_value_map.keys()
                    ), f"Data range of input tensor {input_tensor} has invalid value."
                    max = spec_value_map.get(max)
                tensor = np.full(
                    input_tensor["shape"],
                    max,
                    dtype=get_dtype_by_name(input_tensor["dtype"]),
                )
            input_tensors.append(tensor)
            tensor.tofile(Path(output_path, input_tensor["name"] + ".bin"))

        res = (
            golden_func(input_tensors)
            if len(config["params"]) <= 1
            else golden_func(input_tensors, config["params"])
        )
        for idx in range(len(config["output_tensors"])):
            res[idx].tofile(
                Path(output_path, config["output_tensors"][idx]["name"] + ".bin")
            )
        return True

    case_data_file = str(output_path) + "/test_case_data.json"
    with open(case_data_file, "r") as data_file:
        test_config = json.load(data_file)
    assert test_config is not None, f"Load {case_data_file} failed."

    # 1.跑测试套还是单个用例，生成不同场景的文件夹，一般不用改
    # 2.涉及test_configs数据结构变更，generate_golden_files的接口形式和调用传参可能需联动修改
    if case_index is None:
        output_path = Path(str(output_path) + "/" + str(case_index))
        output_path.mkdir(parents=True, exist_ok=True)
        generate_golden_files(golden_func, output_path, test_config)
    else:
        generate_golden_files(golden_func, output_path, test_config)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestReduceSum/ReduceSumFirstAxisOperationTest.test_reduce_sum",
        "TestReduceSum/ReduceSum4DOperationTest.test_reduce_sum",
        "TestReduceSum/ReduceSum3DOperationTest.test_reduce_sum",
        "TestReduceSum/ReduceSumOperationTest.test_reduce_sum",
        "TestReduceMax/ReduceMaxOperationTest.test_reduce_max",
        "TestReduceMax/ReduceMax3DOperationTest.test_reduce_max",
        "TestReduceMax/ReduceMax4DOperationTest.test_reduce_max",
    ]
)
def gen_reduce_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    shape_list_i = []
    reduce_axis = []
    op_type = "sum"
    if "ReduceSumOperationTest" in case_name:
        shape_list_i = [[512, 128], [4, 1000]]
        reduce_axis = [-1]
    elif "ReduceSum3DOperationTest" in case_name:
        shape_list_i = [[48, 32, 32]]
        reduce_axis = [-1]
    elif "ReduceSum4DOperationTest" in case_name:
        shape_list_i = [[8, 6, 10, 8]]
        reduce_axis = [-1]
    elif "ReduceSumFirstAxisOperationTest" in case_name:
        shape_list_i = [[8, 128]]
        reduce_axis = [0]
    elif "ReduceMaxOperationTest" in case_name:
        shape_list_i = [[512, 128], [4, 1000]]
        reduce_axis = [-1]
        op_type = "max"
    elif "ReduceMax3DOperationTest" in case_name:
        shape_list_i = [[48, 32, 32]]
        reduce_axis = [-1]
        op_type = "max"
    elif "ReduceMax4DOperationTest" in case_name:
        shape_list_i = [[8, 6, 10, 8]]
        reduce_axis = [-1]
        op_type = "max"
    if case_index is None:
        for index, arr in enumerate(shape_list_i):
            output_path = Path(str(output) + "/" + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            x_path = Path(output_path, "x.bin")
            o_path = Path(output_path, "res.bin")
            complete = x_path.exists() and o_path.exists()
            if complete:
                logging.debug("Case(%s), Golden complete.", case_name)
                return True
            else:
                x = np.random.uniform(0, 1, arr).astype(np.float32)
                x.tofile(x_path)
                if op_type == "max":
                    y = x.max(axis=reduce_axis[0], keepdims=True)
                elif op_type == "sum":
                    y = x.sum(axis=reduce_axis[0], keepdims=True)
                y.tofile(o_path)
    else:
        x_path = Path(output, "x.bin")
        o_path = Path(output, "res.bin")
        complete = x_path.exists() and o_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = np.random.uniform(0, 1, shape_list_i[case_index]).astype(np.float32)
            x.tofile(x_path)
            if op_type == "max":
                y = x.max(axis=reduce_axis[0], keepdims=True)
            elif op_type == "sum":
                y = x.sum(axis=reduce_axis[0], keepdims=True)
            y.tofile(o_path)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestTopK/TopK4DOperationTest.test_topk",
        "TestTopK/TopK3DOperationTest.test_topk",
        "TestTopK/TopKOperationTest.test_topk",
    ]
)
def gen_topk_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    shape_list_i = []
    dim_axis = []
    k_list = []
    largest_list = []
    if "TopKOperationTest" in case_name:
        shape_list_i = [[128, 32]]
        dim_axis = [-1]
        k_list = [8]
        largest_list = [True]
    elif "TopK3DOperationTest" in case_name:
        shape_list_i = [[8, 2, 8]]
        dim_axis = [-1]
        k_list = [8]
        largest_list = [True]
    elif "TopK4DOperationTest" in case_name:
        shape_list_i = [[2, 8, 2, 8]]
        dim_axis = [-1]
        k_list = [8]
        largest_list = [True]
    if case_index is None:
        for index, arr in enumerate(shape_list_i):
            output_path = Path(str(output) + "/" + str(index))
            output_path.mkdir(parents=True, exist_ok=True)
            x_path = Path(output_path, "x.bin")
            o0_path = Path(output_path, "value.bin")
            o1_path = Path(output_path, "index.bin")
            complete = x_path.exists() and o0_path.exists()
            if complete:
                logging.debug("Case(%s), Golden complete.", case_name)
                return True
            else:
                x = torch.randn(arr)
                val, idx = x.topk(k_list[index], dim=dim_axis[index], largest=largest_list[index], sorted=True)
                x.numpy().tofile(x_path)
                val.numpy().tofile(o0_path)
                idx.numpy().astype(np.int32).tofile(o1_path)
    else:
        x_path = Path(output, "x.bin")
        o0_path = Path(output, "value.bin")
        o1_path = Path(output, "index.bin")
        complete = x_path.exists() and o0_path.exists()
        if complete:
            logging.debug("Case(%s), Golden complete.", case_name)
            return True
        else:
            x = torch.randn(shape_list_i[case_index])
            val, idx = x.topk(k_list[case_index], dim=dim_axis[case_index],
             largest=largest_list[case_index], sorted=True)
            x.numpy().tofile(x_path)
            val.numpy().tofile(o0_path)
            idx.numpy().astype(np.int32).tofile(o1_path)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestCast/CastOperationTest.TestCast",
    ]
)
def gen_cast_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, params: dict):
        dtype_out = get_dtype_by_name(params["dst_dtype"], True)
        if dtype_out is None:
            return [inputs[0].astype(get_dtype_by_name(params["dst_dtype"]))]
        x = torch.from_numpy(inputs[0])
        if dtype_out == torch.bfloat16:
            x = x.to(torch.float32).numpy().astype(bfloat16)
        else:
            x = x.to(dtype_out).numpy()

        return [x]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Cast", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestExp/ExpOperationTest.TestExp",
    ]
)
def gen_exp_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs):
        return [np.exp(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Exp", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSqrt/SqrtOperationTest.TestSqrt",
    ]
)
def gen_sqrt_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs):
        return [np.sqrt(inputs[0])]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sqrt", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAdd/AddOperationTest.TestAdd",
    ]
)
def gen_add_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs):
        return [inputs[0] + inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Add", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSub/SubOperationTest.TestSub",
    ]
)
def gen_sub_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs):
        return [inputs[0] - inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Sub", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMul/MulOperationTest.TestMul",
    ]
)
def gen_mul_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs):
        return [inputs[0] * inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Mul", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDiv/DivOperationTest.TestDiv",
    ]
)
def gen_div_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs):
        return [inputs[0] / inputs[1]]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Div", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestAdds/AddsOperationTest.TestAdds",
    ]
)
def gen_adds_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, params: dict):
        scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] + scalar]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Adds", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestMuls/MulsOperationTest.TestMuls",
    ]
)
def gen_muls_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, params: dict):
        scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] * scalar]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Muls", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestDivs/DivsOperationTest.TestDivs",
    ]
)
def gen_divs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, params: dict):
        scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] / scalar]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Divs", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestVectorDuplicate/VectorDuplicateOperationTest.TestVectorDuplicate",
    ]
)
def gen_vector_dup_op_golden(
    case_name: str, output: Path, case_index: int = None
) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, params: dict):
        scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [np.full(inputs[0].shape, scalar, inputs[0].dtype)]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("VectorDuplicate", golden_func, output, case_index)


@GoldenRegister.reg_golden_func(
    case_names=[
        "TestSubs/SubsOperationTest.TestSubs",
    ]
)
def gen_subs_op_golden(case_name: str, output: Path, case_index: int = None) -> bool:
    # golden开发者需要根据具体golden逻辑修改，不同注册函数内的generate_golden_files可重名
    def golden_func(inputs, params: dict):
        scalar = get_dtype_by_name(params["scalar_type"])(params["scalar"])
        return [inputs[0] - scalar]

    logging.debug("Case(%s), Golden creating...", case_name)
    return gen_op_golden("Subs", golden_func, output, case_index)


def main() -> bool:
    # 用例名称
    case_name_list: List[str] = [
        "TestAdd/AddOperationTest.TestAdd",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = gen_add_op_golden(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)