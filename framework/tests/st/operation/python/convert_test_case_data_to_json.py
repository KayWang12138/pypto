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

import logging
import os
import json
from pathlib import Path
import sys
from dataclasses import dataclass
import pandas as pd

utils_path: Path = Path(Path(__file__).parent.parent.parent.parent.parent.parent, "python/tests/st/utils").resolve()
if str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))

from test_case_desc import TensorDesc, TestCaseDesc
from test_case_tools import parse_list_str


@dataclass
class MatmulParam:
    trans_list: list
    input_format_list: list
    output_format_list: list
    row_data: dict
    output_dtype: str
    is_k_split: bool


class TestDataReader:
    def __init__(self, case_index: int, case_data, json_path: str):
        self._case_index = case_index
        self._case_data = case_data
        self._json_path = json_path

    def convert_row_data(self, row):
        row_data = row.to_dict()
        input_shape = parse_list_str(row_data.pop("input_shape"))
        if not isinstance(input_shape[0], (list, tuple)):
            input_shape = [input_shape]
        input_dtype = parse_list_str(row_data.pop("input_dtype"))
        data_range = parse_list_str(row_data.pop("input_datarange"))
        if not isinstance(data_range[0], (list, tuple)):
            data_range = [data_range]
        assert len(input_shape) == len(input_dtype)
        assert len(input_shape) == len(data_range)

        input_format_list = parse_list_str(row_data.pop("input_format"))
        assert len(input_format_list) == len(input_shape)

        is_trans_list = []
        # 转换布尔值（支持TRUE/FALSE/1/0）
        isATrans = row_data.pop("isATrans", None)
        isBTrans = row_data.pop("isBTrans", None)
        if isATrans is not None and isBTrans is not None:
            is_a_trans = self.str_to_bool(isATrans)
            is_b_trans = self.str_to_bool(isBTrans)
            is_trans_list = [is_a_trans, is_b_trans]
        # 若输入数量超过2个，剩余默认不转置
        while len(is_trans_list) < len(input_shape):
            is_trans_list.append(None)

        is_k_split = False
        enable_k_split = row_data.pop("enableKSplit", None)
        if enable_k_split is not None:
            is_k_split = self.str_to_bool(enable_k_split)

        input_tensors = []
        for idx in range(len(input_shape)):
            input_tensors.append(
                TensorDesc(
                    "input" + str(idx),
                    input_shape[idx],
                    input_dtype[idx],
                    data_range[idx],
                    tensor_format=input_format_list[idx],
                    is_trans=is_trans_list[idx],
                )
            )
        output_shape = parse_list_str(row_data.pop("output_shape"))
        if not isinstance(output_shape[0], (list, tuple)):
            output_shape = [output_shape]
        output_dtype = parse_list_str(row_data.pop("output_dtype"))
        output_format_list = parse_list_str(row_data.pop("output_format"))
        assert len(output_format_list) == len(output_shape)

        output_tensors = []
        for idx in range(len(output_shape)):
            output_tensors.append(
                TensorDesc(
                    "output" + str(idx),
                    output_shape[idx],
                    output_dtype[idx],
                    None,
                    tensor_format=output_format_list[idx],
                    is_trans=None,
                )
            )
        view_shape = parse_list_str(row_data.pop("view_shape"))
        if isinstance(view_shape[0], (list, tuple)) and len(view_shape[0]) > 1:
            view_shape = view_shape[0]
        tile_shape = parse_list_str(row_data.pop("tile_shape"))
        params = {
            k: "" if pd.isna(v) or pd.isnull(v) else v for k, v in row_data.items()
        }
        # case_index, case_name, operation not need
        params.pop("case_index")
        params.pop("case_name")
        params.pop("operation")
        params["func_id"] = int(params.pop("func_id", "-1"))
        dims = row_data.get("dims", None)
        if dims is not None and not pd.isna(dims) and not pd.isnull(dims):
            params["dims"] = parse_list_str(row_data.get("dims"))
        first_dim = row_data.get("first_dim", None)
        if (
            first_dim is not None
            and not pd.isna(first_dim)
            and not pd.isnull(first_dim)
        ):
            params["first_dim"] = int(first_dim)
        second_dim = row_data.get("second_dim", None)
        if (
            second_dim is not None
            and not pd.isna(second_dim)
            and not pd.isnull(second_dim)
        ):
            params["second_dim"] = int(second_dim)
        count = row_data.get("count", None)
        if count is not None:
            params["count"] = parse_list_str(row_data.get("count"))
        islargest = row_data.get("islargest", None)
        if islargest is not None:
            params["islargest"] = [
                bool(x) for x in parse_list_str(row_data.get("islargest"))
            ]
        axis = row_data.get("axis", None)
        if axis is not None and not pd.isna(axis) and not pd.isnull(axis):
            params["axis"] = int(axis)

        matmulparam = MatmulParam(
            is_trans_list,
            input_format_list,
            output_format_list,
            row_data,
            output_dtype[0],
            is_k_split,
        )
        self.extend_matmul_param(matmulparam, params)

        return TestCaseDesc(
            row_data.get("case_index"),
            row_data.get("case_name"),
            row_data.get("operation"),
            input_tensors,
            output_tensors,
            view_shape,
            tile_shape,
            params,
        )

    def dump_to_json(self, write_to_json: bool = True):
        row_data = self.convert_row_data(self._case_data).dump_to_json()
        test_case = {"test_case": row_data}
        if write_to_json:
            json_file = f"{self._json_path}/{self._case_data['case_name']}.json"
            try:
                with open(json_file, "w", encoding="utf-8") as outfile:
                    json.dump(row_data, outfile, ensure_ascii=False, indent=4)
            except Exception as e:
                logging.error(
                    "Exception occur when writing %s, exception is %s.", json_file, e
                )
            test_case["json_file"] = json_file
        return test_case

    def str_to_bool(self, input_str: str):
        if input_str is None:
            return False
        input_str = str(input_str).strip().upper()
        logging.debug("caseindex: %s, input str: %s", self._case_index, input_str)
        return input_str in ("TRUE", "1")

    def extend_matmul_param(self, matmulparam: MatmulParam, params: dict):
        if matmulparam.row_data.get("operation") not in (
            "Matmul",
            "BatchMatmul",
            "MatmulVerify",
            "BatchMatmulVerify",
        ):
            return
        params["transA"] = matmulparam.trans_list[0]
        params["transB"] = matmulparam.trans_list[1]
        params["isAMatrixNz"] = matmulparam.input_format_list[0] == "NZ"
        params["isBMatrixNz"] = matmulparam.input_format_list[1] == "NZ"
        params["isCMatrixNz"] = matmulparam.output_format_list[0] == "NZ"
        output_dtype_str = matmulparam.output_dtype
        params["outDtype"] = str(output_dtype_str).strip()
        params["func_id"] = 0
        params["enableKSplit"] = matmulparam.is_k_split


def read_test_cases_from_csv(file_name: str, op: list) -> list:
    data_frame = pd.read_csv(file_name)
    if "operation" not in data_frame.columns:
        if not isinstance(op, list) or len(op) != 1:
            raise ValueError("Must set operation for test cases.")
        data_frame["operation"] = op[0]
    return [data_frame]


def read_test_cases_from_excel(file_name: str, op: list) -> list:
    data_frames = []
    file_handler = pd.ExcelFile(file_name)
    sheet_names = op if op is not None else list(file_handler.sheet_names)
    for sheet_name in sheet_names:
        df = pd.read_excel(file_handler, sheet_name=sheet_name)
        if "operation" not in df.columns:
            df["operation"] = sheet_name
        data_frames.append(df)
    file_handler.close()
    return data_frames


def load_test_cases(file_name: str, op: str) -> list:
    if not os.path.exists(file_name):
        print(f"Process File {file_name} failed, file not exist.")
        return None
    if op == "*" or op.lower() == "all":
        op = None
    else:
        op = [op]
    return (
        read_test_cases_from_csv(file_name, op)
        if file_name.endswith(".csv")
        else read_test_cases_from_excel(file_name, op)
    )


def clean_data_frame(
    data_frame: pd.DataFrame, op: str, start_index: int, end_index: int
) -> pd.DataFrame:
    if "case_index" not in data_frame.columns:
        data_frame.loc[:, "case_index"] = data_frame.index
    if start_index < 0:
        start_index = 0
    case_cnt = len(data_frame)
    if start_index >= case_cnt:
        print(f"The start index [{start_index}] exceeds the max index[{case_cnt}].")
        return False
    if end_index < 0 or end_index >= case_cnt:
        end_index = case_cnt

    data_frame = data_frame.iloc[start_index : end_index + 1]
    if "skip" in data_frame.columns:
        data_frame = data_frame.query(
            "skip != 1 and skip != '1' and skip != True and skip != 'TRUE'"
        )
    if "enable" in data_frame.columns:
        data_frame = data_frame.query(
            "(enable == 1 or enable == '1' or enable == True or enable == 'TRUE')"
        )
    data_frame.query(f"operation == '{op}'")
    return data_frame


def dump_data_frame_to_json(data_frames: list, json_path: str):
    data_frame = pd.concat(
        data_frames,
        ignore_index=True,
    )
    if len(data_frame) == 0:
        return {}
    if not os.path.exists(json_path):
        os.makedirs(json_path, exist_ok=True)
    test_cases = []
    for index, row_data in data_frame.iterrows():
        reader = TestDataReader(row_data["case_index"], row_data, json_path)
        case_info = reader.dump_to_json(False)
        case_info["test_case"]["index"] = index
        test_cases.append(case_info["test_case"])
    test_cases.sort(key=lambda x: (x["operation"], x["case_index"]))
    json_file = f"{json_path}/{test_cases[0]['operation']}_st_test_cases.json"
    row_data = {"test_cases": test_cases}
    with open(json_file, "w", encoding="utf-8") as outfile:
        json.dump(row_data, outfile, ensure_ascii=False, indent=4)
    return test_cases


def convert_data_to_json(
    file_name: str, op: str, index_range: list, json_path: str
) -> list:
    test_cases = load_test_cases(file_name, op)
    if test_cases is None or len(test_cases) == 0:
        return []

    test_cases = [
        clean_data_frame(data_frame, op, index_range[0], index_range[1])
        for data_frame in test_cases
    ]
    return dump_data_frame_to_json(test_cases, json_path)
