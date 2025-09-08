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
import pandas as pd


class DataRange:
    def __init__(self, min, max):
        self._min = min
        self._max = max

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def dump_to_json(self) -> dict:
        return {"min": self._min, "max": self._max}


class TensorData:
    def __init__(
        self,
        name: str,
        shape: list,
        dtype: str,
        data_range: list,
        tensor_format: str = None,
        is_trans: bool = None,
    ):
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._data_range = (
            None if data_range is None else DataRange(data_range[0], data_range[1])
        )
        self._format = tensor_format
        self._is_trans = is_trans

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> list:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def data_range(self) -> DataRange:
        return self._data_range

    def dump_to_json(self) -> dict:
        json_content = {"name": self._name, "shape": self._shape, "dtype": self._dtype}
        if self._data_range is not None:
            json_content["data_range"] = self._data_range.dump_to_json()
        if self._format is not None:
            json_content["format"] = self._format
        if self._is_trans is not None:
            json_content["is_trans"] = self._is_trans
        return json_content


class TestCaseData:
    def __init__(
        self,
        case_index: str,
        case_name: str,
        operation: str,
        input_tensors: list,
        output_tensors: list,
        view_shape: list,
        tile_shape: list,
        params: dict,
    ):
        self._case_index = case_index
        self._case_name = case_name
        self._operation = operation
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors
        self._view_shape = view_shape
        self._tile_shape = tile_shape
        self._params = params

    @property
    def index(self) -> str:
        return self._case_index

    @property
    def name(self) -> str:
        return self._case_name

    @property
    def operation(self) -> str:
        return self._operation

    @property
    def input_tensors(self) -> list:
        return self._input_tensors

    @property
    def output_tensors(self) -> list:
        return self._output_tensors

    @property
    def view_shape(self) -> list:
        return self._view_shape

    @property
    def tile_shape(self) -> list:
        return self._tile_shape

    @property
    def params(self) -> dict:
        return self._params

    def dump_to_json(self) -> dict:
        return {
            "case_index": self._case_index,
            "case_name": self._case_name,
            "operation": self._operation,
            "input_tensors": list(map(lambda x: x.dump_to_json(), self._input_tensors)),
            "output_tensors": list(
                map(lambda x: x.dump_to_json(), self._output_tensors)
            ),
            "view_shape": self._view_shape,
            "tile_shape": self._tile_shape,
            "params": self._params,
        }


class TestDataReader:
    def __init__(self, case_index: int, case_data, json_path: str):
        self._case_index = case_index
        self._case_data = case_data
        self._json_path = json_path

    def convert_row_data(self, row):
        row_data = row.to_dict()
        input_shape = self.str_to_list(row_data.pop("input_shape"))
        if not isinstance(input_shape[0], (list, tuple)):
            input_shape = [input_shape]
        input_dtype = self.str_to_list(row_data.pop("input_dtype"))
        data_range = self.str_to_list(row_data.pop("input_datarange"))
        if not isinstance(data_range[0], (list, tuple)):
            data_range = [data_range]
        assert len(input_shape) == len(input_dtype)
        assert len(input_shape) == len(data_range)

        input_format_list = self.str_to_list(row_data.pop("input_format"))
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

        input_tensors = []
        for idx in range(len(input_shape)):
            input_tensors.append(
                TensorData(
                    "input" + str(idx),
                    input_shape[idx],
                    input_dtype[idx],
                    data_range[idx],
                    tensor_format=input_format_list[idx],
                    is_trans=is_trans_list[idx],
                )
            )
        output_shape = self.str_to_list(row_data.pop("output_shape"))
        if not isinstance(output_shape[0], (list, tuple)):
            output_shape = [output_shape]
        output_dtype = self.str_to_list(row_data.pop("output_dtype"))

        output_format_list = self.str_to_list(row_data.pop("output_format"))
        assert len(output_format_list) == len(output_shape)

        output_tensors = []
        for idx in range(len(output_shape)):
            output_tensors.append(
                TensorData(
                    "output" + str(idx),
                    output_shape[idx],
                    output_dtype[idx],
                    None,
                    tensor_format=output_format_list[idx],
                    is_trans=None,
                )
            )
        view_shape = self.str_to_list(row_data.pop("view_shape"))
        if isinstance(view_shape[0], (list, tuple)) and len(view_shape[0]) > 1:
            view_shape = view_shape[0]
        tile_shape = self.str_to_list(row_data.pop("tile_shape"))
        params = row_data.copy()
        # case_index, case_name, operation not need
        params.pop("case_index")
        params.pop("case_name")
        params.pop("operation")
        params["func_id"] = int(params.pop("func_id", "-1"))
        dims = row_data.get("dims", None)
        if dims is not None and not pd.isna(dims) and not pd.isnull(dims):
            params["dims"] = self.str_to_list(row_data.get("dims"))
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
            params["count"] = self.str_to_list(row_data.get("count"))
        islargest = row_data.get("islargest", None)
        if islargest is not None:
            params["islargest"] = [
                bool(x) for x in self.str_to_list(row_data.get("islargest"))
            ]
        axis = row_data.get("axis", None)
        if axis is not None and not pd.isna(axis) and not pd.isnull(axis):
            params["axis"] = int(axis)

        self.extend_matmul_param(
            params, is_trans_list, input_format_list, output_format_list, row_data
        )

        return TestCaseData(
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

    def str_to_list(self, input: str):
        ret_list = []
        input = input.replace(" ", "")
        if input.startswith("{") or input.startswith("["):
            input = input[1:-1]
        element_split_ident = " "
        if "{" in input:
            element_split_ident = "},{"
        if "[" in input:
            element_split_ident = "],["
        if element_split_ident in input:
            for sub_str in input.split(element_split_ident):
                ret_list.append(self.str_to_list(sub_str))
        else:
            for sub_str in input.split(","):
                is_num = sub_str.isdecimal() or (
                    (sub_str.startswith("-") or sub_str.startswith("+"))
                    and sub_str[1:].isdecimal()
                )
                ret_list.append(int(sub_str) if is_num else sub_str)
        return ret_list

    def str_to_bool(self, input_str: str):
        if input_str is None:
            return False
        input_str = str(input_str).strip().upper()
        logging.debug("caseindex: %s, input str: %s", self._case_index, input_str)
        return input_str in ("TRUE", "1")

    def extend_matmul_param(
        self,
        params: dict,
        trans_list: list,
        input_format_list: list,
        output_format_list: list,
        row_data: dict,
    ):
        if row_data.get("operation") not in (
            "Matmul",
            "BatchMatmul",
            "MatmulVerify",
            "BatchMatmulVerify",
        ):
            return
        params["transA"] = trans_list[0]
        params["transB"] = trans_list[1]
        params["isAMatrixNz"] = input_format_list[0] == "NZ"
        params["isBMatrixNz"] = input_format_list[1] == "NZ"
        params["isCMatrixNz"] = output_format_list[0] == "NZ"
        output_dtype_str = self.str_to_list(row_data.get("output_dtype"))[0]
        params["outDtype"] = str(output_dtype_str).strip()
        params["func_id"] = 0


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


def dump_data_frame_to_json(data_frames: list, json_path: str, json_only: bool):
    data_frame = pd.concat(
        data_frames,
        ignore_index=True,
    )
    if len(data_frame) == 0:
        return {}
    if not os.path.exists(json_path):
        os.makedirs(json_path, exist_ok=True)
    test_cases = []
    test_case_info_list = []
    for index, row_data in data_frame.iterrows():
        reader = TestDataReader(row_data["case_index"], row_data, json_path)
        case_info = reader.dump_to_json(not json_only)
        test_case_info_list.append(
            {
                "index": index,
                "case_index": case_info["test_case"]["case_index"],
                "case_name": case_info["test_case"]["case_name"],
                "operation": case_info["test_case"]["operation"],
                "json_file": None if json_only else case_info["json_file"],
            }
        )
        if json_only:
            test_cases.append(case_info["test_case"])
    test_cases.sort(key=lambda x: (x["operation"], x["case_index"]))
    test_case_info_list.sort(key=lambda x: (x["operation"], x["case_index"]))
    if json_only:
        json_file = f"{json_path}/{test_cases[0]['operation']}_st_test_cases.json"
        row_data = {"test_cases": test_cases}
        with open(json_file, "w", encoding="utf-8") as outfile:
            json.dump(row_data, outfile, ensure_ascii=False, indent=4)
    return test_case_info_list


def convert_data_to_json(
    file_name: str, op: str, index_range: list, json_path: str, json_only: bool = False
) -> list:
    test_cases = load_test_cases(file_name, op)
    if test_cases is None or len(test_cases) == 0:
        return []

    test_cases = [
        clean_data_frame(data_frame, op, index_range[0], index_range[1])
        for data_frame in test_cases
    ]
    return dump_data_frame_to_json(test_cases, json_path, json_only)
