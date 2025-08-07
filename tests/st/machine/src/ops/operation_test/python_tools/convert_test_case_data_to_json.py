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
import sys
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
    def __init__(self, name: str, shape: list, dtype: str, data_range: list):
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._data_range = (
            None if data_range is None else DataRange(data_range[0], data_range[1])
        )

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
        input_shape = self.str_to_list(row_data.get("input_shape"))
        if not isinstance(input_shape[0], (list, tuple)):
            input_shape = [input_shape]
        input_dtype = self.str_to_list(row_data.get("input_dtype"))
        data_range = self.str_to_list(row_data.get("input_datarange"))
        if not isinstance(data_range[0], (list, tuple)):
            data_range = [data_range]
        assert len(input_shape) == len(input_dtype)
        assert len(input_shape) == len(data_range)
        input_tensors = []
        for idx in range(len(input_shape)):
            input_tensors.append(
                TensorData(
                    "input" + str(idx),
                    input_shape[idx],
                    input_dtype[idx],
                    data_range[idx],
                )
            )
        output_shape = self.str_to_list(row_data.get("output_shape"))
        if not isinstance(output_shape[0], (list, tuple)):
            output_shape = [output_shape]
        output_dtype = self.str_to_list(row_data.get("output_dtype"))
        output_tensors = []
        for idx in range(len(output_shape)):
            output_tensors.append(
                TensorData(
                    "output" + str(idx), output_shape[idx], output_dtype[idx], None
                )
            )
        view_shape = self.str_to_list(row_data.get("view_shape"))
        if isinstance(view_shape[0], (list, tuple)) and len(view_shape[0]) > 1:
            view_shape = view_shape[0]
        tile_shape = self.str_to_list(row_data.get("tile_shape"))
        params = {}
        mode = row_data.get(
            "mode", "0" if row_data.get("operation") == "Cast" else None
        )
        if mode is not None and not pd.isna(mode) and not pd.isnull(mode):
            params["mode"] = int(mode)
            dst_dtype = row_data.get("dst_dtype", output_dtype[0])
            params["dst_dtype"] = output_dtype[0] if dst_dtype is None else dst_dtype
        scalar = row_data.get("scalar", row_data.get("input_value", None))
        if scalar is not None and not pd.isna(scalar) and not pd.isnull(scalar):
            params["scalar"] = float(scalar)
            default_value = row_data.get("input_value_type", "fp32")
            params["scalar_type"] = row_data.get("scalar_type", default_value)
        params["func_id"] = int(row_data.get("func_id", "-1"))
        dims = row_data.get("dims", None)
        if dims is not None and not pd.isna(dims) and not pd.isnull(dims):
            params["dims"] = self.str_to_list(row_data.get("dims"))
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

    def dump_to_json(self):
        json_file = f"{self._json_path}/{self._case_data['operation']}_test_case_data_{self._case_index}.json"
        row_data = self.convert_row_data(self._case_data)
        try:
            with open(json_file, "w", encoding="utf-8") as outfile:
                json.dump(
                    row_data.dump_to_json(), outfile, ensure_ascii=False, indent=4
                )
        except Exception as e:
            logging.error(
                "Exception occur when writing %s, exception is %s.", json_file, e
            )

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


def handle_excel_sheet(
    sheet_data: pd.DataFrame, start_index: int, end_index: int, json_path: str
):
    if start_index < 0:
        start_index = 0
    case_cnt = len(sheet_data)
    if start_index >= case_cnt:
        print(f"The start index [{start_index}] exceeds the max index[{case_cnt}].")
        return False
    if end_index < 0 or end_index >= case_cnt:
        end_index = case_cnt

    dump_json_cnt = 0
    for index in range(start_index, end_index + 1):
        row_data = sheet_data.iloc[index]
        if (
            "skip" in sheet_data.columns
            and not pd.isnull(row_data["skip"])
            and not pd.isna(pd.isnull(row_data["skip"]))
            and bool(row_data["skip"])
        ):
            print(
                f"Test case[{index}] will be skipped due to column 'skip' set to {row_data['skip']}."
            )
            continue
        if "case_index" not in sheet_data.columns:
            row_data["case_index"] = index
        reader = TestDataReader(index, row_data, json_path)
        reader.dump_to_json()
        dump_json_cnt = dump_json_cnt + 1
    return dump_json_cnt > 0


def read_test_cases_from_csv(file_name: str, op: list) -> pd.DataFrame:
    data_frame = pd.read_csv(file_name)
    if "operation" not in data_frame.columns:
        assert len(op) == 1, "Not set operation for test cases."
        data_frame["operation"] = op[0]
    return data_frame.query(f"operation in {op}")


def read_test_cases_from_excel(file_name: str, op: list) -> pd.DataFrame:
    data_frame = None
    try:
        file_handler = pd.ExcelFile(file_name)
        sheet_names = op if op is not None else list(file_handler.sheet_names)
        for sheet_name in sheet_names:
            df = pd.read_excel(file_handler, sheet_name=sheet_name)
            if "operation" not in df.columns:
                df["operation"] = sheet_name
            data_frame = (
                df
                if data_frame is None
                else pd.concat(
                    [data_frame, df],
                    ignore_index=True,
                )
            )
        return data_frame
    except Exception as e:
        raise e
    finally:
        file_handler.close()


def load_test_cases(file_name: str, op: str) -> pd.DataFrame:
    if not os.path.exists(file_name):
        print(f"Process File {file_name} failed, file not exist.")
        return None
    if op == "*" or op.lower() == "all":
        op = None
    else:
        op = [op]
    try:
        return (
            read_test_cases_from_csv(file_name, op)
            if file_name.endswith(".csv")
            else read_test_cases_from_excel(file_name, op)
        )
    except Exception as e:
        logging.error("Try to process %s fail, exception is %s.", file_name, e)
    return None


def main(file_name: str, op: str, index_range: list, json_path: str) -> bool:
    test_cases = load_test_cases(file_name, op)
    if test_cases is None or len(test_cases) == 0:
        return False

    return handle_excel_sheet(test_cases, index_range[0], index_range[1], json_path)


if __name__ == "__main__":
    exit(
        0
        if main(
            sys.argv[1], sys.argv[2], [int(sys.argv[3]), int(sys.argv[4])], sys.argv[5]
        )
        else 1
    )
