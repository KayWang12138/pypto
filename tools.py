# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import os 
import re
import json 
import pypto
import torch 
import torch_npu 
import subprocess
torch_npu.npu.config.allow_internal_format = True
from pathlib import Path
import numpy as np
import tensorflow as tf 
from openpyxl import load_workbook
from pandas import DataFrame


def get_precision_np_dtype(type_str):
    type_dict = {
        "fp16": np.float32, "float16": np.float32,
        "bf16": np.float32, "bfloat16": np.float32,
        "fp32": np.float64, "float32": np.float64,
        "int8": np.int8, "int32": np.int32,
        "int64": np.int64, "int16": np.int16,
        "bool": np.bool_
    }
    assert type_str in type_dict, f"type_str {type_str} not in high precision type_dict, please check"
    return type_dict[type_str]


def get_upper_dtype(type_str):
    if type_str in ["fp16", "bf16"]:
        return "fp32"
    else:
        return type_str 
    

def get_np_dtype(type_str):
    type_dict = {
        'fp64': np.float64, 'fp32': np.float32, 'fp16': np.float16,
        'int64': np.int64, 'int32': np.int32, 'int16': np.int16, 'int8': np.int8,
        'uint64': np.uint64, 'uint32': np.uint32, 'uint16': np.uint16, 'uint8': np.uint8,
        'bool': np.bool_, 'complex64': np.complex64, 'complex128': np.complex128,
        'complex32': np.float16,
        'bf16': tf.bfloat16.as_numpy_dtype,
        'bfloat16': tf.bfloat16.as_numpy_dtype,
        'float4_e2m1': np.uint8,
        'float4_e1m2': np.uint8,
        'float8_e8m0': np.uint8,
        'hifloat8': np.uint8,
        'fp4_e1m2': np.uint8,
        "fp4_e2m1": np.uint8,
        "fp8_e8m0": np.uint8,
        "float32": np.float32,
        "float16": np.float16,
        "qint8": np.int8,
        "qint32": np.int32,
        "quint8": np.uint8,
        "qint16": np.int16,
        "uint1": np.uint8,
        "quint16": np.uint16,
        "fp4_e2m1_as_fp32": np.float32
    }
    assert type_str in type_dict, f"type_str {type_str} not in numpy type_dict, please check"
    return type_dict[type_str]


def get_torch_dtype(type_str):
    type_dict = {
        "fp32": torch.float32, "float32": torch.float32, 
        "fp64": torch.float64, "float64": torch.float64,
        "fp16": torch.float16, "float16": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "int8": torch.int8, "int32": torch.int32,
        "int64": torch.int64, "int16": torch.int16,
        "bool": torch.bool,"uint8": torch.uint8
    }
    assert type_str in type_dict, f"type_str {type_str} not in torch type_dict, please check"
    return type_dict[type_str]


def get_pypto_dtype(type_str):
    type_dict = {
        "fp32": pypto.DT_FP32, "float32": pypto.DT_FP32, 
        "fp16": pypto.DT_FP16, "float16": pypto.DT_FP16,
        "bf16": pypto.DT_BF16, "bfloat16": pypto.DT_BF16,
        "int8": pypto.DT_INT8, "int32": pypto.DT_INT32,
        "int64": pypto.DT_INT64, "int16": pypto.DT_INT16,
        "bool": pypto.DT_BOOL
    }
    assert type_str in type_dict, f"type_str {type_str} not in pypto type_dict, please check"
    return type_dict[type_str]


def convert_torch_to_numpy(torch_tensor):
    torch_tensor = torch_tensor.cpu()
    if torch_tensor.dtype == torch.bfloat16:
        numpy_tensor = torch_tensor.to(torch.float32).numpy().astype(tf.bfloat16.as_numpy_dtype)
    else:
        numpy_tensor = torch_tensor.numpy()
    return numpy_tensor 

def convert_numpy_to_torch(input_data, dtype):
    if dtype in ["bf16", "bfloat16"]:
        input_tensor = torch.from_numpy(input_data.astype(np.float32)).to(torch.bfloat16)
    else:
        input_tensor = torch.from_numpy(input_data)
    return input_tensor
    
#为了解决in out用统一tensor的问题增加规避措施，待主线实现后和开发代码同步
def convert_torch_tensor(tensor_dict, dynamic_axis_dict, name_prefix):
    dynamic_count = 0
    pypto_tensors = []
    for name, tensor in tensor_dict.items():
        if name in dynamic_axis_dict.keys():
            dynamic_axis = dynamic_axis_dict[name]
            pypto_tensors.append(pypto.from_torch(tensor, name_prefix + name, dynamic_axis=dynamic_axis))
            dynamic_count += 1
        else:
            pypto_tensors.append(pypto.from_torch(tensor, name_prefix + name))
    assert dynamic_count == len(dynamic_axis_dict)
    return pypto_tensors


# 将csv格式的用例结果写入xlsx
def write_xlsx_result(case_file_path, result_list, report_path, sheet_idx):
    wb = load_workbook(case_file_path)
    ws = wb.worksheets[sheet_idx]

    new_col = ws.max_column + 1
    new_col_begin = new_col
    for new_header in ['precision_result', 'precision_result_triple', 'result']:
        ws.cell(row=1, column=new_col, value=new_header)
        new_col += 1

    for case_result in result_list:
        case_idx = case_result[0]
        case_name = case_result[1]
        result = "ERROR" if case_result[2][1] is None else case_result[2][1][0]
        precision_result = str(case_result[2][0])
        precision_result_triple = str(case_result[2][1])
        for row_idx, row in enumerate(ws.iter_rows(min_row=1, min_col=1, max_col=1), start=1):
            cell = row[0]
            if row_idx == case_idx and cell.value == case_name:
                ws.cell(row=cell.row, column=new_col_begin, value=precision_result)
                ws.cell(row=cell.row, column=new_col_begin+1, value=precision_result_triple)
                ws.cell(row=cell.row, column=new_col_begin+2, value=result)
                break
    wb.save(report_path)


# 根据关键词搜索日志, 对结果进行初步分析
def analyse_log(log_path, case_name):
    pass_output = os.path.join(log_path, "pass_output")
    plog_output = os.path.join(log_path, "plog_output")
    print_log = os.path.join(log_path, f"{case_name}.log")

    # 日志关键词列表
    error_key_dict = {
        "is_sync_fail": {
            "key": [
                f"cat {print_log} | grep 'sync stream failed'",
                f"cat {print_log} | grep 'stream synchronize failed'"],
            "result": False
        },
        "is_core_dump": {
            "key": f"cat {print_log} | grep 'Segmentation fault'",
            "result": False
        },
        "pass_error_info": {
            "key": f"grep -rns -i 'run pass' {pass_output} | grep fail",
            "result": None
        },
        "plog_error_info": {
            "key": f"grep -rns ERROR {plog_output} | head -n 10",
            "result": None
        }
    }

    for error_type in error_key_dict:
        error_key = error_key_dict[error_type]["key"]
        if isinstance(error_key, list):
            for key in error_key:
                result = subprocess.run(key, shell=True, capture_output=True, text=True).stdout
                if result != "":
                    error_key_dict[error_type]["result"] = True
        else:
            result = subprocess.run(error_key, shell=True, capture_output=True, text=True).stdout
            if result != "":
                if error_type == "core_dump":
                    error_key_dict[error_type]["result"] = True
                else:
                    error_key_dict[error_type]["result"] = result

    return error_key_dict


# 将json格式的用例结果写入xlsx
def write_xlsx_result_from_json(result_list, report_path, output_dir):
    results = {
        "case_idx": [],
        "case_name": [],
        "result": []
    }
    resultlen = 0
    for case_result in result_list:
        if case_result[2] == "PASS" or case_result[2] == "FAILED":
            resultlen = len(case_result[3])
            break
    for case_result in result_list:
        print(*case_result, sep=', ')
        if len(case_result) == 4:
            case_idx, case_name, result, precision_result = case_result
            results["case_idx"].append(case_idx)
            results["case_name"].append(case_name)
            results["result"].append(result)
            for output_index in range(len(precision_result)):
                key_name = "output_{}".format(output_index)
                if key_name not in results.keys():
                    results.update({key_name: [precision_result[output_index]]})
                else:
                    results[key_name].append(precision_result[output_index])
        else:
            case_idx, case_name = case_result[0], case_result[1]
            results["case_idx"].append(case_idx)
            results["case_name"].append(case_name)
            results["result"].append("ERROR")
            for output_index in range(resultlen):
                key_name = "output_{}".format(output_index)
                if key_name not in results.keys():
                    results.update({key_name: [None]})
                else:
                    results[key_name].append(None)
        log_path = os.path.join(output_dir, str(case_idx))
        error_key_dict = analyse_log(log_path, case_name)
        for error_type in error_key_dict:
            if error_type not in results.keys():
                results.update({error_type: [error_key_dict[error_type]["result"]]})
            else:
                results[error_type].append(error_key_dict[error_type]["result"])

    data_frame = DataFrame(results)
    data_frame.to_excel(report_path, index=False)


# 将case result写入result.txt
def write_txt_result(result_list, txt_path):
    with open(txt_path, "w+") as f:
        for case_result in result_list:
            if len(case_result) == 3:
                result = "ERROR" if case_result[2][1] is None else case_result[2][1][0]
                precision_result = case_result[2][0]
                precision_result_triple = case_result[2][1]
                case_idx = case_result[0]
                case_name = case_result[1]
                f.writelines(f"case_idx: {case_idx}, {case_name}: {result}, precision_result: {precision_result}, precision_result_triple: {precision_result_triple}\n")
            else:
                case_idx, case_name, result, precision_result = case_result
                f.writelines(f"case_idx: {case_idx}, {case_name}: {result}, precision_result: {precision_result}\n")


def re_find_csv_case_keyword(content, key_dict):
    try:
        try:
            precision_result = eval(re.findall(r'precision_result: (.*?);', content)[0])
            precision_result_triple = eval(re.findall(r'precision_result_triple: (.*?);', content)[0])
        except NameError:
            precision_result = re.findall(r'precision_result: (.*?);', content)[0]
            precision_result_triple = re.findall(r'precision_result_triple: (.*?);', content)[0]
            for key in key_dict.keys():
                if key in precision_result:
                    precision_result = precision_result.replace(key, key_dict[key])
                if key in precision_result_triple:
                    precision_result_triple = precision_result_triple.replace(key, key_dict[key])

            precision_result = eval(precision_result)
            precision_result_triple = eval(precision_result_triple)

        if precision_result == "None":
            precision_result = None
            precision_result_triple = None
    except IndexError:
        precision_result = None
        precision_result_triple = None
    
    return precision_result, precision_result_triple


def re_find_json_case_keyword(content, case_name, key_dict):
    try:
        result = re.findall(rf'{case_name} (.*?);', content)[0]
        try:
            precision_result = eval(re.findall(r'precision_result: (.*?);', content)[0])
        except NameError:
            precision_result = re.findall(r'precision_result: (.*?);', content)[0]
            for key in key_dict.keys():
                if key in precision_result:
                    precision_result = precision_result.replace(key, key_dict[key])
            precision_result = eval(precision_result)

        if precision_result == "None":
            precision_result = None
    except IndexError:
        result = "ERROR"
        precision_result = None

    return result, precision_result


# 用例还未执行完，生成result.txt和xlsx查看已执行部分的用例结果
def generate_report(output_dir, op_type, case_file_path, sheet_idx, case_from_json=False):
    log_list = []
    key_dict = {
        "nan": "float('nan')",
        "inf": "float('inf')"
    }

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.startswith(op_type) and file.endswith(".log"):
                log_path = os.path.join(root, file)
                if not ("pass_output" in log_path or "plog_output" in log_path):
                    log_list.append(log_path)

    case_result_list = []
    for log in log_list:
        with open(log, "r") as f:
            content = f.read()
        case_name = list(Path(log).parts)[-1].rsplit(".log", 1)[0]
        case_idx = list(Path(log).parts)[-2]

        if not case_from_json:
            # csv入口用例
            precision_result, precision_result_triple = re_find_csv_case_keyword(content, key_dict)
            case_result_list.append([int(case_idx), case_name, (precision_result, precision_result_triple)])
            # 将结果写入csv
            report_path = os.path.join(output_dir, f"{op_type}_part_result.xlsx")
            write_xlsx_result(case_file_path, case_result_list, report_path, sheet_idx)
        else:
            # json入口用例
            result, precision_result = re_find_json_case_keyword(content, case_name, key_dict)
            case_result_list.append([int(case_idx), case_name, result, precision_result])
            # 将结果写入csv
            report_path = os.path.join(output_dir, f"{op_type}_part_result.xlsx")
            write_xlsx_result_from_json(case_result_list, report_path)

    # 将结果写入txt
    txt_path = os.path.join(output_dir, f"{op_type}_part_result.txt")
    write_txt_result(case_result_list, txt_path)
    

def broadcast_shape_offset(input_shape, input_offsets, input_valid_shape, broadcast_dims):
    input_shape_new = []
    input_offsets_new = []
    input_valid_shape_new = []  
    for dim_index, (shape, input_offset, valid_shape) in enumerate(zip(input_shape, input_offsets, input_valid_shape)):
        if dim_index not in broadcast_dims:
            input_shape_new.append(shape)
            input_offsets_new.append(input_offset)
            input_valid_shape_new.append(valid_shape)
        else:
            input_shape_new.append(1)
            input_offsets_new.append(0)
            input_valid_shape_new.append(1)
    return input_shape_new, input_offsets_new, input_valid_shape_new


def generate_broadcast_info(input_shape_list, output_shape):
    assert isinstance(input_shape_list, list), "input_shape_list is invalid, pls check"
    assert isinstance(input_shape_list[0], list), "input_shape_list is invalid, pls check"
    assert isinstance(output_shape, list), "output_shape is invalid, pls check"
    broadcast_info = []
    for input_shape in input_shape_list:
        cur_bcast_info = []
        for index, (inp_shape, oup_shape) in enumerate(zip(input_shape, output_shape)):
            if inp_shape != oup_shape:
                cur_bcast_info.append(index)
        broadcast_info.append(cur_bcast_info)
    return broadcast_info


def get_broadcast_offsets(origin_offsets, broadcast_dims):
    new_offsets = []
    for dim_index, origin_offset in enumerate(origin_offsets):
        if dim_index in broadcast_dims:
            new_offsets.append([0, 1])
        else:
            new_offsets.append(origin_offset)
    return new_offsets


def get_tensor_broadcast(input_tensor_1: pypto.Tensor, input_tensor_2: pypto.Tensor):
    for i, (dim1, dim2) in enumerate(zip(input_tensor_1.shape, input_tensor_2.shape)):
        if dim1 != dim2:
            if dim1 == 1 and dim2 != 1:
                return 1, i
            elif dim1 != 1 and dim2 == 1:
                return 2, i
            else:
                raise ValueError(f"Cannot broadcast tensors: dimension mismatch at axis {i}: {dim1} vs dim2")
    return None, None


def get_tensor_broadcast1(input_tensor_1: pypto.Tensor, input_tensor_2: pypto.Tensor, input_tensor_3: pypto.Tensor):
    for i, (dim1, dim2, dim3) in enumerate(zip(input_tensor_1.shape, input_tensor_2.shape, input_tensor_3.shape)):
        print(f"Debug: i = {i}")

        if dim1 != dim2 or dim2 != dim3:
            if dim1 == 1 and dim2 != 1 and dim3 != 1:
                return 1, i
            elif dim1 != 1 and dim2 == 1 and dim3 != 1:
                return 2, i
            elif dim1 != 1 and dim2 != 1 and dim3 == 1:
                return 3, i
            else:
                raise ValueError(f"Cannot broadcast tensors: dimension mismatch at axis {i}: {dim1} vs dim2")

    return None, None


def read_inputs_from_bin_dir(case_info, bin_dir, device="npu"):
    case_name = case_info["case_name"]

    inputs_list = []
    for index, (input_shape, input_dtype) in enumerate(zip(case_info["shape_input"], case_info["dtype_input"])):
        input_bin_path = os.path.join(bin_dir, f"{case_name}_input_{index}.bin")
        input_torch = load_file(input_bin_path, input_shape, input_dtype)
        input_torch = input_torch.to(device)
        inputs_list.append(input_torch)
    return inputs_list


def generate_outputs(case_info, device="npu"):
    outputs_list = []
    for output_shape, output_dtype in zip(case_info["shape_output"], case_info["dtype_output"]):
        output_tensor = torch.empty(output_shape, dtype=get_torch_dtype(output_dtype)).to(device)
        outputs_list.append(output_tensor)
    return outputs_list


def convert_specific_NZ(inputs_list, case_info):
    for input_index in range(len(inputs_list)):
        if "NZ" not in case_info["format_input"][input_index]:
            continue
        cur_tensor = inputs_list[input_index]
        if cur_tensor.dtype == torch.int8:
            scale = 32
        elif cur_tensor.dtype in [torch.float16, torch.bfloat16]:
            scale = 16
        elif cur_tensor.dtype in [torch.float32]:
            scale = 8
        else:
            raise NotImplementedError(f"{cur_tensor.dtype} does not support NZ format")
        cur_shape = cur_tensor.shape
        cur_tensor_nz = cur_tensor.reshape(cur_shape[0] // 16, 16, cur_shape[1] // scale, scale).permute(2, 0, 1, 3)
        cur_tensor_nz = torch_npu.npu_format_cast(cur_tensor_nz.contiguous(), 29)
        inputs_list[input_index] = cur_tensor_nz


def convert_NZ(inputs_list, case_info):
    for input_index in range(len(inputs_list)):
        if "NZ" not in case_info["format_input"][input_index]:
            continue
        inputs_list[input_index] = torch_npu.npu_format_cast(inputs_list[input_index].contiguous(), 29)


def read_tensor(input_path, dtype):
    input_tensor = np.fromfile(input_path, dtype=get_np_dtype(dtype))
    return input_tensor


def dump_tensor(input_tensor, input_path):
    input_tensor = input_tensor.cpu()
    if input_tensor.dtype == torch.bfloat16:
        input_tensor_np = input_tensor.to(torch.float32).numpy().astype(tf.bfloat16.as_numpy_dtype)
    else:
        input_tensor_np = input_tensor.numpy()
    input_tensor_np.tofile(input_path)

#读取bin文件返回torch的tensor
def load_file(file_path, shape, dtype):
    data = np.fromfile(file_path, get_np_dtype(dtype))
    data = np.reshape(data, shape)
    # numpy不支持bfloat16类型的直接转换
    if dtype == "bf16":
        data_torch = torch.from_numpy(data.astype(np.float32)).to(torch.bfloat16)
    else:
        data_torch = torch.from_numpy(data)
    return data_torch

# 生成json格式的case，用于operation 融合算子
def generate_cases(input_dir, output_path):
    cases_dir = os.path.join(input_dir, "cases")
    inputs_dir = os.path.join(input_dir, "inputs")
    cases_list = []
    for case_index, case_file in enumerate(sorted(os.listdir(cases_dir))):
        if not case_file.endswith(".cs"):
            continue
        case_name = case_file.replace(".cs", "")
        cs_path = os.path.join(cases_dir, case_file)
        bin_path = os.path.join(inputs_dir, case_name)
        cases_list.append(
            {
                "case_idx": case_index,
                "case_name": case_name,
                "op_type": "MlaIndexerProlog",
                "cs_path": cs_path,
                "bin_path": bin_path
            }
        )
    with open(output_path, "w") as f:
        json.dump(cases_list, f, indent=4, ensure_ascii=False)


# 目录下文件生成软链接
def soft_link_origin_data(base_dir, case_output_dir):
    for file_name in os.listdir(base_dir):
        src_path = os.path.realpath(os.path.join(base_dir, file_name))
        dst_path = os.path.realpath(os.path.join(case_output_dir, file_name))
        os.symlink(src_path, dst_path)


if __name__ == "__main__":
    # csv格式的用例
    sheet_idx = -1
    op_type = "Matmul1"
    output_dir = "/home/l30036931/PyPTO_Test/output/202512011819201"
    case_file_path = "/home/l30036931/PyPTO_Test/testcase/Case_Matmul.xlsx"
    generate_report(output_dir, op_type, case_file_path, sheet_idx)

    # json格式的用例
    # op_type = "pypto_indexer_prolog"
    # output_dir = "/home/l30036931/mmmm/PyPTO_Test/output/20251226164433"
    # generate_report(output_dir, op_type, "", "", True)