#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import os
import csv
import pandas as pd
import numpy as np
import networkx  as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ml_dtypes, json
import argparse
import sys
import os
import torch
from tensor_diff import check_isclose, print_isclose_info
from run_float_diff import fix_input_and_compute


verify_path_pass1 = ""
verify_path_pass2 = ""

atol=1e-3
rtol=1e-3
is_sort = False
topk=50

dtype_dict = {
    "DT_BF16": ml_dtypes.bfloat16,
    "DT_FP32": np.float32,
    "DT_FP16": np.float16,
    "DT_INT32": np.int32,
    "DT_INT8": np.int8,
    "DT_INT64": np.int64,
    "DT_INT16": np.int16
}

torch_dtype_dict = {
    ml_dtypes.bfloat16: torch.bfloat16,
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int32: torch.int32,
    np.int8: torch.int8,
    np.int64: torch.int64,
    np.int16: torch.int16
}

def is_contain(a_offset, b_offset, a_shape, b_shape):
    lenth = len(a_offset)
    for i in range(lenth):
        if (a_offset[i] < b_offset[i]) or ((a_offset[i] + a_shape[i]) > (b_offset[i] + b_shape[i])):
            return False
    return True


def compare(a, b):
    abs_diff = np.abs(a-b)
    avg = np.abs(a+b) / 2
    rel_diff = abs_diff / avg
    # print(rel_diff)
    condition = (rel_diff>0.001) & (abs_diff>0.0001)
    # condition = abs_diff>0.000001
    cond = np.where(condition)
    # print(cond)
    print(f"err size {cond[0].size}, {cond[0].shape}")
    count = 0
    for i in cond[0]:
        if count > 50:
            break
        print(f"index: {i},  {a[i]},  {b[i]},  {abs_diff[i]}, {rel_diff[i]}")
        count += 1
    # if len(cond[0]) > 0:
    if cond[0].size > a.size * 0.0001:
        return False
    return True

def compare_data(a, b):
    dtype = a["outputDtype"]
    if a["passName"] == "tensor_graph":
        f_a = verify_path_pass1 + a["passName"]+ "/" + a["outputTensor"]
    else :
        f_a = verify_path_pass1 + a["passName"]+ "/" +a["verifyType"] + "/" + a["outputTensor"]
    
    if b["passName"] == "tensor_graph":
        f_b = verify_path_pass2 + b["passName"]+ "/" + b["outputTensor"]
    else :
        f_b = verify_path_pass2 + b["passName"]+ "/" +b["verifyType"] + "/" + b["outputTensor"]
    
    if not os.path.exists(f_a) or not os.path.exists(f_b):
        print("有些文件不存在，直接跳过")
        return True
    a_offset = json.loads(a["tensorOffset"])
    b_offset = json.loads(b["tensorOffset"])
    a_shape = json.loads(a["outputValidShape"])
    b_shape = json.loads(b["outputValidShape"])
    np_dtype_a = dtype_dict.get(dtype, "unknown")
    data_a = np.fromfile(f_a, np_dtype_a)
    data_b = np.fromfile(f_b, np_dtype_a)
    data_b = data_b.reshape(b_shape)
    slices = []
    print("------" * 10)
    print(f'functionName : {a["verifyType"]}')
    print(f'rawTensorMagic : {a["rawTensorMagic"]},    path : {a["loopInfo"]}')
    print(f'line : {a["No."]} , 数据a的shape : {a_shape},  offset : {a_offset}, dtype : {a["outputDtype"]}')
    print(f'line : {b["No."]} , 数据b的shape : {b_shape},  offset : {b_offset}, dtype : {b["outputDtype"]}')
    for dim in range(data_b.ndim):
        start = a_offset[dim] - b_offset[dim]
        stop = start + a_shape[dim]
        slices.append(slice(start, stop))
    b_slice = data_b[tuple(slices)]
    t_dtype_a = torch_dtype_dict.get(np_dtype_a)
    
    if dtype == "DT_BF16" :
        tensor_a = torch.frombuffer(memoryview(data_a.tobytes()),dtype=t_dtype_a).reshape(a_shape)
        tensor_b = torch.frombuffer(memoryview(b_slice.tobytes()),dtype=t_dtype_a).reshape(a_shape)
    else:
        tensor_a = torch.from_numpy(data_a).to(dtype=t_dtype_a)
        tensor_b = torch.from_numpy(b_slice).to(dtype=t_dtype_a)
    result_is_close, result_reason_str, result_info = check_isclose(tensor_a, tensor_b, rtol, atol, calc_dtype=torch.float32, is_ignore_bothzero=True, is_detail=True)
    if not result_is_close:
        print_isclose_info(result_is_close, result_reason_str, result_info, topk)
        print("数据对比失败")
        fix_input_and_compute(data_a, b_slice, [data_a.dtype, b_slice.dtype], is_sort)
        return False
    print("数据对比通过")
    return True

def loop_compare(pass_a, pass_b, df_loop, rawTensorList):
    df_a = df_loop[df_loop["passName"].str.contains(pass_a)]
    df_b = df_loop[df_loop["passName"].str.contains(pass_b)]

    values_a = df_a["rawTensorMagic"].dropna().unique()
    values_b = df_b["rawTensorMagic"].dropna().unique()
    common_values_list = list(set(values_a) & set(values_b))
    if len(rawTensorList) != 0:
        common_values_list = rawTensorList
    for raw_magic in common_values_list:
        a = df_a[df_a["rawTensorMagic"] == raw_magic].to_dict(orient='records')
        b = df_b[df_b["rawTensorMagic"] == raw_magic].to_dict(orient='records')
        if len(a) < len(b):
            a, b = b, a #遍历rawtensor时，把数据多的一方全遍历完。
        for ai in a:
            for bi in b:
                a_offset = json.loads(ai["tensorOffset"])
                b_offset = json.loads(bi["tensorOffset"])
                a_shape = json.loads(ai["outputValidShape"])
                b_shape = json.loads(bi["outputValidShape"])
                if is_contain(a_offset, b_offset, a_shape, b_shape):
                    is_right = compare_data(ai, bi)
                    if is_right:
                        continue
                    else:
                        return False
    return True
                    
def pass_compare(pass_a, pass_b, path, rawTensorList):
    csv_path = os.path.join(verify_path_pass1, "verify_result.csv")
    df = pd.read_csv(csv_path, encoding="utf-8",na_values=["", " ", "NaN", "NA"])
    
    df_pass = df[df["passName"].str.contains(f'{pass_a}|{pass_b}',na=False, regex=True)]
    if path == []:
        path = df_pass["verifyType"].dropna().unique()
    for pi in path:
        df_path = df_pass[df_pass["verifyType"] == pi]
        loop_info = df_path["loopInfo"].dropna().unique()
        for loopi in loop_info:
            df_loop = df_path[df_path["loopInfo"] == loopi]
            res = loop_compare(pass_a, pass_b, df_loop, rawTensorList)
            if not res:
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Pass Compare", epilog="")
    parser.add_argument("--p", nargs='*', type=str, default=[], required=True, help = 'Enter the names of the two passes to be compared, separated by a space, for example, ExpandFunction RemoveUndrivenView.')
    parser.add_argument("--path", nargs='*', type=str, default=[], help = 'Names of the paths in the pass to be compared. The path names are separated by spaces. If this parameter is left blank, all paths are executed.')
    parser.add_argument("--raw", nargs='*',type=int, default=[], help = 'Specify which raw tensors need to be compared, separated by spaces. If this field is left blank, all raw tensors will be compared.')
    parser.add_argument("--verify_path" , nargs='*', type=str, default=[], help = 'Directory where the verify file is stored, end with "/". The default value is empty, If the size is 2, the two values respectively represent two passes')
    parser.add_argument("--sort",action='store_true', help = 'Whether to sort the data when drawing a chart when there is no comparison (default value: False)')
    parser.add_argument("--atol",type=float, default=1e-3, help = 'atol')
    parser.add_argument("--rtol",type=float, default=1e-3, help = 'rtol')
    parser.add_argument("--topk",type=int, default=50, help = 'Print lines')
    
    args = parser.parse_args()
    pass_name = args.p
    path = args.path
    verify_path = args.verify_path
    print(verify_path)
    if len(verify_path) == 2 :
        verify_path_pass1 = verify_path[0]
        verify_path_pass2 = verify_path[1]
    elif len(verify_path) == 1:
        verify_path_pass1 = verify_path[0]
        verify_path_pass2 = verify_path[0]
    rawTensorList = args.raw
    atol = args.atol
    rtol = args.rtol
    is_sort = args.sort
    topk = args.topk
    if len(pass_name) != 2:
        print("传入的pass的个数不为2!!!")
        sys.exit()
    print(f"对比的两个pass为 {pass_name[0]}, {pass_name[1]}")
    print(f"rawTensorList : {rawTensorList}")
    print(f"path : {path}")
    print(f"verify_path_pass1 : {verify_path_pass1}")
    print(f"verify_path_pass2 : {verify_path_pass2}")
    pass_compare(pass_name[0], pass_name[1], path, rawTensorList)