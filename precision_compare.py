# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np
from libs.tools import get_np_dtype, get_precision_np_dtype, convert_torch_to_numpy


small_value_thres_dict = {
        "fp16": 2**-11, "float16": 2**-11,
        "bf16": 2**-8, "bfloat16": 2**-8,
        "fp32": 2**-14, "float32": 2**-14
    }


small_value_error_thres_dict = {
        "fp16": 2**-16, "float16": 2**-16,
        "bf16": 2**-16, "bfloat16": 2**-16,
        "fp32": 2**-30, "float32": 2**-30
    }


def filter_inf_nan(input_data, golden_data, bm_data=None):
    if golden_data.dtype == np.float32 and input_data.dtype != np.float32:
        precision_index = np.where(np.abs(golden_data) < 65504)[0]
    else:
        precision_index = None 
    inf_index = np.where(np.logical_not(np.isinf(golden_data)))[0]
    nan_index = np.where(np.logical_not(np.isnan(golden_data)))[0]
    filter_index = np.intersect1d(inf_index, nan_index)
    if precision_index is not None:
        filter_index = np.intersect1d(filter_index, precision_index)
    
    input_data_filter = input_data[filter_index]
    golden_data_filter = golden_data[filter_index]
    if bm_data is not None:
        bm_data_filter = bm_data[filter_index]
        return input_data_filter, golden_data_filter, bm_data_filter
    else:
        return input_data_filter, golden_data_filter
    

def precision_compare_double(npu_output_path, bm_output_path, dtype, atol, rtol, pct_thd):
    np_dtype = get_np_dtype(dtype)
    npu_data = np.fromfile(npu_output_path, dtype=np_dtype)
    bm_data = np.fromfile(bm_output_path, dtype=np_dtype)
    npu_data, bm_data = filter_inf_nan(npu_data, bm_data)
    if bm_data.size == 0:
        return "INVALID", 100.0
    
    if dtype in ["bf16", "bfloat16"]: # np.isclose在bf16场景下有bug
        npu_data = npu_data.astype(np.float32)
        bm_data = bm_data.astype(np.float32)
    diff_result = np.isclose(npu_data, bm_data, rtol=rtol, atol=atol, equal_nan=True)
    error_idx = np.where(diff_result != np.array((True, )))[0]
    total_count = bm_data.size 
    fullfill_percent = float(total_count - error_idx.size) / float(total_count) * 100.0
    
    if fullfill_percent >= (1 - pct_thd) * 100.0:
        result = "PASS"
    else:
        result = "FAILED"
    return result, fullfill_percent


def precision_compare_double_data(npu_data, bm_data, dtype, atol, rtol, pct_thd):
    npu_data = npu_data.flatten()
    bm_data = bm_data.flatten()
    npu_data, bm_data = filter_inf_nan(npu_data, bm_data)
    if bm_data.size == 0:
        return "INVALID", 100.0

    if dtype in ["bf16", "bfloat16"]:  # np.isclose在bf16场景下有bug
        npu_data = npu_data.astype(np.float32)
        bm_data = bm_data.astype(np.float32)
    diff_result = np.isclose(
        npu_data, bm_data, rtol=rtol, atol=atol, equal_nan=True)
    error_idx = np.where(diff_result != np.array((True, )))[0]
    total_count = bm_data.size
    fullfill_percent = float(
        total_count - error_idx.size) / float(total_count) * 100.0

    if fullfill_percent >= (1 - pct_thd) * 100.0:
        result = "PASS"
    else:
        result = "FAILED"
    return result, fullfill_percent


def get_split_index(golden_data, dtype):
    small_value_thres = small_value_thres_dict[dtype]
    large_value_idx = np.where(np.abs(golden_data) >= small_value_thres)[0]
    small_value_idx = np.where(np.abs(golden_data) < small_value_thres)[0]
    return large_value_idx, small_value_idx, small_value_thres
    

def compute_matrix_small_value(input_data, golden_data, dtype, small_index):
    if small_index.size == 0:
        return 0
    thres = small_value_error_thres_dict[dtype]
    error_idx = np.where(np.abs(input_data[small_index] - golden_data[small_index]) > thres)[0]
    error_count = error_idx.size 
    return error_count 


def compute_matrix_large_value(input_data, golden_data, large_index):
    if large_index.size == 0:
        return 0, 0, 0, 0, 0
    input_data_large = input_data[large_index]
    golden_data_large = golden_data[large_index]
    relative_error = np.abs(input_data_large - golden_data_large) / (np.abs(golden_data_large) + 1e-7)
    mare = np.max(relative_error)
    mere = np.mean(relative_error)
    rmse = np.sqrt(np.mean((input_data_large - golden_data_large) ** 2))
    #print(f"mare:{mare}")
    #print(f"mere:{mere}")
    #print(f"rmse:{rmse}")
    #print(f"relative_error:{relative_error}")
    return mare, mere, rmse, relative_error, np.abs(input_data_large - golden_data_large)


def compute_re_matrix(input_value, bm_value, small_value_thres):
    if bm_value in [np.inf, np.nan]:
        return 1 
    else:
        if input_value in [np.inf, np.nan]:
            return 1000
        else:
            return input_value / max(bm_value, small_value_thres)


def compute_re_triplet_matrix(npu_matrix, golden_matrix, small_value_thres):
    mare_npu, mere_npu, rmse_npu = npu_matrix
    mare_bm, mere_bm, rmse_bm = golden_matrix
    mare_matrix = compute_re_matrix(mare_npu, mare_bm, small_value_thres)
    mere_matrix = compute_re_matrix(mere_npu, mere_bm, small_value_thres)
    #print(f"rmse_npu:{rmse_npu}")
    #print(f"rmse_bm:{rmse_bm}")
    #print(f"small_value_thres:{small_value_thres}")
    rmse_matrix = compute_re_matrix(rmse_npu, rmse_bm, small_value_thres)
    #print(f"rmse_matrix:{rmse_matrix}")
    return mare_matrix, mere_matrix, rmse_matrix
    

def precision_compare_triple(npu_output_path, bm_output_path, golden_output_path, dtype, thres=(2, 1.2, 1.2)):
    if dtype in ["int8", "int32"]:
        raise NotImplementedError("precision compare triplet only support float")
    np_dtype = get_np_dtype(dtype)
    np_precision_dtype = get_precision_np_dtype(dtype)
    
    npu_data = np.fromfile(npu_output_path, dtype=np_dtype)
    bm_data = np.fromfile(bm_output_path, dtype=np_dtype)
    golden_data = np.fromfile(golden_output_path, dtype=np_precision_dtype)
    npu_data, golden_data, bm_data = filter_inf_nan(npu_data, golden_data, bm_data)
    if golden_data.size == 0:
        return "INVALID", 0.0, 0.0, 0.0, 0.0
    large_value_idx, small_value_idx, small_value_thres = get_split_index(golden_data, dtype)
    
    # 小值域场景
    npu_error_count = compute_matrix_small_value(npu_data, golden_data, dtype, small_value_idx)
    bm_error_count = compute_matrix_small_value(bm_data, golden_data, dtype, small_value_idx)
    small_value_matrix = npu_error_count / max(bm_error_count, 1)
    
    # 大值域场景
    mare_npu, mere_npu, rmse_npu, npu_relative_error, npu_absolute_error = compute_matrix_large_value(npu_data, golden_data, large_value_idx)
    mare_bm, mere_bm, rmse_bm, bm_relative_error, bm_absolute_error = compute_matrix_large_value(bm_data, golden_data, large_value_idx)
    mare_matrix, mere_matrix, rmse_matrix = compute_re_triplet_matrix([mare_npu, mere_npu, rmse_npu], [mare_bm, mere_bm, rmse_bm], small_value_thres)
    
    if small_value_matrix <= 2 and mare_matrix <= thres[0] and mere_matrix <= thres[1] and rmse_matrix <= thres[2]:
        result = "PASS"
    else:
        result = "FAILED"
    return result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix


def precision_compare_triple_data(npu_data, bm_data, golden_data, dtype, thres=(2, 1.2, 1.2), logger=None):
    if dtype in ["int8", "int32"]:
        raise NotImplementedError("precision compare triplet only support float")
    npu_data = npu_data.flatten()
    bm_data = bm_data.flatten()
    golden_data = golden_data.flatten()
    if logger is not None:
        logger.info("# origin shape; pypto_data: {} bm_data: {} cpu_data: {}".format(npu_data.shape, bm_data.shape, golden_data.shape))
        
    npu_data, golden_data, bm_data = filter_inf_nan(npu_data, golden_data, bm_data)
    if golden_data.size == 0:
        return "INVALID", 0.0, 0.0, 0.0, 0.0
    if logger is not None:
        logger.info("# filter shape; pypto_data: {} bm_data: {} cpu_data: {}".format(npu_data.shape, bm_data.shape, golden_data.shape))
        
    large_value_idx, small_value_idx, small_value_thres = get_split_index(golden_data, dtype)
    
    # 小值域场景
    npu_error_count = compute_matrix_small_value(npu_data, golden_data, dtype, small_value_idx)
    bm_error_count = compute_matrix_small_value(bm_data, golden_data, dtype, small_value_idx)
    small_value_matrix = npu_error_count / max(bm_error_count, 1)
    if logger is not None:
        logger.info("# small_value_matrix; pypto error_count: {} bm_error_count: {} total_count: {}".format(npu_error_count, bm_error_count, small_value_idx.size))
    
    # 大值域场景
    #print(f"npu_data:{npu_data}")
    #print(f"golden_data:{golden_data}")
    #print(f"large_value_idx:{large_value_idx}")

    mare_npu, mere_npu, rmse_npu, npu_relative_error, npu_absolute_error = compute_matrix_large_value(npu_data, golden_data, large_value_idx)
    mare_bm, mere_bm, rmse_bm, bm_relative_error, bm_absolute_error = compute_matrix_large_value(bm_data, golden_data, large_value_idx)
    if logger is not None and large_value_idx.size >= 20:
        logger.info("# large_value_martix; pypto MARE: {} pypto MERE: {} pypto RMSE: {}".format(mare_npu, mere_npu, rmse_npu))
        logger.info("# large_value_martix; bm MARE: {} bm MERE: {} bm RMSE: {}".format(mare_bm, mere_bm, rmse_bm))
        
        pypto_re_top20 = np.argsort(npu_relative_error)[-20:][::-1]
        logger.info("=" * 60 + " pypto relative error top 20: ")
        logger.info("-" * 100)
        logger.info(f"{'Index':<10} {'pypto output':<30} {'golden':<30} {'relative error':<30}")
        logger.info("-" * 100)
        for idx in pypto_re_top20:
            real_idx = large_value_idx[idx]
            logger.info(f"{real_idx:<10} {npu_data[real_idx]:<30} {golden_data[real_idx]:<30} {npu_relative_error[idx]:<30}")
        logger.info("-" * 100)
        
        pypto_ae_top20 = np.argsort(npu_absolute_error)[-20:][::-1]
        logger.info("=" * 60 + " pypto absolute error top 20: ")
        logger.info("-" * 100)
        logger.info(f"{'Index':<10} {'pypto output':<30} {'golden':<30} {'absolute error':<30}")
        logger.info("-" * 100)
        for idx in pypto_ae_top20:
            real_idx = large_value_idx[idx]
            logger.info(f"{real_idx:<10} {npu_data[real_idx]:<30} {golden_data[real_idx]:<30} {npu_absolute_error[idx]:<30}")
        logger.info("-" * 100)
        
    mare_matrix, mere_matrix, rmse_matrix = compute_re_triplet_matrix([mare_npu, mere_npu, rmse_npu], [mare_bm, mere_bm, rmse_bm], small_value_thres)
    if small_value_matrix <= 2 and mare_matrix <= thres[0] and mere_matrix <= thres[1] and rmse_matrix <= thres[2]:
        result = "PASS"
    else:
        result = "FAILED"
    return result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix


def get_precision_thres(dtype, compute_count):
    if dtype in ["fp16", "float16"]:
        if compute_count >= 2048:
            err = 2 ** -7
        else:
            err = 2 ** -8 
    elif dtype in ["bf16", "bfloat16"]:
        if compute_count >= 2048:
            err = 2 ** -6
        else:
            err = 2 ** -7
    elif dtype in ["fp32", "float32"]:
        if compute_count >= 16384:
            err = 2 ** -9
        elif compute_count >= 2048:
            err = 2 ** -10
        else:
            err = 2 ** -11
    else:
        raise NotImplementedError
    return err 


def precision_compare_binary_consistency(input_data, golden_data):
    input_data = input_data.flatten()
    golden_data = golden_data.flatten()
    input_data, golden_data = filter_inf_nan(input_data, golden_data)
    
    if input_data.dtype == "bfloat16": # np.isclose在bf16场景下有bug
        input_data = input_data.astype(np.float32)
        golden_data = golden_data.astype(np.float32)
    diff_result = np.isclose(input_data, golden_data, rtol=0, atol=0, equal_nan=True)
    error_idx = np.where(diff_result != np.array((True, )))[0]
    error_count = error_idx.size
    total_count = input_data.size 
    if error_count != 0:
        result = "FAILED"
    else:
        result = "PASS"
    fullfill_percent = float(total_count - error_count) / float(total_count) * 100.0
    return result, fullfill_percent


def precision_compare_quant(input_data, golden_data):
    input_data = input_data.flatten()
    golden_data = golden_data.flatten()
    input_data, golden_data = filter_inf_nan(input_data, golden_data)
    diff_result = np.isclose(input_data, golden_data, rtol=0, atol=1, equal_nan=True)
    error_idx = np.where(diff_result != np.array((True, )))[0]
    error_count = error_idx.size
    total_count = input_data.size 
    if error_count != 0:
        result = "FAILED"
    else:
        result = "PASS"
    fullfill_percent = float(total_count - error_count) / float(total_count) * 100.0
    return result, fullfill_percent


def precision_compare_float(input_data, golden_data, dtype, compute_count):
    err = get_precision_thres(dtype, compute_count)
    input_data = input_data.flatten()
    golden_data = golden_data.flatten()
    input_data, golden_data = filter_inf_nan(input_data, golden_data)
    
    small_value_thres = small_value_thres_dict[dtype]
    large_value_idx = np.where(np.abs(golden_data) >= small_value_thres)[0]
    small_value_idx = np.where(np.abs(golden_data) < small_value_thres)[0]
    
    input_data_large = input_data[large_value_idx]
    golden_data_large = golden_data[large_value_idx]
    relative_error = np.abs(input_data_large - golden_data_large) / (np.abs(golden_data_large) + 1e-7)
    relative_error_idx = np.where(relative_error > err)[0]
    relative_error_count = relative_error_idx.size
    if input_data_large.size == 0:
        relative_error_percent = 100.0
    else:
        relative_error_percent = float(input_data_large.size - relative_error_count) / float(input_data_large.size) * 100.0
    
    input_data_small = input_data[small_value_idx]
    golden_data_small = golden_data[small_value_idx]
    absolute_error = np.abs(input_data_small - golden_data_small)
    absolute_error_idx = np.where(absolute_error > err)[0]
    absolute_error_count = absolute_error_idx.size 
    if input_data_small.size == 0:
        absolute_error_percent = 100.0
    else:
        absolute_error_percent = float(input_data_small.size - absolute_error_count) / float(input_data_small.size) * 100.0
    
    if relative_error_count != 0 or absolute_error_count != 0:
        result = "FAILED"
    else:
        result = "PASS"
    return result, relative_error_percent, absolute_error_percent
    
