# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch
import numpy as np
import tensorflow as tf
from libs.tools import get_np_dtype


def convert_numpy_to_torch(input_data, dtype):
    if dtype in ["bf16", "bfloat16"]:
        input_tensor = torch.from_numpy(input_data.astype(np.float32)).to(torch.bfloat16)
    else:
        input_tensor = torch.from_numpy(input_data)
    return input_tensor 


def generate_uniform_inputs(shape, dtype, low, high):
    np_dtype = get_np_dtype(dtype)
    data = np.random.uniform(low=low, high=high, size=shape).astype(np_dtype)
    return data 


def generate_normal_inputs(shape, dtype, loc, scale):
    np_dtype = get_np_dtype(dtype)
    data = np.random.normal(loc=loc, scale=scale, size=shape).astype(np_dtype)
    return data 


def generate_outliers(input_data):
    input_shape = input_data.shape
    input_dtype = input_data.dtype
    total_count = np.prod(input_shape)
    input_data_flat = input_data.reshape(-1)
    outliers_index = np.random.randint(0, total_count, size=int(total_count * 0.001))
    scales = np.ones((total_count, ), dtype=input_dtype)
    scales[outliers_index] *= 1000
    input_data_flat = input_data_flat * scales
    output_data = input_data_flat.reshape(input_shape)
    return output_data


def generate_data_range():
    data_range_dict = {
        "0": "-5_5_uniform_normal",
        "1": None,
        "2": "-1_1_uniform_normal",
        "3": "0_1_uniform_normal",
        "4": "-0.01_0.01_uniform_normal",
        "5": "-0.001_0.001_uniform_normal",
        "6": "-1_1_uniform_robust",
        "7": "0_1_uniform_robust",
        "8": "-0.01_0.01_uniform_robust",
        "9": "-0.001_0.001_uniform_robust",
        "10": "0_1_normal_normal",
        "11": "1_1_normal_normal",
        "12": "0_0.001_normal_normal",
        "13": "0_1_normal_robust",
        "14": "1_1_normal_robust",
        "15": "0_0.001_normal_robust",
    }
    
    data_range = np.random.choice(list(data_range_dict.values()))
    if data_range is None:
        loc = np.round(np.random.uniform(-100, 100), decimals=2)
        scale = np.round(np.random.uniform(1, 25), decimals=2)
        data_range = f"{loc}_{scale}_normal_normal"
    return data_range


def generate_operator_data_range():
    # 对于融合算子而言，不需要超级大的data_range，会导致最终的输出溢出
    data_range_dict_normal = {
        "2": "-1_1_uniform_normal",
        "3": "0_1_uniform_normal",
        "4": "-0.01_0.01_uniform_normal",
        "5": "-0.001_0.001_uniform_normal",
        "10": "0_1_normal_normal",
        "11": "1_1_normal_normal",
        "12": "0_0.001_normal_normal",
    }
    
    data_range_dict_robust = {
        "6": "-1_1_uniform_robust",
        "7": "0_1_uniform_robust",
        "8": "-0.01_0.01_uniform_robust",
        "9": "-0.001_0.001_uniform_robust",
        "13": "0_1_normal_robust",
        "14": "1_1_normal_robust",
        "15": "0_0.001_normal_robust"
    }
    
    if np.random.random() >= 0.9:
        data_range = np.random.choice(list(data_range_dict_robust.values()))
    else:
        data_range = np.random.choice(list(data_range_dict_normal.values()))
    return data_range


def generate_inputs(shape, dtype, data_range, return_pt=True):
    # 浮点数生成
    # "xx_xx_xx_xx" -1_1_uniform_normal -0.001_0.001_uniform_robust 0_1_normal_robust -5_5_uniform_normal
    data_range_split = data_range.split("_")
    assert len(data_range_split) == 4, "data_range must be low/loc_high/scale_uniform/normal_normal/robust, pls check"
    data_dist_mode = data_range_split[2]
    outlier_mode = data_range_split[3]
    assert data_dist_mode in ["uniform", "normal"]
    assert outlier_mode in ["normal", "robust"]
    
    if data_dist_mode == "uniform":
        data = generate_uniform_inputs(shape, dtype, float(data_range_split[0]), float(data_range_split[1]))
    else:
        data = generate_normal_inputs(shape, dtype, float(data_range_split[0]), float(data_range_split[1]))
    
    if outlier_mode == "robust":
        data = generate_outliers(data)
    
    if return_pt:
        data = convert_numpy_to_torch(data, dtype)
    return data


def generate_quant_inputs(shape, dtype, data_range, quant_mode="per_tensor", return_pt=True):
    # 整形生成，对于inputs来说，建议采用per_tensor/per_token量化，对于weights来说，建议采用per_channel量化
    assert quant_mode in ["per_tensor", "per_channel", "per_token"], "currently only support per_tensor per_channel per_token quant"
    input_fp32 = generate_inputs(shape, "fp32", data_range, return_pt=False)
    abs_res = np.abs(input_fp32.astype(np.float32))
    if quant_mode == "per_channel":
        max_value = np.max(abs_res + 1e-10, axis=-2, keepdims=True) + 1e-10
    elif quant_mode == "per_token":
        max_value = np.max(abs_res + 1e-10, axis=-1, keepdims=True) + 1e-10
    else:
        max_value = np.max(abs_res) + 1e-10
    scale_quant = 127.0 / max_value
    out_fp32 = input_fp32 * scale_quant
    out_int32 = np.round(out_fp32).astype(np.int32)
    out_int8 = np.clip(out_int32, a_min=-128, a_max=127).astype(get_np_dtype(dtype))
    scale_dequant = 1.0 / scale_quant 
    
    if quant_mode == "per_tensor":
        scale_dequant = np.float32(scale_dequant)
    else:
        scale_dequant = scale_dequant
    if return_pt:
        out_int8 = convert_numpy_to_torch(out_int8, dtype)
        if quant_mode in ["per_channel", "per_token"]:
            scale_dequant = convert_numpy_to_torch(scale_dequant, "fp32")
    return out_int8, scale_dequant


def generate_quant_inputs_nonnegative(shape, dtype, data_range, quant_mode="per_tensor", return_pt=True):
    if isinstance(data_range, str):
        import ast
        try:
            data_range = ast.literal_eval(data_range)  
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid data_range string: {data_range}, error: {e}")
    print(f"  debug------data_range = {data_range}")

# 确保 data_range 是一个包含两个元素的列表或元组
    print(f"  debug------len(data_range) = {len(data_range)}")

    assert len(data_range) == 2, "data_range should be a list or tuple with two elements"
    min_value, max_value = data_range
    
    # 生成指定范围内的整数数据
    input_int = np.random.randint(min_value, max_value + 1, size=shape, dtype=get_np_dtype(dtype))
    
    if return_pt:
        input_int = convert_numpy_to_torch(input_int, dtype)

    print(f"  debug------input_int = {input_int}")

    return input_int, None


def generate_boolean_inputs(shape, p):
    data = np.random.binomial(n=1, p=p, size=shape).astype(bool)
    # print(f"  debugboolean_inputs data = {data}")
    return data


def generate_l0_matmul_shape():
    tp_size_list = [1, 2, 4, 8, 16, 32, 64]
    # [b s h] [h h] [bs h] [h h]

    batch_size_decode_list = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_size_prefill_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    decode_seq_len_list = [1, 2, 3]
    # prefill_seq_len_list = list(range(1024, 16385, 512))  # 当前先约束到128k
    prefill_seq_len_list = list(range(16, 2049, 16)) + np.random.randint(low=2048, high=16385, size=8).tolist()

    ori_hidden_size_list = [1536, 2048, 4096, 5120, 6636, 7168, 8192, 12288, 16384, 24576]
    hidden_size_list = []
    for ori_hidden_size in ori_hidden_size_list:
        for tp_size in tp_size_list:
            if ori_hidden_size % tp_size == 0:
                hidden_size_list.append(ori_hidden_size // tp_size)
    hidden_size_list = list(set(hidden_size_list))
    
    m_list_decode = []
    for batch_size_decode in batch_size_decode_list:
        for seq_len in decode_seq_len_list:
            m_list_decode.append(batch_size_decode * seq_len)
    
    m_list_prefill = [] 
    for batch_size_prefill in batch_size_prefill_list:
        for seq_len in prefill_seq_len_list:
            m_list_prefill.append(batch_size_prefill * seq_len)
    
    is_prefill = np.random.random() >= 0.9
    m = np.random.choice(m_list_prefill) if is_prefill else np.random.choice(m_list_decode)
    k = np.random.choice(hidden_size_list)
    if np.random.random() >= 0.75:
        n = k 
    else:
        n = np.random.choice(hidden_size_list)
    if m * k < 2 ** 31 and k * n < 2 ** 31:
        return [m, k, n]
    else:
        return None
        

def generate_l1_matmul_shape():
    tp_size_list = [1, 2, 4, 8, 16, 32, 64]

    batch_size_decode_list = list(range(1, 129)) + list(range(128, 513, 32))  # 均值为80
    batch_size_prefill_list = list(range(1, 17)) # 均值为8
    decode_seq_len_list = list(range(1, 9))
    prefill_seq_len_list = list(range(1, 2049)) + np.random.randint(low=2048, high=16385, size=128).tolist()  # 当前先约束到128k 均值为1.5k # 2k * 3072

    ori_hidden_size_list = [1536, 2048, 4096, 5120, 6636, 7168, 8192, 12288, 16384, 24576]
    hidden_size_list = []
    for ori_hidden_size in ori_hidden_size_list:
        for tp_size in tp_size_list:
            if ori_hidden_size % tp_size == 0:
                hidden_size_list.append(ori_hidden_size // tp_size)
    hidden_size_list = list(set(hidden_size_list))
    
    m_list_decode = []
    for batch_size_decode in batch_size_decode_list:
        for seq_len in decode_seq_len_list:
            m_list_decode.append(batch_size_decode * seq_len)
    
    m_list_prefill = []
    for batch_size_prefill in batch_size_prefill_list:
        for seq_len in prefill_seq_len_list:
            m_list_prefill.append(batch_size_prefill * seq_len)
    
    is_prefill = np.random.random() >= 0.9
    m = np.random.choice(m_list_prefill) if is_prefill else np.random.choice(m_list_decode)
    k = np.random.choice(hidden_size_list)
    if np.random.random() >= 0.75:
        n = k 
    else:
        n = np.random.choice(hidden_size_list)
    if m * k < 2 ** 31 and k * n < 2 ** 31:
        return [m, k, n]
    else:
        return None
    

def generate_l0_vector_input_shape(dim_size):
    tp_size_list = [1, 2, 4, 8, 16, 32, 64]

    batch_size_decode_list = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_size_prefill_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    decode_seq_len_list = [1, 2, 3]
    prefill_seq_len_list = list(range(16, 2049, 16)) + np.random.randint(low=2048, high=16385, size=8).tolist()

    ori_hidden_size_list = [1536, 2048, 4096, 5120, 6636, 7168, 8192, 12288, 16384, 24576]
    hidden_size_list = []
    for ori_hidden_size in ori_hidden_size_list:
        for tp_size in tp_size_list:
            if ori_hidden_size % tp_size == 0:
                hidden_size_list.append(ori_hidden_size // tp_size)
    hidden_size_list = list(set(hidden_size_list))

    ori_num_heads_list = [1, 4, 8, 16, 24, 32, 40, 48, 64, 96]
    num_heads_list = []
    for ori_num_heads in ori_num_heads_list:
        for tp_size in tp_size_list:
            if ori_num_heads % tp_size == 0:
                num_heads_list.append(ori_num_heads // tp_size)
    num_heads_list = list(set(num_heads_list))
    head_dim_list = [32, 64, 128, 256, 512]
    
    def generate_dim2_shape():
        # 2维的shape主要是[b*s, h]
        is_prefill = np.random.random() >= 0.9
        while True:
            batch_size = np.random.choice(batch_size_prefill_list) if is_prefill else np.random.choice(batch_size_decode_list)
            seq_len = np.random.choice(prefill_seq_len_list) if is_prefill else np.random.choice(decode_seq_len_list)
            hidden_size = np.random.choice(hidden_size_list)
            if batch_size * seq_len * hidden_size < 2 ** 31:
                return [batch_size * seq_len, hidden_size]
            else:
                return None 

    def generate_dim3_shape():
        # 3维shape主要是 [b s h]
        is_prefill = np.random.random() >= 0.9
        while True:
            batch_size = np.random.choice(batch_size_prefill_list) if is_prefill else np.random.choice(batch_size_decode_list)
            seq_len = np.random.choice(prefill_seq_len_list) if is_prefill else np.random.choice(decode_seq_len_list)
            hidden_size = np.random.choice(hidden_size_list)
            # print(batch_size, batch_size_decode_list, seq_len, decode_seq_len_list, hidden_size, hidden_size_list)
            if batch_size * seq_len * hidden_size < 2 ** 31:
                return [batch_size, seq_len, hidden_size]
            else:
                return None 

    def generate_dim4_shape():
        # 4维shape主要是 [b s n d] [b n s d] [b n s s]
        is_bnss = np.random.random() >= 0.9
        is_prefill = np.random.random() >= 0.9
        is_bsnd = np.random.random() >= 0.5
        while True:
            batch_size = np.random.choice(batch_size_prefill_list) if is_prefill else np.random.choice(batch_size_decode_list)
            q_seq_len = np.random.choice(decode_seq_len_list)
            kv_seq_len = np.random.choice(prefill_seq_len_list)
            num_heads = np.random.choice(num_heads_list)
            head_dim = np.random.choice(head_dim_list)

            if is_bnss:
                if is_prefill:
                    # 全量推理场景
                    if batch_size * num_heads * kv_seq_len * kv_seq_len < 2 ** 31:
                        return [batch_size, num_heads, kv_seq_len, kv_seq_len]
                    else:
                        return None 
                else:
                    if batch_size * num_heads * q_seq_len * kv_seq_len < 2 ** 31:
                        return [batch_size, num_heads, q_seq_len, kv_seq_len]
                    else:
                        return None 
            else:
                if is_prefill:
                    if batch_size * num_heads * kv_seq_len * head_dim <= 2 ** 31:
                        if is_bsnd:
                            return [batch_size, kv_seq_len, num_heads, head_dim]
                        else:
                            return [batch_size, num_heads, kv_seq_len, head_dim]
                    else:
                        return None 
                else:
                    if batch_size * num_heads * q_seq_len * head_dim <= 2 ** 31:
                        if is_bsnd:
                            return [batch_size, q_seq_len, num_heads, head_dim]
                        else:
                            return [batch_size, num_heads, q_seq_len, head_dim]
                    else:
                        return None 
    
    assert dim_size in [2, 3, 4]
    if dim_size == 2:
        input_shape = generate_dim2_shape()
    elif dim_size == 3:
        input_shape = generate_dim3_shape()
    else:
        input_shape = generate_dim4_shape()
    return input_shape


def generate_l1_vector_input_shape(dim_size):
    tp_size_list = [1, 2, 4, 8, 16, 32, 64]

    batch_size_decode_list = list(range(1, 129)) + list(range(128, 513, 32))  # 均值为80
    batch_size_prefill_list = list(range(1, 17)) # 均值为8
    
    decode_seq_len_common_list = [1, 2, 3, 4] # 均值为2 
    decode_seq_len_list = list(range(1, 9)) # 均值为4  #400 * 3000
    
    prefill_seq_len_common_list = list(range(32, 2048, 32)) + list(range(2048, 16384, 2048)) # 均值为2k
    prefill_seq_len_list = list(range(1, 2049)) + np.random.randint(low=2048, high=16385, size=128).tolist()  # 当前先约束到128k 均值为1.5k # 2k * 3072

    ori_hidden_size_list = [1536, 2048, 4096, 5120, 6636, 7168, 8192, 12288, 16384, 24576]
    hidden_size_list = []

    for ori_hidden_size in ori_hidden_size_list:
        for tp_size in tp_size_list:
            if ori_hidden_size % tp_size == 0:
                hidden_size_list.append(ori_hidden_size // tp_size)
    hidden_size_list = list(set(hidden_size_list))

    ori_num_heads_list = [1, 4, 8, 16, 24, 32, 40, 48, 64, 96]
    num_heads_list = []
    for ori_num_heads in ori_num_heads_list:
        for tp_size in tp_size_list:
            if ori_num_heads % tp_size == 0:
                num_heads_list.append(ori_num_heads // tp_size)
    num_heads_list = list(set(num_heads_list))
    head_dim_list = [16, 24, 32, 48, 64, 128, 256, 512]

    def generate_dim2_shape():
        # 2维的shape主要是[b*s, h]
        is_prefill = np.random.random() >= 0.9
        is_common = np.random.random() >= 0.8
        while True:
            batch_size = np.random.choice(batch_size_prefill_list) if is_prefill else np.random.choice(batch_size_decode_list)
            if is_prefill:
                if is_common:
                    seq_len = np.random.choice(prefill_seq_len_common_list)
                else:
                    seq_len = np.random.choice(prefill_seq_len_list)
            else:
                if is_common:
                    seq_len = np.random.choice(decode_seq_len_common_list)
                else:
                    seq_len = np.random.choice(decode_seq_len_list)
            hidden_size = np.random.choice(hidden_size_list)
            if batch_size * seq_len * hidden_size < 2 ** 31:
                return [batch_size * seq_len, hidden_size]
            else:
                return None 

    def generate_dim3_shape():
        # 3维shape主要是 [b s h]
        is_prefill = np.random.random() >= 0.9
        is_common = np.random.random() >= 0.8
        while True:
            batch_size = np.random.choice(batch_size_prefill_list) if is_prefill else np.random.choice(batch_size_decode_list)
            if is_prefill:
                if is_common:
                    seq_len = np.random.choice(prefill_seq_len_common_list)
                else:
                    seq_len = np.random.choice(prefill_seq_len_list)
            else:
                if is_common:
                    seq_len = np.random.choice(decode_seq_len_common_list)
                else:
                    seq_len = np.random.choice(decode_seq_len_list)
            hidden_size = np.random.choice(hidden_size_list)
            if batch_size * seq_len * hidden_size < 2 ** 31:
                return [batch_size, seq_len, hidden_size]
            else:
                return None 

    def generate_dim4_shape():
        # 4维shape主要是 [b s n d] [b n s d] [b n s s]
        is_bnss = np.random.random() >= 0.9
        is_prefill = np.random.random() >= 0.9
        is_common = np.random.random() >= 0.8
        is_bsnd = np.random.random() >= 0.5
        while True:
            batch_size = np.random.choice(batch_size_prefill_list) if is_prefill else np.random.choice(batch_size_decode_list)
            q_seq_len = np.random.choice(decode_seq_len_common_list) if is_common else np.random.choice(decode_seq_len_list)
            kv_seq_len = np.random.choice(prefill_seq_len_common_list) if is_common else np.random.choice(prefill_seq_len_list)
            # num_heads = np.random.choice(num_heads_list)
            hidden_size = np.random.choice(hidden_size_list)
            head_dim = np.random.choice(head_dim_list)
            num_heads = hidden_size // head_dim 
            if num_heads not in num_heads_list or hidden_size % head_dim != 0:
                return None 

            if is_bnss:
                if is_prefill:
                    # 全量推理场景
                    if batch_size * num_heads * kv_seq_len * kv_seq_len < 2 ** 31:
                        return [batch_size, num_heads, kv_seq_len, kv_seq_len]
                    else:
                        return None
                else:
                    if batch_size * num_heads * q_seq_len * kv_seq_len < 2 ** 31:
                        return [batch_size, num_heads, q_seq_len, kv_seq_len]
                    else:
                        return None
            else:
                if is_prefill:
                    if batch_size * num_heads * kv_seq_len * head_dim <= 2 ** 31:
                        if is_bsnd:
                            return [batch_size, kv_seq_len, num_heads, head_dim]
                        else:
                            return [batch_size, num_heads, kv_seq_len, head_dim]
                    else:
                        return None 
                else:
                    if batch_size * num_heads * q_seq_len * head_dim <= 2 ** 31:
                        if is_bsnd:
                            return [batch_size, q_seq_len, num_heads, head_dim]
                        else:
                            return [batch_size, num_heads, q_seq_len, head_dim]
                    else:
                        return None 

    assert dim_size in [2, 3, 4]
    if dim_size == 2:
        input_shape = generate_dim2_shape()
    elif dim_size == 3:
        input_shape = generate_dim3_shape()
    else:
        input_shape = generate_dim4_shape()
    return input_shape


if __name__ == "__main__":
    total_count1 = [] 
    total_count2 = [] 
    for _ in range(10000):
        cur_shape = generate_l0_vector_input_shape(4)
        print(cur_shape)
        if cur_shape is None:
            continue 
        total_count1.append(cur_shape[0] * cur_shape[1] * cur_shape[2] * cur_shape[3])
    print(np.mean(total_count1))
