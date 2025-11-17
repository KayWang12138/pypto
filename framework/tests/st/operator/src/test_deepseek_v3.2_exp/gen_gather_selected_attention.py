# !/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
"""

本脚本有 2 种执行模式:
1. CI批跑时, 由 tests/cmake/scripts/golden_ctrl.py 调用, 为避免日志过多, 此时 logging 级别为 logging.INFO;
2. 单独调试时, 本脚本单独被调用, 此时 logging 级别为 logging.DEBUG;
"""
import math
import sys
import logging
from pathlib import Path
from typing import List
import numpy as np
import torch
from bfloat16 import bfloat16

if __name__ == "__main__":
    """ 单独调试时配置 """
    # 日志级别
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    # 系统 import 路径
    g_src_root: Path = Path(Path(__file__).parent, "../../../../../").resolve()
    logging.debug("SrcRoot: %s", g_src_root)
    g_ctrl_path: Path = Path(g_src_root, "tests/cmake/scripts")
    if str(g_ctrl_path) not in sys.path:
        sys.path.append(str(g_ctrl_path))
    from golden_register import GoldenRegister  # 单独调试 import 失败, 需确认上文中 '系统 import 路径' 配置正确
else:
    from golden_register import GoldenRegister


def dump_file(data, data_path, dtype):
    """将PyTorch张量保存到文件，支持BFloat16类型转换"""
    if dtype == torch.float16:
        np_dtype = np.float16
    elif dtype == torch.float32:
        np_dtype = np.float32
    elif dtype == torch.int32:
        np_dtype = np.int32
    elif dtype == torch.bfloat16:
        np_dtype = bfloat16
    elif dtype == torch.int8:
        np_dtype = np.int8
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    if isinstance(data, torch.Tensor):
        # 处理BFloat16类型：转换为float32后再转NumPy（NumPy不支持BFloat16）
        if data.dtype == torch.bfloat16:
            data_np = data.cpu().to(torch.float32).numpy()
        else:
            data_np = data.cpu().numpy()
    else:
        data_np = np.array(data)
    # 确保最终类型与指定dtype一致
    data_np = data_np.astype(np_dtype)
    data_np.tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    """
    PyTorch版本的均匀分布数据生成，与NumPy版本行为完全一致
    严格保持 [min_value, max_value) 左闭右开区间特性
    """
    # 特殊情况：全零张量
    if min_value == 0 and max_value == 0:
        return torch.zeros(data_shape, dtype=dtype)
    # 布尔类型处理：等概率生成True/False
    if dtype == torch.bool:
        # 生成[0,2)的整数，转换为bool即等概率True/False
        return torch.randint(0, 2, data_shape, dtype=dtype)
    # 浮点类型：[min_value, max_value)
    if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
        # torch.rand生成[0,1)，缩放后得到[min_value, max_value)
        return min_value + (max_value - min_value) * torch.rand(data_shape, dtype=dtype)
    # 整数类型：[min_value, max_value)
    else:
        # torch.randint的high参数为开区间，直接对应[min_value, max_value)
        return torch.randint(low=min_value, high=max_value, size=data_shape, dtype=dtype)


def softmax(x):
    """PyTorch实现的softmax函数"""
    x = x.float()
    x_max = torch.max(x, dim=-1, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = torch.sum(y, dim=-1, keepdim=True)
    ans = y
    return ans, x_sum, x_max


def compute_attention(input_data, params):
    """
    计算注意力机制，支持不同批次的序列长度不同
    使用PyTorch实现
    """
    q, kn, kr, kn_scales, offsets, actual_seq = input_data
    # 提取维度信息
    b, s_q, n_q, d_q = q.shape
    _, kv_lora_rank = kn.shape
    _, qk_rope_dim = kr.shape
    d_k = kv_lora_rank + qk_rope_dim
    scalar, topk, d_v, is_kn_quant = params
    atten_out_shape = [b, s_q, n_q, d_v]
    # 初始化输出张量
    attention_output = torch.zeros(atten_out_shape, dtype=torch.float32)
    slc_kn = torch.zeros([topk, kv_lora_rank], dtype=torch.float32)
    slc_kr = torch.zeros([topk, qk_rope_dim], dtype=torch.float32)
    slc_kn_scales = torch.zeros([topk, 4], dtype=torch.float32)
    # 遍历每个批次
    for i in range(b):
        # 遍历每个s_q
        for j in range(s_q):
            # 获取当前批次的实际序列长度
            if isinstance(actual_seq, torch.Tensor):
                kv_seq_len = actual_seq[i].item()
            else:
                kv_seq_len = actual_seq[i]
            # s_q!=1 MTP场景下的casual计算
            seq_len = min(max(kv_seq_len - s_q + 1 + j, 0), topk)

            # 当前批次的gather，获取对应的slc_kn
            offset = offsets[i * s_q + j, :]
            for idx in range(seq_len):
                slc_idx = offset[idx]
                slc_kn[idx, :] = kn[slc_idx, :]
                slc_kr[idx, :] = kr[slc_idx, :]
                slc_kn_scales[idx, :] = kn_scales[slc_idx, :]

            # 获取当前批次和s_q的q [n_q, d_q]
            q_bs = q[i, j]
            # 获取当前批次的[seq_len, d_k/d_v]
            if is_kn_quant:
                kn_bs = slc_kn[:seq_len, :]
                kn_scales_tmp = slc_kn_scales[:seq_len, :]
                kn_bs = kn_bs.reshape(-1, 128).to(torch.float)
                kn_scales_tmp = kn_scales_tmp.reshape(-1, 1)
                kn_tmp = kn_bs * kn_scales_tmp
                kn_tmp = kn_tmp.reshape(-1, 512).to(torch.bfloat16)
            else:
                kn_tmp = slc_kn[:seq_len, :]
            kr_tmp = slc_kr[:seq_len, :]
            k_bs = torch.concat([kn_tmp, kr_tmp], dim=-1)
            v_bs = kn_tmp

            # MM1: 矩阵乘法
            qk_bmm_res = torch.matmul(q_bs.float(), k_bs.transpose(1, 0).float())
            qk_ele_res = qk_bmm_res * scalar
            # Softmax计算
            softmax_res, softmax_sum, softmax_max = softmax(qk_ele_res)
            # MM2: 矩阵乘法
            bmm2_res = torch.matmul(softmax_res / softmax_sum, v_bs.float())
            # 存储结果
            attention_output[i, j] = bmm2_res
    return attention_output


def gen_dsa_gather_sa_entry(dtype, bn1n2s1, is_kn_quant, actual_seq, output):
    torch.manual_seed(42)
    b, n_q, n_kv, s_q = bn1n2s1  # 48, 128, 1, 1
    kv_lora_rank = 512
    qk_rope_dim = 64
    topk = 2048
    np.random.seed(None)
    # q head dim
    d_q = kv_lora_rank + qk_rope_dim
    # k head dim
    d_k = kv_lora_rank + qk_rope_dim
    # v head dim
    d_v = kv_lora_rank
    scalar = d_q ** -0.5
    if isinstance(actual_seq, int):
        actual_seq = [actual_seq] * b
    elif isinstance(actual_seq, list):
        if len(actual_seq) == b:
            actual_seq = actual_seq
        else:
            raise RuntimeError("unsupported actual_seq list length")
    else:
        raise RuntimeError("unsupported actual_seq data type")
    s_max = ((max(actual_seq) + 128 - 1) // 128) * 128  # 默认block size = 128, 向上取整
    s_max = max(s_max, topk)
    # 1. 定义shape
    shape_q = [b, s_q, n_q, d_q]
    shape_kn = [b, s_max, kv_lora_rank]
    shape_kr = [b, s_max, qk_rope_dim]
    atten_out_shape = [b, s_q, n_q, d_v]

    slc_actual_seq = []
    for i in range(b):
        slc_actual_seq.append(min(actual_seq[i], topk))
    offsets = torch.zeros(b, s_q, topk).to(torch.int32)
    for b_i in range(b):
        all_nums = torch.arange(0, b * s_max, dtype=torch.int32)  # [blk_num, blk_size, ...]
        for s_q_i in range(s_q):
            perm = torch.randperm(b * s_max)
            offsets[b_i, s_q_i, :slc_actual_seq[b_i]] = all_nums[perm[:slc_actual_seq[b_i]]]
    offsets = offsets.reshape(b * s_q, n_kv * topk)

    q_bsnd = gen_uniform_data(shape_q, -1, 1, dtype)
    kn_bsnd_tmp = gen_uniform_data(shape_kn, -1, 1, dtype)

    kn_scales = kn_bsnd_tmp.reshape(b, s_max, 4, 128).to(torch.float32).abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    if is_kn_quant == 1:
        kn_quant = kn_bsnd_tmp.reshape(b, s_max, 4, 128) / kn_scales
        kn = torch.round(kn_quant).clamp(-128, 127).to(torch.int8)
        kn_bsnd = (kn.to(torch.float32) * kn_scales).to(dtype)
        kn_bsnd = kn_bsnd.reshape(b, s_max, 4 * 128)
    else:
        kn_bsnd = kn_bsnd_tmp
        kn = kn_bsnd_tmp
    kr = gen_uniform_data(shape_kr, -1, 1, dtype)
    # 2D
    kn = kn.reshape(b * s_max * n_kv, kv_lora_rank)
    kn_scales = kn_scales.reshape(b * s_max * n_kv, 4)
    kr = kr.reshape(b * s_max * n_kv, qk_rope_dim)

    # 3. 计算attention
    params = [scalar, topk, kv_lora_rank, is_kn_quant]
    input_data = [q_bsnd, kn, kr, kn_scales, offsets, actual_seq]
    atten_out = compute_attention(input_data, params)

    # 4.dump 数据
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]
    q_nope = q_nope.reshape(b * s_q * n_q, kv_lora_rank)
    q_rope = q_rope.reshape(b * s_q * n_q, qk_rope_dim)
    # input params
    input_params = [b, s_q, n_q, n_kv, kv_lora_rank, qk_rope_dim, s_max, topk, is_kn_quant]
    kn_aux_tensor = torch.eye(512, dtype=torch.float32).to(torch.int8)
    scale_aux_tensor = torch.eye(4, dtype=torch.float32)
    q_nope_path = Path(output, 'q_nope.bin')
    q_rope_path = Path(output, 'q_rope.bin')
    kn_path = Path(output, 'k_nope.bin')
    kn_aux_tensor_path = Path(output, 'knAuxTensor.bin')
    scale_aux_tensor_path = Path(output, 'scaleAuxTensor.bin')
    kr_path = Path(output, 'k_rope.bin')
    kn_scales_path = Path(output, 'kn_scales.bin')
    offsets_path = Path(output, 'offsets.bin')
    actual_seq_path = Path(output, 'actual_seq.bin')
    atten_out_path = Path(output, 'atten_out.bin')
    input_param_path = Path(output, 'input_param.bin')
    # dump golden file
    dump_file(q_nope, q_nope_path, dtype)
    dump_file(q_rope, q_rope_path, dtype)
    dump_file(kn, kn_path, kn.dtype)
    dump_file(kn_aux_tensor, kn_aux_tensor_path, torch.int8)
    dump_file(scale_aux_tensor, scale_aux_tensor_path, torch.float32)
    dump_file(kr, kr_path, dtype)
    dump_file(kn_scales, kn_scales_path, kn_scales.dtype)
    dump_file(offsets, offsets_path, torch.int32)
    dump_file(actual_seq, actual_seq_path, torch.int32)
    dump_file(atten_out, atten_out_path, dtype)
    dump_file(input_params, input_param_path, torch.int32)
    return True


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2",
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8",
    ],
    version=0,
    timeout=0
)
def dsa_sa_func(case_name: str, output: Path) -> bool:
    if case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511":
        # bn1n2s1数据: b, n_q, n_kv, s_q; n_kv=1
        bn1n2s1 = (32, 128, 1, 1)
        # 0为kn非量化情况，1为kn量化情况
        is_kn_quant = 0
        actual_seq = 511
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s1_seq511_int8":
        bn1n2s1 = (32, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = 511
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049":
        bn1n2s1 = (1, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = 2049
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s1_seq2049_int8":
        bn1n2s1 = (1, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = 2049
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047":
        bn1n2s1 = (1, 128, 1, 3)
        is_kn_quant = 0
        actual_seq = 2047
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b1_s3_seq2047_int8":
        bn1n2s1 = (1, 128, 1, 3)
        is_kn_quant = 1
        actual_seq = 2047
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k":
        bn1n2s1 = (128, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = 8096
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b128_s1_seq8k_int8":
        bn1n2s1 = (128, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = 8096
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = 131072  # 128k
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seq128k_int8":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = 131072
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1":
        bn1n2s1 = (4, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b4_s1_seqTest1_int8":
        bn1n2s1 = (4, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s1_seqTest2_int8":
        bn1n2s1 = (8, 128, 1, 1)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2":
        bn1n2s1 = (8, 128, 1, 4)
        is_kn_quant = 0
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    elif case_name == "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b8_s4_seqTest2_int8":
        bn1n2s1 = (8, 128, 1, 4)
        is_kn_quant = 1
        actual_seq = [666, 532, 768, 900, 5698, 2358, 324, 2048]
        gen_dsa_gather_sa_entry(torch.bfloat16, bn1n2s1, is_kn_quant, actual_seq, output)
    else:
        logging.error("Can't get func to gen golden, Case(%s)", case_name)
        return False
    return True


def main() -> bool:
    """
    单独调试 入口函数
    """
    # 用例名称
    case_name_list: List[str] = [
        "DynamicGatherSlcFlashAttnDSASTest.dsa_gather_slc_attn_bf16_b32_s2",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output: Path = Path(g_src_root, "build/output/bin/golden", cs).resolve()
        output.mkdir(parents=True, exist_ok=True)
        ret = dsa_sa_func(case_name=cs, output=output)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
