
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
import time

import numpy as np
from bfloat16 import bfloat16

from golden.net.deepseekv3.nsa.gen_slc_attn import compute_attention
from golden.op.kv_slc import kv_slc_compute
from golden.net.deepseekv3.nsa.attention_post_golden import post_compute, gen_post_input_data


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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gated_score_mlp_standard(x, w_1, w_2, output: Path):
    b, s, h = x.shape
    _, n3 = w_2.shape
    n = n3 // 3
    # if True:
    #     w_1 = np.random.rand(h, 4 * h)
    #     w_2 = np.random.rand(4*h, 3 * n)
    print(f'b {b} s {s} h {h} n {n} \n')
    x_path = Path(output, 'x.bin')
    w1_path = Path(output, 'w1.bin')
    w2_path = Path(output, 'w2.bin')
    score_path = Path(output, 'score.bin')
    _, n_heads = w_2.shape
    n = n_heads // 3
    x_2d = x.reshape(-1, h)
    mm1 = np.matmul(x_2d, w_1)
    mm1_sigmoid = sigmoid(mm1)
    mm2 = np.matmul(mm1_sigmoid, w_2)
    gating_score = mm2.reshape(b, s, 3, n)
 
    x.astype(np.float16).tofile(x_path)
    w_1.astype(np.float16).tofile(w1_path)
    w_2.astype(np.float16).tofile(w2_path)
    gating_score.astype(np.float16).tofile(score_path)
 
    return gating_score, mm1_sigmoid, mm2
 
 
def gated_score_mlp_simple(x, w_1, output: Path):
    b, s, h = x.shape
    _, n_heads = w_1.shape
    n = n_heads // 3
    x_2d = x.reshape(-1, h)
    mm1 = np.matmul(x_2d, w_1)
    mm1_sigmoid = sigmoid(mm1)
    gating_score = mm1_sigmoid.reshape(b, s, n, 3)
 
    return gating_score
 
 
def gen_gated_score(x, gate_sim_w1, gate_w1, gate_w2, output: Path, mode='standard'):
    if mode == 'standard':
        gating_score, mm1, mm2 = gated_score_mlp_standard(x, gate_w1, gate_w2, output)
    else:
        gating_score = gated_score_mlp_simple(x, gate_sim_w1, output)
 
    gating_score = gating_score.transpose((0, 1, 3, 2))
    return gating_score, mm1, mm2


def dump_file(data_pool, data_path, dtype):
    np.array(data_pool).astype(dtype).tofile(data_path)


def gen_uniform_data(data_shape, min_value, max_value, dtype):
    if min_value == 0 and max_value == 0:
        return np.zeros(data_shape, dtype=dtype)
    if dtype == np.bool_:
        return np.random.choice([True, False], size=data_shape)
    return np.random.uniform(low=min_value, high=max_value, size=data_shape).astype(
        dtype
    )


def gen_block_table(b, actual_seq_len, block_size):
    block_num_per_batch = []
    block_num_min = 0
    block_num = 0
    for actual_seq in actual_seq_len:
        block_num_per_batch.append(math.ceil(actual_seq / block_size))
        block_num_min += math.ceil(actual_seq / block_size)

    slc_s_max = max(actual_seq_len)
    # gen block table [b, slc_s_max/block_size]
    block_table_shape = [b, math.ceil(slc_s_max / block_size)]
    block_num = block_num_min

    block_idx_list = np.arange(0, block_num, 1)
    block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)

    block_idx = 0
    block_table = [-1] * block_table_shape[1]

    block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)
    block_table_batch_idx = 0
    for idx in block_num_per_batch:
        block_idx = 0
        for j in range(idx):
            block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
            block_idx += 1
        block_table_batch_idx += 1

    return block_num, block_table


def gen_kv_cache(params, actual_seq_list, dtype, output_dir):
    '''
    生成kv_cache, 包括kv_nope_cache, k_rope_cache, block_table
    '''
    b = params.get("b")
    s1 = params.get("s")
    n2 = params.get("n2")
    rope_dim = params.get("rope_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    front = params.get("front")
    near = params.get("near")
    topk = params.get("topk")
    block_size = params.get("block_size")

    block_num, block_table = gen_block_table(b, actual_seq_list, block_size)

    shape_topk_indices = [b, s1, topk - front - near]
    shape_kv_nope_cache = [block_num * block_size, n2 * kv_lora_rank]
    shape_k_rope_cache = [block_num * block_size, n2 * rope_dim]

    kv_nope_cache = gen_uniform_data(shape_kv_nope_cache, -1, 1, dtype)
    k_rope_cache = gen_uniform_data(shape_k_rope_cache, -1, 1, dtype)

    kv_nope_cache_path = Path(output_dir, 'kv_nope_cache.bin')
    kr_cache_path = Path(output_dir, 'k_rope_cache.bin')
    block_table_path = Path(output_dir, 'block_table.bin')
    kv_cache_actual_seq_path = Path(output_dir, 'kv_cache_actual_seq_len.bin')

    dump_file(kv_nope_cache, kv_nope_cache_path, dtype)
    dump_file(k_rope_cache, kr_cache_path, dtype)
    dump_file(block_table, block_table_path, np.int32)
    dump_file(actual_seq_list, kv_cache_actual_seq_path, np.int32)

    return kv_nope_cache, k_rope_cache, block_table


def gen_atten_golden_data(cmp_atten, sel_atten, win_atten, gating_score, dtype):
    '''
    内部以fp32进行运算
    '''
    fp32 = np.float32
    cmp_atten_fp32 = cmp_atten.astype(fp32)
    sel_atten_fp32 = sel_atten.astype(fp32)
    win_atten_fp32 = win_atten.astype(fp32)
    gating_score_fp32 = gating_score.astype(fp32)
    w_cmp, w_slc, w_win = np.split(gating_score_fp32, 3, axis = -1)
    attention_out_fp32 = (w_cmp * cmp_atten_fp32 + w_slc * sel_atten_fp32 + w_win * win_atten_fp32)
    attention_out = attention_out_fp32.astype(dtype)
    return attention_out


def dump_gen_kv_slc_file(topk_indices, topk_tensor_shape, kv_slc_out, kr_slc_out, kv_slc_actual_seqs, dtype, output_dir):
    topk_tensor_path = Path(output_dir, 'topk_tensor.bin')
    topk_tensor_shape_path = Path(output_dir, 'topk_tensor_shape.bin')
    # kv_slc_out_path = Path(output_dir, 'kv_slc_out.bin')
    # kr_slc_out_path = Path(output_dir, 'kr_slc_out.bin')
    kv_slc_actual_seqs_path = Path(output_dir, 'kv_slc_actual_seqs.bin')

    dump_file(topk_indices, topk_tensor_path, np.int32)
    dump_file(topk_tensor_shape, topk_tensor_shape_path, np.int32)
    # dump_file(kv_slc_out, kv_slc_out_path, dtype)
    # dump_file(kr_slc_out, kr_slc_out_path, dtype)
    dump_file(kv_slc_actual_seqs, kv_slc_actual_seqs_path, np.int32)


def dump_slc_atten_file(q_bsnd, k_bsnd, v_bsnd, actual_seq, kv_lora_rank, dtype, output_dir, atten_out, input_params):
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]

    q_nope_path = Path(output_dir, 'q_nope.bin')
    q_rope_path = Path(output_dir, 'q_rope.bin')
    k_slc_path = Path(output_dir, 'k_slc.bin')
    v_slc_path = Path(output_dir, 'v_slc.bin')
    # actual_seq_path = Path(output_dir, 'slc_actual_seq.bin')

    dump_file(q_nope, q_nope_path, dtype)
    dump_file(q_rope, q_rope_path, dtype)
    dump_file(k_bsnd, k_slc_path, dtype)
    dump_file(v_bsnd, v_slc_path, dtype)
    # dump_file(actual_seq, actual_seq_path, np.int32)


def dump_gen_atten_file(cmp_atten, sel_atten, win_atten, attention_out, dtype, output_dir):
    cmp_atten_path = Path(output_dir, 'cmp_atten.bin')
    # sel_atten_path = Path(output_dir, 'sel_atten.bin')
    win_atten_path = Path(output_dir, 'win_atten.bin')
    attention_out_path = Path(output_dir, 'attention_out.bin')

    dump_file(cmp_atten, cmp_atten_path, dtype)
    # dump_file(sel_atten, sel_atten_path, np.float32) # slc_attn, fp32
    dump_file(win_atten, win_atten_path, dtype)
    dump_file(attention_out, attention_out_path, dtype)


def dump_gated_score_file(x, gate_sim_w1, gate_w1, gate_w2, gating_score, dtype, output_dir):
    x_path = Path(output_dir, 'x.bin')
    gate_sim_w1_path = Path(output_dir, 'gate_sim_w1.bin')
    gate_w1_path = Path(output_dir, 'gate_w1.bin')
    gate_w2_path = Path(output_dir, 'gate_w2.bin')
    gating_score_path = Path(output_dir, 'gating_score.bin')

    x.astype(dtype).tofile(x_path)
    gate_sim_w1.astype(dtype).tofile(gate_sim_w1_path)
    gate_w1.astype(dtype).tofile(gate_w1_path)
    gate_w2.astype(dtype).tofile(gate_w2_path)
    gating_score.astype(dtype).tofile(gating_score_path)


def gen_nsa_golden(params, dtypes, output_dir: Path, is_nz=False):
    '''
    将整个nsa分为6个子图进行串联
    subgragh 1: gen_win_attn
    subgragh 2: kv_compression
    subgragh 3: gen_cmp_atten
    subgragh 4: gen_slc_atten, 其中包括: gen_kv_slc及slc_attn
    subgragh 5: gen_gated_score
    subgragh 6: gen_attn
    subgragh 7: post
    '''
    print("=========== start =============: nsa golden")

    dtype, w_dtype = dtypes
    logging.debug(f"gen_nsa_golden  dtype:{dtype}, w_dtype:{w_dtype}")
    b = params.get("b")
    s = params.get("s")
    s2 = params.get("s2")
    h = params.get("h")
    n1 = params.get("n1")
    n2 = params.get("n2")
    q_dim = params.get("q_dim")
    k_dim = params.get("k_dim")
    v_dim = params.get("v_dim")
    rope_dim = params.get("rope_dim")
    kv_lora_rank = params.get("kv_lora_rank")
    cmp_block_size = params.get("cmp_block_size")
    cmp_stride = params.get("cmp_stride")
    slc_block_size = params.get("slc_block_size")
    front = params.get("front")
    near = params.get("near")
    topk = params.get("topk")
    block_size = params.get("block_size")
    v_head_dim = params.get("v_head_dim")
    is_quant = params.get("is_quant")
    has_smooth = params.get("is_smooth")

    softmax_scale = q_dim ** -0.5
    slc_s_max = topk * slc_block_size

    # kv cache actual_seq
    kv_cache_actual_seq_p = params.get("kv_cache_actual_seq")
    if isinstance(kv_cache_actual_seq_p, int):
        kv_cache_actual_seq = [kv_cache_actual_seq_p] * b
    elif isinstance(kv_cache_actual_seq_p, list):
        if len(kv_cache_actual_seq_p) == b:
            kv_cache_actual_seq = kv_cache_actual_seq_p
        else:
            raise RuntimeError("unsupported this kv_cache_actual_seq")
    else:
        raise RuntimeError("unsupported kv_cache_actual_seq data type")

    # 1. 设置shape
    # gen kv_slc
    shape_topk_indices = [b, s, topk - front - near]

    # gen slc atten
    slc_q_shape = [b, s, n1, q_dim]
    slc_k_shape = [b, s, n2, slc_s_max, k_dim]
    slc_v_shape = [b, s, n2, slc_s_max, v_dim]
    slc_atten_out_shape = [b, s, n1, v_dim]
    # gen gated_score

    x_shape = [b, s, h]
    gate_sim_w1_shape = [h, n1 * 3]
    gate_w1_shape = [h, h * 4]
    gate_w2_shape = [h * 4, n1 * 3]

    # gen attn
    cmp_atten_shape = [b, s, n1, v_dim]
    sel_atten_shape = [b, s, n1, v_dim]
    win_atten_shape = [b, s, n1, v_dim]
    gating_score_shape = [b, s, n1, 3]

    np.random.seed(int(time.time()))

    # 2. 生成数据
    kv_nope_cache, k_rope_cache, block_table = gen_kv_cache(params, kv_cache_actual_seq, dtype, output_dir) # 生成kvcache
    # gen kv_slc
    s_slc = 128 # TODO: 中间输出，后续topk子图拼接后，需要删除topk_indices的生成
    topk_indices = gen_uniform_data(shape_topk_indices, 0, s_slc, dtype=np.int32)
    topk_tensor_shape = np.zeros([b, s], dtype=np.int32)
    for batchIdx in range(b):
        for seqIdx in range(s):
            topk_tensor_shape[batchIdx][seqIdx] = s_slc

    # gen slc attn
    slc_q_bsnd = gen_uniform_data(slc_q_shape, -1, 1, dtype)
    slc_k_bsnd = gen_uniform_data(slc_k_shape, -1, 1, dtype)
    slc_v_bsnd = slc_k_bsnd[:, :, :, :, :kv_lora_rank]

    # gen gated_score
    x = gen_uniform_data(x_shape, -1, 1, dtype)
    gate_sim_w1 = gen_uniform_data(gate_sim_w1_shape, -0.1, 0.1, dtype)
    gate_w1 = gen_uniform_data(gate_w1_shape, -0.1, 0.1, dtype)
    gate_w2 = gen_uniform_data(gate_w2_shape, -0.1, 0.1, dtype)

    # gen attn
    cmp_atten = np.random.uniform(-1, 1, cmp_atten_shape).astype(dtype)
    # slc_atten = np.random.uniform(-1, 1, sel_atten_shape).astype(dtype)
    win_atten = np.random.uniform(-1, 1, win_atten_shape).astype(dtype)

    # post
    post_params = [b, n1, s, h, kv_lora_rank, v_head_dim]
    w_uv, w_o, w_o_scale, smooth_wo = gen_post_input_data(output_dir, post_params, dtype, is_quant, has_smooth, is_nz)


    # 3. 计算 & dump file
    # kv compression
    # gen_kv_compression()

    # cmp atten
    # gen_cmp_attn()

    # win atten
    # gen_win_attn()

    # gen kv_slc
    print("========== gen kv_slc ==============")
    compute_input_params = [block_size, n2, front, near, topk, slc_block_size]
    k_slc_out, v_slc_out, kv_slc_actual_seqs = kv_slc_compute(compute_input_params, topk_indices, topk_tensor_shape, kv_nope_cache, k_rope_cache, block_table, kv_cache_actual_seq)
    dump_gen_kv_slc_file(topk_indices, topk_tensor_shape, k_slc_out, v_slc_out, kv_slc_actual_seqs, dtype, output_dir)

    # slc atten
    print("========== gen slc_attn ==============")
    input_params = [b, s, n1, n2, kv_lora_rank, rope_dim, slc_s_max, slc_s_max]
    k_slc = np.reshape(k_slc_out, slc_k_shape) # [b*s*n2*slc_s_max, k_dim] -> [b, s, n2, slc_s_max, k_dim]
    v_slc = np.reshape(v_slc_out, slc_v_shape) # [b*s*n2*slc_s_max, v_dim] -> [b, s, n2, slc_s_max, v_dim]
    slc_atten = compute_attention(slc_q_bsnd, k_slc, v_slc, kv_slc_actual_seqs, softmax_scale, slc_atten_out_shape) # 输出fp64
    dump_slc_atten_file(slc_q_bsnd, k_slc, v_slc, kv_slc_actual_seqs, kv_lora_rank, dtype, output_dir, slc_atten, input_params)

    # gen gated_score
    print("========== gen gated_score ==============")
    gating_score, _, _ = gen_gated_score(x.astype(np.float64), gate_sim_w1.astype(np.float64),
        gate_w1.astype(np.float64), gate_w2.astype(np.float64), output_dir, mode='standard') # 升精度运算
    dump_gated_score_file(x, gate_sim_w1, gate_w1, gate_w2, gating_score, dtype, output_dir)

    # gen atten
    print("========== gen attn ==============")
    attention_out = gen_atten_golden_data(cmp_atten, slc_atten, win_atten, gating_score, dtype)
    dump_gen_atten_file(cmp_atten, slc_atten, win_atten, attention_out, dtype, output_dir)
    print("========== attention_out: ", attention_out.shape, attention_out.dtype)

    # post
    print("========== gen post output ==============")
    post_inputs = {"dtype": dtype, "is_quant": is_quant, "has_smooth": has_smooth}
    post_inputs["x"] = attention_out
    post_inputs["w_uv"] = w_uv
    post_inputs["w_o"] = w_o
    if is_quant:
        post_inputs["w_o_scale"] = w_o_scale
        if has_smooth:
            post_inputs["smooth_wo"] = smooth_wo
    post_out = post_compute(post_inputs)

    # dump output to file
    output_path = Path(output_dir, 'golden_output.bin')
    post_out.tofile(output_path)

    return True


def nsa_entry(dtypes, bs1s2h, quant_smooth, output_dir: Path):
    b, s1, s2, h = bs1s2h
    is_quant, is_smooth = quant_smooth
    kv_lora_rank = 512
    rope_dim = 64
    q_dim = kv_lora_rank + rope_dim
    k_dim = kv_lora_rank + rope_dim
    v_dim = kv_lora_rank
    topk = 16
    slc_block_size = 64
    v_head_dim = 128

    params = {
        "b": b,
        "s": s1,
        "s2": s2,
        "n1": 128,
        "n2": 1,
        "h": h,
        "q_lora_rank": 1536,
        "kv_lora_rank": kv_lora_rank,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "rope_dim": rope_dim,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "cmp_block_size": 32,
        "cmp_stride": 16,
        "slc_block_size": slc_block_size,
        "front": 1,
        "near": 2,
        "topk": topk,
        "block_size": 128,
        "kv_cache_actual_seq": s2,
        "v_head_dim": v_head_dim,
        "is_quant": is_quant,
        "is_smooth": is_smooth,
    }
    gen_nsa_golden(params, dtypes, output_dir)

    # 将变化的参数保存到文件中，供测试用例直接读取
    input_params = [params.get("b"), params.get("s"), params.get("s2"), params.get("n1"), params.get("n2")]
    input_params.append(1 if is_quant else 0)
    input_params.append(1 if is_smooth else 0)
    dump_file(input_params, Path(output_dir, 'input_params.bin'), np.int32)


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicNSATest.subgraph_4_5_6_fp16_b16",
        "DynamicNSATest.subgraph_4_5_6_fp16_b16_quant",
    ]
)
def gen_nsa_v1_func(case_name: str, output: Path) -> bool:
    input_path = Path(output, 'x.bin')
    complete = input_path.exists()
    if complete:
        file_mod_time = input_path.stat().st_mtime
        # 获取当前时间（Unix 时间戳）
        current_time = time.time()
        # 判断文件的修改时间是否超过1小时（3600秒）
        if current_time - file_mod_time > 3600:
            logging.info("文件的修改时间超过1小时，重新生成文件...")
            complete = False
        else:
            logging.info("文件的修改时间在1小时内，无需重新生成。")

    complete = False # TODO: del complete
    if complete:
        logging.info("Case(%s), Golden data exits. cache catch", case_name)
    else:
        if case_name == "DynamicNSATest.subgraph_4_5_6_fp16_b16": # gen_slc_attn + gen_gated_score + gen_attn
            nsa_entry((np.float16, np.float16), (16, 1, 8192, 7168), (False, False), output)
        elif case_name == "DynamicNSATest.subgraph_4_5_6_fp16_b16_quant": # quant
            nsa_entry((np.float16, np.float16), (16, 1, 8192, 7168), (True, True), output)
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
        "DynamicNSATest.subgraph_4_5_6_fp16_b16",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output_dir: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ret = gen_nsa_v1_func(case_name=cs, output=output_dir)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
