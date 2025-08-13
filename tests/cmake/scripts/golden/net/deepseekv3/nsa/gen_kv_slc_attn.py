
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
from golden.net.deepseekv3.mla.mla_prolog_golden_v2 import gen_prolog_input_data, mla_prolog_compute


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
    # shape_kv_nope_cache = [block_num * block_size, n2 * kv_lora_rank]
    # shape_k_rope_cache = [block_num * block_size, n2 * rope_dim]

    # kv_nope_cache = gen_uniform_data(shape_kv_nope_cache, -1, 1, dtype)
    # k_rope_cache = gen_uniform_data(shape_k_rope_cache, -1, 1, dtype)

    # kv_nope_cache_path = Path(output_dir, 'kv_nope_cache.bin')
    # kr_cache_path = Path(output_dir, 'k_rope_cache.bin')
    block_table_path = Path(output_dir, 'block_table.bin')
    kv_cache_actual_seq_path = Path(output_dir, 'kv_cache_actual_seq_len.bin')

    # dump_file(kv_nope_cache, kv_nope_cache_path, dtype)
    # dump_file(k_rope_cache, kr_cache_path, dtype)
    dump_file(block_table, block_table_path, np.int32)
    dump_file(actual_seq_list, kv_cache_actual_seq_path, np.int32)

    # return kv_nope_cache, k_rope_cache, block_table
    return block_table, block_num


def dump_gen_kv_slc_file(topk_indices, topk_tensor_shape, kv_slc_out, kr_slc_out, kv_slc_actual_seqs, kv_cache_actual_seq, dtype, output_dir):
    topk_tensor_path = Path(output_dir, 'topk_tensor.bin')
    topk_tensor_shape_path = Path(output_dir, 'topk_tensor_shape.bin')
    kv_slc_out_path = Path(output_dir, 'kv_slc_out.bin')
    kr_slc_out_path = Path(output_dir, 'kr_slc_out.bin')
    kv_slc_actual_seqs_path = Path(output_dir, 'kv_slc_actual_seqs.bin')
    kv_cache_actual_seq_path = Path(output_dir, 'kv_cache_actual_seq_len.bin')

    dump_file(topk_indices, topk_tensor_path, np.int32)
    dump_file(topk_tensor_shape, topk_tensor_shape_path, np.int32)
    dump_file(kv_slc_out, kv_slc_out_path, dtype)
    dump_file(kr_slc_out, kr_slc_out_path, dtype)
    dump_file(kv_slc_actual_seqs, kv_slc_actual_seqs_path, np.int32)
    dump_file(kv_cache_actual_seq, kv_cache_actual_seq_path, np.int32)


def dump_slc_atten_file(q_bsnd, k_bsnd, v_bsnd, actual_seq, kv_lora_rank, dtype, output_dir, atten_out, input_params):
    # data split to [nope + rope]
    q_nope = q_bsnd[:, :, :, :kv_lora_rank]
    q_rope = q_bsnd[:, :, :, kv_lora_rank:]

    q_nope_path = Path(output_dir, 'q_nope.bin')
    q_rope_path = Path(output_dir, 'q_rope.bin')
    k_slc_path = Path(output_dir, 'k_slc.bin')
    v_slc_path = Path(output_dir, 'v_slc.bin')
    actual_seq_path = Path(output_dir, 'slc_actual_seq.bin')

    dump_file(q_nope, q_nope_path, dtype)
    dump_file(q_rope, q_rope_path, dtype)
    dump_file(k_bsnd, k_slc_path, dtype)
    dump_file(v_bsnd, v_slc_path, dtype)
    dump_file(actual_seq, actual_seq_path, np.int32)


def gen_kv_slc_attn_golden(params, dtypes, output_dir: Path, is_nz=False):
    '''
    gen_kv_slc_atten, 其中包括: gen_kv_slc及slc_attn
    '''
    dtype, w_dtype = dtypes
    logging.debug(f"gen_kv_slc_attn_golden  dtype:{dtype}, w_dtype:{w_dtype}")
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
    win_size = params.get("win_size")
    epsilon = params.get("epsilon")
    cache_mode = params.get("cache_mode")
    q_lora_rank = params.get("q_lora_rank")
    qk_nope_head_dim = params.get("qk_nope_head_dim")
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
    skv_max = max(kv_cache_actual_seq)

    # 1. 设置shape
    # gen kv_slc
    shape_topk_indices = [b, s, topk - front - near]

    # gen slc atten
    slc_k_shape = [b, s, n2, slc_s_max, k_dim]
    slc_v_shape = [b, s, n2, slc_s_max, v_dim]
    slc_atten_out_shape = [b, s, n1, v_dim]

    np.random.seed(int(time.time()))

    # 2. 生成数据
    # mla_prolog
    block_table, block_num = gen_kv_cache(params, kv_cache_actual_seq, dtype, output_dir) # 生成 block_table

    prolog_params = {
        "b": b,
        "s": s,
        "s2": s2,
        "h": h,
        "num_heads": n1,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": rope_dim,
        "kv_lora_rank": kv_lora_rank,
        "v_head_dim": v_head_dim,
        "block_num": block_num,
        "block_table": block_table,
        "skv_max": skv_max,
    }
    x, wDq, wUqQr, smooth_cq, w_qb_scale, wDkvKr, wUk, gamma_cq, gamma_ckv, cos, sin, kv_len, kv_cache, kr_cache = \
        gen_prolog_input_data(prolog_params, [dtype, dtype], epsilon, output_dir, is_quant, is_nz, has_smooth,
                              block_size, cache_mode)

    s_slc = (((s2-32)//16+1)+3)//4
    topk_indices = gen_uniform_data(shape_topk_indices, 0, s_slc, dtype=np.int32)
    topk_tensor_shape = np.zeros([b, s], dtype=np.int32)
    for batchIdx in range(b):
        for seqIdx in range(s):
            topk_tensor_shape[batchIdx][seqIdx] = s_slc

    # 3. 计算 & dump file
    # mla_prolog
    prolog_inputs = {"dtype": dtype, "is_quant": is_quant, "has_smooth": has_smooth}
    prolog_inputs["cache_mode"] = cache_mode
    prolog_inputs["gamma_cq"] = gamma_cq
    prolog_inputs["gamma_ckv"] = gamma_ckv
    prolog_inputs["epsilon"] = epsilon
    prolog_inputs["x"] = x
    prolog_inputs["wDq"] = wDq
    prolog_inputs["wUqQr"] = wUqQr
    prolog_inputs["wUk"] = wUk
    prolog_inputs["wDkvKr"] = wDkvKr
    prolog_inputs["cos"] = cos
    prolog_inputs["sin"] = sin
    prolog_inputs["kv_cache"] = kv_cache
    prolog_inputs["kr_cache"] = kr_cache
    prolog_inputs["cache_index"] = kv_len
    if is_quant:
        prolog_inputs["w_qb_scale"] = w_qb_scale
        if has_smooth:
            prolog_inputs["smooth_cq"] = smooth_cq
    # q_out: [b, s, n1, kv_lora_rank], q_rope_out: [b, s, n1, rope_dim]
    # kv_cache_out: [block_num, block_size, n2, kv_lora_rank], kr_cache_out: [block_num, block_size, n2, rope_dim]
    q_out, q_rope_out, kv_cache_out, kr_cache_out = mla_prolog_compute(prolog_inputs)

    # reshape
    kv_nope_cache = kv_cache_out.reshape([block_num * block_size, n2 * kv_lora_rank])
    k_rope_cache = kr_cache_out.reshape([block_num * block_size, n2 * rope_dim])
    dump_file(kv_nope_cache, Path(output_dir, 'kv_nope_cache.bin'), dtype)
    dump_file(k_rope_cache, Path(output_dir, 'k_rope_cache.bin'), dtype)

    q_bsnd = np.concatenate([q_out, q_rope_out], axis=-1)  # [b, s, n1, kv_lora_rank + rope_dim]

    # gen kv_slc
    print("========== gen kv_slc ==============")
    compute_input_params = [block_size, n2, front, near, topk, slc_block_size]
    k_slc_out, v_slc_out, kv_slc_actual_seqs = kv_slc_compute(compute_input_params, topk_indices, topk_tensor_shape, kv_nope_cache, k_rope_cache, block_table, kv_cache_actual_seq)
    dump_gen_kv_slc_file(topk_indices, topk_tensor_shape, k_slc_out, v_slc_out, kv_slc_actual_seqs, kv_cache_actual_seq, dtype, output_dir)

    # slc atten
    print("========== gen slc_attn ==============")
    input_params = [b, s, n1, n2, kv_lora_rank, rope_dim, slc_s_max, slc_s_max]
    k_slc = np.reshape(k_slc_out, slc_k_shape) # [b*s*n2*slc_s_max, k_dim] -> [b, s, n2, slc_s_max, k_dim]
    v_slc = np.reshape(v_slc_out, slc_v_shape) # [b*s*n2*slc_s_max, v_dim] -> [b, s, n2, slc_s_max, v_dim]
    slc_atten = compute_attention(q_bsnd, k_slc, v_slc, kv_slc_actual_seqs, softmax_scale, slc_atten_out_shape) # 输出fp64
    dump_slc_atten_file(q_bsnd, k_slc, v_slc, kv_slc_actual_seqs, kv_lora_rank, dtype, output_dir, slc_atten, input_params)

    dump_file(slc_atten, Path(output_dir, 'slc_attn_out.bin'), np.float32)

    return True


def gen_kv_slc_attn_entry(dtypes, bs1s2h, quant_smooth, output_dir: Path):
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
    epsilon = 1e-5
    cache_mode = "PA_BSND"

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
        "win_size": 512,
        "kv_cache_actual_seq": s2,
        "epsilon": epsilon,
        "cache_mode": cache_mode,
        "v_head_dim": v_head_dim,
        "is_quant": is_quant,
        "is_smooth": is_smooth,
    }
    gen_kv_slc_attn_golden(params, dtypes, output_dir)

    # 将变化的参数保存到文件中，供测试用例直接读取
    input_params = [params.get("b"), params.get("s"), params.get("s2"), params.get("n1"), params.get("n2")]
    input_params.append(1 if is_quant else 0)
    input_params.append(1 if is_smooth else 0)
    dump_file(input_params, Path(output_dir, 'input_params.bin'), np.int32)


@GoldenRegister.reg_golden_func(
    case_names=[
        "DynamicKvSATest.kv_slc_attn_b48_s1_fp16",
        "DynamicKvSATest.kv_slc_attn_b32_s2_bf16",
    ]
)
def gen_kv_slc_attn_func(case_name: str, output: Path) -> bool:
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
        if case_name == "DynamicKvSATest.kv_slc_attn_b48_s1_fp16":
            gen_kv_slc_attn_entry((np.float16, np.float16), (48, 1, 8192, 7168), (False, False), output)
        elif case_name == "DynamicKvSATest.kv_slc_attn_b32_s2_bf16":
            gen_kv_slc_attn_entry((bfloat16, bfloat16), (32, 2, 32768, 7168), (False, False), output)
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
        "DynamicKvSATest.kv_slc_attn_b48_s1_fp16",
    ]
    # 函数调用
    ret: bool = True
    for cs in case_name_list:
        output_dir: Path = Path(g_src_root, "build/tests/st/golden", cs).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ret = gen_kv_slc_attn_func(case_name=cs, output=output_dir)
    return ret


if __name__ == "__main__":
    exit(0 if main() else 1)
