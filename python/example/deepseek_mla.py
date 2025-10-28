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
"""
from dataclasses import dataclass
from typing import (Union, List)
import math
import pto

NUM_2 = 2
NUM_4 = 4
NUM_16 = 16
NUM_32 = 32
NUM_64 = 64
NUM_128 = 128
NUM_256 = 256
NUM_384 = 384
NUM_512 = 512
NUM_576 = 576
NUM_1024 = 1024
NUM_1536 = 1536
NUM_7168 = 7168

KEY_ONLY_HOST_COMPILE = "ONLY_HOST_COMPILE"

g_deepseek_config: dict[str, Union[bool, int, float, str]] = {}
g_deepseek_config = {
    "architectures": "DeepseekForCausalLM",
    "attention_bias": False,
    "attentionDropout": 0,
    "AutoConfig": "DeepseekConfig",
    "AutoModel": "DeepseekModel",
    "AutoModelForCausalLM": "DeepseekForCausalLM",
    "auxLossAlpha": 0.001,
    "bosTokenId": 100000,
    "eosTokenId": 100001,
    "epSize": 1,
    "firstKDenseReplace": 3,
    "hiddenAct": "silu",
    "hiddenSize": 256, # 7168
    "initializerRange": 0.02,
    "intermediateSize": 18432,
    "kvLoraRank": 512,
    "lmHead": False,
    "maxPositionEmbeddings": 4096,
    "modelType": "deepseek_v3",
    "moeIntermediateSize": 2048,
    "moeLayerFreq": 1,
    "nGroup": 8,
    "nRoutedExperts": 256,
    "nSharedExperts": 1,
    "normTopkProb": True,
    "numAttentionHeads": 2, # 128
    "numExpertsPerTok": 8,
    "numHiddenLayers": 61,
    "numKeyValueHeads": 128,
    "pretrainingTp": 1,
    "qLoraRank": 512, # 1536
    "qkNopeHeadDim": 128,
    "qkRopeHeadDim": 64,
    "rmHead": False,
    "rmsNormEps": 1e-06,
    "ropeScaling": 1,
    "ropeTheta": 10000,
    "routedScalingFactor": 2.5,
    "scoringFunc": "sigmoid",
    "seqAux": True,
    "tieWordEmbeddings": False,
    "topkGroup": 4,
    "topkMethod": "noaux_tc",
    "torchDtype": "bfloat16",
    "transformersVersion": "4.33.1",
    "useCache": True,
    "vHeadDim": 128,
    "vocabSize": 129280,
    "fp8Format": "e4m3",
    "initFp8Params": True
}


@dataclass
class AttentionW:
    q_a_proj_w: pto.tensor = pto.tensor()
    q_b_proj_w: pto.tensor = pto.tensor()
    q_b_proj_w_scale: pto.tensor = pto.tensor()
    kv_a_proj_with_mqa_w: pto.tensor = pto.tensor()
    kv_b_proj_wk: pto.tensor = pto.tensor()
    kv_b_proj_wv: pto.tensor = pto.tensor()
    o_proj_w: pto.tensor = pto.tensor()


class DeepSeekAttention:
    def __init__(self, config: dict, aw: AttentionW, in_layer_idx: int):
        self.layer_idx = in_layer_idx
        self.attention_dropout = config["attentionDropout"]
        self.hidden_size = config["hiddenSize"]
        self.num_heads = config["numAttentionHeads"]
        self.max_position_embeddings = config["maxPositionEmbeddings"]
        self.rope_theta = config["ropeTheta"]
        self.q_lora_rank = config["qLoraRank"]
        self.qk_rope_head_dim = config["qkRopeHeadDim"]
        self.kv_lora_rank = config["kvLoraRank"]
        self.v_head_dim = config["vHeadDim"]
        self.qk_nope_head_dim = config["qkNopeHeadDim"]
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.is_causal = True

        self.q_a_proj_w = aw.q_a_proj_w
        self.q_b_proj_w = aw.q_b_proj_w
        self.q_b_proj_w_scale = aw.q_b_proj_w_scale
        self.kv_a_proj_with_mqa_w = aw.kv_a_proj_with_mqa_w
        self.kv_b_proj_wk = aw.kv_b_proj_wk
        self.kv_b_proj_wv = aw.kv_b_proj_wv
        self.o_proj_w = aw.o_proj_w
        self.softmax_scale = (1.0 / math.sqrt(self.q_head_dim))

        if config["ropeScaling"] == 1:
            factor = 40
            m_scale = 1.0
            m_scale_all_dim = 1.0
            value_point_one = 0.1
            if m_scale_all_dim > 1:
                m_scale = value_point_one * m_scale * math.log(factor) + 1.0
            self.softmax_scale = self.softmax_scale * m_scale * m_scale


    def attention(self, q: pto.tensor, kv: pto.tensor, atten_mask: pto.tensor) -> pto.tensor:
        b = q.shape[0]
        n2 = kv.shape[1]
        s1 = q.shape[2]
        s2 = kv.shape[2]
        kv_lora_rank_v = g_deepseek_config["kvLoraRank"]
        d_type = q.dtype
        pto.set_cube_tile_shapes([min(NUM_128, s1), min(NUM_128, s1)], [NUM_64, NUM_64], [NUM_128, NUM_128])

        qk = pto.matmul(q, kv, d_type, a_trans=False, b_trans=True)
        pto.set_vec_tile_shapes(1, 1, NUM_128, NUM_64)
        qk_fp32 = pto.cast(qk, pto.DT_FP32)
        qk_fp32 = pto.mul_s(qk_fp32, pto.element(pto.DT_FP32, self.softmax_scale))
        atten_mask_fp32 = pto.cast(atten_mask, pto.DT_FP32)
        qk_fp32 = pto.add(qk_fp32, atten_mask_fp32)
        qk_16 = pto.cast(qk_fp32, d_type)
        softmax = pto.softmax(qk_16) # [b, n, s1, s2]
        # no drop
        v = pto.view(kv, [b, n2, s2, kv_lora_rank_v], [0, 0, 0, 0])
        pto.set_cube_tile_shapes([min(NUM_128, s1), min(NUM_128, s1)], [NUM_64, NUM_64], [NUM_128, NUM_128])

        atten_res = pto.matmul(softmax, v, d_type)
        return atten_res


    def attention_post(self, atten_res: pto.tensor) -> pto.tensor:
        b = atten_res.shape[0]
        n = atten_res.shape[1]
        s = atten_res.shape[2]
        bs = b * s
        d_type = atten_res.dtype

        pto.set_vec_tile_shapes(1, 1, 1, NUM_512)
        atten_res0 = pto.transpose(atten_res, [1, 2])
        pto.set_vec_tile_shapes(1, 1, NUM_128, NUM_64)
        atten_res1 = pto.reshape(atten_res0, [b * s, n, self.kv_lora_rank])
        pto.set_vec_tile_shapes(1, 1, NUM_512)
        atten_res2 = pto.transpose(atten_res1, [0, 1])
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        pto.set_cube_tile_shapes([min(NUM_128, bs), min(NUM_128, bs)], [NUM_128, NUM_128], [NUM_128, NUM_128])
        pto.set_vec_tile_shapes(NUM_128, NUM_64)
        mm7_res = pto.matmul(atten_res2, self.kv_b_proj_wv, d_type)

        pto.set_vec_tile_shapes(1, 1, NUM_128)
        mm7_res1 = pto.transpose(mm7_res, [0, 1])
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        mm7_res2 = pto.reshape(mm7_res1, [b, s, n * self.v_head_dim])

        pto.set_vec_tile_shapes(NUM_128, NUM_64)
        atten_out_w = pto.unsqueeze(self.o_proj_w, 0)
        pto.set_cube_tile_shapes([min(NUM_128, s), min(NUM_128, s)], [NUM_128, NUM_128], [NUM_128, NUM_128])
        pto.set_vec_tile_shapes(NUM_128, NUM_64)
        atten_output = pto.matmul(mm7_res2, atten_out_w, d_type)

        return atten_output


    def attention_post2(self, atten_res: pto.tensor) -> pto.tensor:
        b = atten_res.shape[0]
        n = atten_res.shape[1]
        s = atten_res.shape[2]
        bs = b * s
        h = self.o_proj_w.shape[1]
        d_type = atten_res.dtype

        pto.set_vec_tile_shapes(NUM_16, NUM_16, 1, NUM_128)
        atten_res0 = pto.transpose(atten_res, [1, 2])
        pto.set_vec_tile_shapes(NUM_16, 1, NUM_16, NUM_128)
        atten_res1 = pto.reshape(atten_res0, [b * s, n, self.kv_lora_rank])
        pto.set_vec_tile_shapes(NUM_16, NUM_16, NUM_128)
        atten_res2 = pto.transpose(atten_res1, [0, 1])
        pto.set_cube_tile_shapes([min(NUM_128, bs), min(NUM_128, bs)], [NUM_128, NUM_128],
                                [min(NUM_128, h), min(NUM_128, h)])
        mm7_res = pto.matmul(atten_res2, self.kv_b_proj_wv, d_type)

        pto.set_vec_tile_shapes(NUM_16, NUM_16, NUM_128)
        mm7_res1 = pto.transpose(mm7_res, [0, 1])
        pto.set_vec_tile_shapes(NUM_16, NUM_16, NUM_128)
        mm7_res2 = pto.reshape(mm7_res1, [b, s, n * self.v_head_dim])

        pto.set_vec_tile_shapes(NUM_128, min(NUM_256, h))
        atten_out_w = pto.unsqueeze(self.o_proj_w, 0)
        pto.set_cube_tile_shapes([min(NUM_128, s), min(NUM_128, s)], [NUM_128, NUM_128],
                                [min(NUM_128, h), min(NUM_128, h)])
        atten_output = pto.matmul(mm7_res2, atten_out_w, d_type)

        return atten_output


    def qkv_pre(self, hidden_states: pto.tensor) -> List[pto.tensor]:
        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        d_type = hidden_states.dtype

        pto.set_vec_tile_shapes(NUM_128, NUM_64)
        q_a_proj_w1 = pto.unsqueeze(self.q_a_proj_w, 0)
        q_b_proj_w1 = pto.unsqueeze(self.q_b_proj_w, 0)
        kv_a_proj_with_mqa_w1 = pto.unsqueeze(self.kv_a_proj_with_mqa_w, 0)

        pto.set_cube_tile_shapes([min(NUM_128, s), min(NUM_128, s)], [NUM_128, NUM_128], [NUM_64, NUM_64])
        pto.set_vec_tile_shapes(NUM_128, NUM_64) # for Assemble
        q_a_proj = pto.matmul(hidden_states, q_a_proj_w1, d_type) # bf16

        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        q_a_layer_norm = pto.rms_norm(q_a_proj)

        pto.set_cube_tile_shapes([min(NUM_128, s), min(NUM_128, s)], [NUM_128, NUM_128], [NUM_64, NUM_64])
        pto.set_vec_tile_shapes(NUM_128, NUM_64) # for Assemble
        q = pto.matmul(q_b_proj_w1, q_a_layer_norm, d_type)

        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        q2 = pto.reshape(q, [b, s, self.num_heads, self.q_head_dim])

        pto.set_cube_tile_shapes([min(NUM_128, s), min(NUM_128, s)], [NUM_128, NUM_128], [NUM_64, NUM_64])
        pto.set_vec_tile_shapes(NUM_128, NUM_64) # for Assemble
        compressd_kv = pto.matmul(hidden_states, kv_a_proj_with_mqa_w1, d_type)

        return [q2, compressd_kv]


    def qkv_pre_cv(self, hidden_states: pto.tensor) -> List[pto.tensor]:
        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        d_type = hidden_states.dtype

        pto.set_vec_tile_shapes(NUM_128, NUM_64)
        q_a_proj_w1 = pto.unsqueeze(self.q_a_proj_w, 0) # [NUM_256, NUM_512]
        q_b_proj_w1 = pto.unsqueeze(self.q_b_proj_w, 0) # [NUM_512, 2 * 192]
        kv_a_proj_with_mqa_w1 = pto.unsqueeze(self.kv_a_proj_with_mqa_w, 0) # [NUM_256, 576]

        pto.set_cube_tile_shapes([min(NUM_128, s), min(NUM_128, s)], [NUM_128, NUM_128], [NUM_64, NUM_64])
        q_a_proj = pto.matmul(hidden_states, q_a_proj_w1, d_type) # bf16 2_1_512

        pto.set_vec_tile_shapes(NUM_2, 1, NUM_512)
        q_a_layer_norm = pto.rms_norm(q_a_proj)

        pto.set_cube_tile_shapes([min(NUM_128, s), min(NUM_128, s)], [NUM_128, NUM_128], [NUM_64, NUM_64])
        q = pto.matmul(q_a_layer_norm, q_b_proj_w1, d_type)

        pto.set_vec_tile_shapes(NUM_2, 1, NUM_384)
        q2 = pto.reshape(q, [b, s, self.num_heads, self.q_head_dim])

        pto.set_cube_tile_shapes([min(NUM_128, s), min(NUM_128, s)], [NUM_128, NUM_128], [NUM_64, NUM_64])
        compressd_kv = pto.matmul(hidden_states, kv_a_proj_with_mqa_w1, d_type) # bf16

        return [q2, compressd_kv]


    def qkv_pre2(self, hidden_states: pto.tensor, is_quant: bool = False) -> List[pto.tensor]:
        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        h = hidden_states.shape[2]
        bs = b * s

        d_type = hidden_states.dtype
        d_type_quant_out = pto.DT_INT32 if is_quant else d_type
        qkv_pre2_res = []

        input_data = pto.reshape(hidden_states, [bs, h])

        c0 = NUM_16
        m = (min(NUM_32, bs) + c0 - 1) // c0 * c0
        tile_m = min(NUM_16, m)
        pto.set_cube_tile_shapes([tile_m, tile_m], [NUM_256, NUM_256], [NUM_128, NUM_128])
        q_a_proj = pto.matmul(input_data, self.q_a_proj_w, d_type, a_trans=False, b_trans=False)

        pto.set_vec_tile_shapes(min(NUM_16, bs), NUM_128)
        q_a_proj_norm = pto.rms_norm(q_a_proj)

        q_a_proj_norm_scale_dequant = pto.tensor()
        if is_quant:
            q_a_proj_norm_quant_res = pto.quant(q_a_proj_norm)
            q_a_proj_norm, q_a_proj_norm_scale_dequant = q_a_proj_norm_quant_res
            pto.set_cube_tile_shapes([tile_m, tile_m], [NUM_256, NUM_256], [NUM_256, NUM_256])
        else:
            pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_64, NUM_64])
        q = pto.matmul(q_a_proj_norm, self.q_b_proj_w, d_type_quant_out) # bf16 quant A8W8O32 -> bf16
        qkv_pre2_res.append(q)

        pto.set_cube_tile_shapes([m, m], [NUM_256, NUM_256], [NUM_64, NUM_64])
        compressd_kv = pto.matmul(input_data, self.kv_a_proj_with_mqa_w, d_type, a_trans=False, b_trans=False) # bf16
        compressed_kv_res = pto.reshape(compressd_kv, [b, s, self.kv_lora_rank + self.qk_rope_head_dim])
        qkv_pre2_res.append(compressed_kv_res)

        if is_quant:
            qkv_pre2_res.append(q_a_proj_norm_scale_dequant)

        return qkv_pre2_res


    def qkv_pre_fp32(self, hidden_states: pto.tensor) -> List[pto.tensor]:
        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        h = hidden_states.shape[2]
        bs = b * s
        d_type = hidden_states.dtype

        input_data = pto.reshape(hidden_states, [bs, h]) # [b, s, h] -> [b * s, h]

        pto.set_cube_tile_shapes([min(NUM_64, bs), min(NUM_64, bs)], [NUM_256, NUM_256], [NUM_128, NUM_128])
        q_a_proj_fp32 = pto.matmul(input_data, self.q_a_proj_w, pto.DT_FP32, a_trans=False, b_trans=False)

        pto.set_vec_tile_shapes(NUM_32, NUM_32)
        q_a_proj_norm_fp32 = pto.rms_norm(q_a_proj_fp32)

        pto.set_vec_tile_shapes(NUM_32, NUM_128)
        q_a_proj_norm = pto.cast(q_a_proj_norm_fp32, d_type)

        pto.set_cube_tile_shapes([min(NUM_64, bs), min(NUM_64, bs)], [NUM_256, NUM_256], [NUM_64, NUM_64])
        q_fp32 = pto.matmul(q_a_proj_norm, self.q_b_proj_w, pto.DT_FP32, a_trans=False, b_trans=False)
        q_res = pto.reshape(q_fp32, [b, s, self.num_heads, self.q_head_dim])

        pto.set_cube_tile_shapes([min(NUM_64, bs), min(NUM_64, bs)], [NUM_256, NUM_256], [NUM_64, NUM_64])
        compressed_kv_fp32 = pto.matmul(input_data, self.kv_a_proj_with_mqa_w, pto.DT_FP32)
        compressed_kv_res = pto.reshape(compressed_kv_fp32, [b, s, self.kv_lora_rank + self.qk_rope_head_dim])

        return [q_res, compressed_kv_res]


    def forward(self, **kwargs) -> pto.tensor:
        hidden_states = kwargs.get("hidden_states")
        atten_mask = kwargs.get("atten_mask")
        position_ids = kwargs.get("position_ids")
        cos = kwargs.get("cos")
        sin = kwargs.get("sin")
        kv_len = kwargs.get("kv_len")
        past_key_states = kwargs.get("past_key_states")
        rope_tile_shape_config = kwargs.get("rope_tile_shape_config")
        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        bs = b * s
        d_type = hidden_states.dtype

        qkv = self.qkv_pre(hidden_states)
        q = qkv[0]
        compressd_kv = qkv[1]

        q_nope = pto.view(q, [b, s, self.num_heads, self.qk_nope_head_dim], [0, 0, 0, 0])
        q_pe = pto.view(q, [b, s, self.num_heads, self.qk_rope_head_dim], [0, 0, 0, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(1, 1, 1, NUM_64)
        q_pe = pto.transpose(q_pe, [1, 2])
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)

        k_pe = pto.view(compressd_kv, [b, s, self.qk_rope_head_dim], [0, 0, self.kv_lora_rank]) # (b, s, qkRopeHeadDim)
        compressd_kv = pto.view(compressd_kv, [b, s, self.kv_lora_rank], [0, 0, 0]) # (b, s, kvLoraRank)
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        k_pe = pto.reshape(k_pe, [b, 1, s, self.qk_rope_head_dim]) # setTileShapes 4维

        pto.set_vec_tile_shapes(1, NUM_128, 1, NUM_64) # SetVecTileShapes(1, 1, NUM_128, NUM_64)
        q_nope1 = pto.reshape(q_nope, [b * s, self.num_heads, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(1, 1, NUM_128)
        q_nope2 = pto.transpose(q_nope1, [0, 1]) # (n, bs, d)
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)

        pto.set_cube_tile_shapes([min(NUM_128, bs), min(NUM_128, bs)], [NUM_128, NUM_128], [NUM_128, NUM_128])
        pto.set_vec_tile_shapes(NUM_128, NUM_64) # for Assemble
        q_nope_new = pto.matmul(q_nope2, self.kv_b_proj_wk, d_type)
        pto.set_vec_tile_shapes(1, 1, NUM_512)
        q_nope_new2 = pto.transpose(q_nope_new, [0, 1])
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        q_nope_new2 = pto.reshape(q_nope_new2, [b, s, self.num_heads, self.kv_lora_rank])
        pto.set_vec_tile_shapes(1, 1, 1, NUM_512)
        q_nope_new2 = pto.transpose(q_nope_new2, [1, 2])
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64) # (b, n, s,kvLoraRank)

        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        k_nope = pto.rms_norm(compressd_kv)
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        k_nope = pto.reshape(k_nope, [b, 1, s, self.kv_lora_rank]) # (b, 1, s,kvLoraRank)

        q_pe_rope = pto.tensor([b, self.num_heads, s, self.qk_rope_head_dim], k_pe.dtype, "q_pe_rope")
        k_pe_rope = pto.tensor([b, 1, s, self.qk_rope_head_dim], k_pe.dtype, "k_pe_rope")
        pto.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, q_pe_rope, k_pe_rope, 1, rope_tile_shape_config)
        pto.set_vec_tile_shapes(1, 1, NUM_128, NUM_64)

        query_states = pto.concat([q_nope_new2, q_pe_rope], -1) # (b, numHeads, s, kvLoraRank + qkRopeHeadDim)
        key_states = pto.concat([k_nope, k_pe_rope], -1) # (b, 1, s, kvLoraRank + qkRopeHeadDim)

        past_key_states_mew = pto.scatter_update(past_key_states, kv_len, key_states, -2) # 增量

        atten_res = self.attention(query_states, past_key_states_mew, atten_mask)

        return self.attention_post(atten_res)


    def attention_pre_forward(self, **kwargs) -> List[pto.tensor]:
        hidden_states = kwargs.get("hidden_states")
        atten_mask = kwargs.get("atten_mask")
        position_ids = kwargs.get("position_ids")
        cos = kwargs.get("cos")
        sin = kwargs.get("sin")
        kv_len = kwargs.get("kv_len")
        past_key_states = kwargs.get("past_key_states")
        rope_tile_shape_config = kwargs.get("rope_tile_shape_config")

        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        bs = b * s
        d_type = hidden_states.dtype

        qkv = self.qkv_pre(hidden_states)
        q = qkv[0]
        compressd_kv = qkv[1]

        q_nope = pto.view(q, [b, s, self.num_heads, self.qk_nope_head_dim], [0, 0, 0, 0])
        q_pe = pto.view(q, [b, s, self.num_heads, self.qk_rope_head_dim], [0, 0, 0, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(1, 1, 1, NUM_64)
        q_pe = pto.transpose(q_pe, [1, 2])
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)

        k_pe = pto.view(compressd_kv, [b, s, self.qk_rope_head_dim], [0, 0, self.kv_lora_rank])
        compressd_kv = pto.view(compressd_kv, [b, s, self.kv_lora_rank], [0, 0, 0]) # (b, s, kvLoraRank)
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        k_pe = pto.reshape(k_pe, [b, 1, s, self.qk_rope_head_dim]) # setTileShapes 4维

        pto.set_vec_tile_shapes(1, NUM_128, 1, NUM_64) # SetVecTileShapes(1, 1, NUM_128, NUM_64)
        q_nope1 = pto.reshape(q_nope, [b * s, self.num_heads, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(1, 1, NUM_128)
        q_nope2 = pto.transpose(q_nope1, [0, 1]) # (n, bs, d)
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)

        pto.set_cube_tile_shapes([min(NUM_128, bs), min(NUM_128, bs)], [NUM_128, NUM_128], [NUM_128, NUM_128])
        pto.set_vec_tile_shapes(NUM_128, NUM_64) # for Assemble
        q_nope_new = pto.matmul(q_nope2, self.kv_b_proj_wk, d_type)
        pto.set_vec_tile_shapes(1, 1, NUM_512)
        q_nope_new2 = pto.transpose(q_nope_new, [0, 1])
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        q_nope_new2 = pto.reshape(q_nope_new2, [b, s, self.num_heads, self.kv_lora_rank])
        pto.set_vec_tile_shapes(1, 1, 1, NUM_512)
        q_nope_new2 = pto.transpose(q_nope_new2, [1, 2])
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64) # (b, n, s,kvLoraRank)

        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        k_nope = pto.rms_norm(compressd_kv)
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)
        k_nope = pto.reshape(k_nope, [b, 1, s, self.kv_lora_rank]) # (b, 1, s,kvLoraRank)

        q_pe_rope = pto.tensor([b, self.num_heads, s, self.qk_rope_head_dim], k_pe.dtype, "q_pe_rope")
        k_pe_rope = pto.tensor([b, 1, s, self.qk_rope_head_dim], k_pe.dtype, "k_pe_rope")
        pto.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, q_pe_rope, k_pe_rope, 1, rope_tile_shape_config)
        pto.set_vec_tile_shapes(1, 1, NUM_128, NUM_64)

        query_states = pto.concat([q_nope_new2, q_pe_rope], -1) # (b, numHeads, s, kvLoraRank + qkRopeHeadDim)
        key_states = pto.concat([k_nope, k_pe_rope], -1) # (b, 1, s, kvLoraRank + qkRopeHeadDim)

        past_key_states_mew = pto.scatter_update(past_key_states, kv_len, key_states, -2) # 增量
        return [query_states, past_key_states_mew]


    def attention_pre_forward_cv(self, **kwargs) -> List[pto.tensor]:
        hidden_states = kwargs.get("hidden_states")
        atten_mask = kwargs.get("atten_mask")
        position_ids = kwargs.get("position_ids")
        cos = kwargs.get("cos")
        sin = kwargs.get("sin")
        kv_len = kwargs.get("kv_len")
        past_key_states = kwargs.get("past_key_states")
        rope_tile_shape_config = kwargs.get("rope_tile_shape_config")

        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        bs = b * s
        d_type = hidden_states.dtype

        qkv = self.qkv_pre_cv(hidden_states)
        q = qkv[0] # 2_1_32_192
        compressd_kv = qkv[1] # 2_1_576

        q_nope = pto.view(q, [b, s, self.num_heads, self.qk_nope_head_dim], [0, 0, 0, 0]) # 2_1_32_128
        q_pe = pto.view(q, [b, s, self.num_heads, self.qk_rope_head_dim], [0, 0, 0, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(NUM_2, 1, NUM_32, NUM_64)
        q_pe = pto.transpose(q_pe, [1, 2])

        k_pe = pto.view(compressd_kv, [b, s, self.qk_rope_head_dim], [0, 0, self.kv_lora_rank])
        compressd_kv = pto.view(compressd_kv, [b, s, self.kv_lora_rank], [0, 0, 0]) # (b, s ,kvLoraRank) 2_1_512
        pto.set_vec_tile_shapes(NUM_2, 1, NUM_64)
        k_pe = pto.reshape(k_pe, [b, 1, s, self.qk_rope_head_dim]) # setTileShapes 4维 2_1_1_64

        pto.set_vec_tile_shapes(NUM_2, 1, NUM_32, NUM_128)
        q_nope1 = pto.reshape(q_nope, [b * s, self.num_heads, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(NUM_2, NUM_32, NUM_128)
        q_nope2 = pto.transpose(q_nope1, [0, 1]) # (n, bs, d)
        pto.set_vec_tile_shapes(1, NUM_128, NUM_64)

        pto.set_cube_tile_shapes([min(NUM_128, bs), min(NUM_128, bs)], [NUM_128, NUM_128], [NUM_128, NUM_128])
        q_nope_new = pto.matmul(q_nope2, self.kv_b_proj_wk, d_type)
        pto.set_vec_tile_shapes(NUM_16, NUM_2, NUM_512)
        q_nope_new2 = pto.transpose(q_nope_new, [0, 1])
        pto.set_vec_tile_shapes(1, NUM_32, NUM_512)
        q_nope_new2 = pto.reshape(q_nope_new2, [b, s, self.num_heads, self.kv_lora_rank])
        pto.set_vec_tile_shapes(NUM_2, 1, NUM_32, NUM_256)
        q_nope_new2 = pto.transpose(q_nope_new2, [1, 2])

        pto.set_vec_tile_shapes(NUM_2, 1, NUM_512)
        k_nope = pto.rms_norm(compressd_kv)
        pto.set_vec_tile_shapes(NUM_2, 1, NUM_512)
        k_nope = pto.reshape(k_nope, [b, 1, s, self.kv_lora_rank]) # (b, 1, s,kvLoraRank)

        q_pe_rope = pto.tensor([b, self.num_heads, s, self.qk_rope_head_dim], k_pe.dtype, "q_pe_rope")
        k_pe_rope = pto.tensor([b, 1, s, self.qk_rope_head_dim], k_pe.dtype, "k_pe_rope")
        pto.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, q_pe_rope, k_pe_rope, 1, rope_tile_shape_config)
        pto.set_vec_tile_shapes(NUM_2, NUM_32, 1, NUM_64)

        query_states = pto.concat([q_nope_new2, q_pe_rope], -1) # (b, numHeads, s, kvLoraRank + qkRopeHeadDim)
        key_states = pto.concat([k_nope, k_pe_rope], -1) # (b, 1, s, kvLoraRank + qkRopeHeadDim)

        pto.set_vec_tile_shapes(NUM_2, 1, NUM_128, NUM_64)
        past_key_states_mew = pto.scatter_update(past_key_states, kv_len, key_states, -2) # 增量
        return [query_states, past_key_states_mew]


    def mla_prolog_ab_forward(self, hidden_states: pto.tensor,
                     q_pe_rope: pto.tensor, is_quant: bool = False) -> List[pto.tensor]:
        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        bs = b * s
        d_type = hidden_states.dtype

        qkv = self.qkv_pre2(hidden_states, is_quant)
        q = qkv[0]
        kv_tmp = qkv[1]

        if is_quant:
            pto.set_vec_tile_shapes(min(NUM_32, bs), NUM_64)
            q_tmp_fp32 = pto.cast(q, pto.DT_FP32)
            q_tmp_scale_dequant = qkv[2]
            q_tmp_dequant_per_token = pto.mul(q_tmp_fp32, q_tmp_scale_dequant)
            q_tmp_dequant_channel = pto.mul(q_tmp_dequant_per_token, self.q_b_proj_w_scale)
            q = pto.cast(q_tmp_dequant_channel, d_type)
        q_tmp = pto.reshape(q, [b, s, self.num_heads, self.q_head_dim])

        q_nope = pto.view(q_tmp, [b, s, self.num_heads, self.qk_nope_head_dim], [0, 0, 0, 0])

        pto.set_vec_tile_shapes(NUM_2, 1, NUM_32, NUM_128)
        q_nope_r = pto.reshape(q_nope, [bs, self.num_heads, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(NUM_2, NUM_32, self.qk_nope_head_dim)
        q_nope_t = pto.transpose(q_nope_r, [0, 1])

        c0 = NUM_16
        m = (min(NUM_32, bs) + c0 - 1) // c0 * c0
        pto.set_cube_tile_shapes([m, m], [NUM_128, NUM_128], [NUM_128, NUM_128])
        q_nope_new = pto.matmul(q_nope_t, self.kv_b_proj_wk, d_type)

        pto.set_vec_tile_shapes(NUM_16, NUM_2, self.kv_lora_rank)
        q_nope_new_t = pto.transpose(q_nope_new, [0, 1])
        q_nope_new_r = pto.reshape(q_nope_new_t, [b, s, self.num_heads, self.kv_lora_rank])
        pto.set_vec_tile_shapes(NUM_2, 1, NUM_32, self.kv_lora_rank)
        q_nope_new_t2 = pto.transpose(q_nope_new_r, [1, 2])

        pto.set_vec_tile_shapes(NUM_2, NUM_32, 1, NUM_64)
        query_states = pto.concat([q_nope_new_t2, q_pe_rope], -1)

        return [query_states, kv_tmp]


    def mla_prolog_forward(self, **kwargs) -> List[pto.tensor]:
        hidden_states = kwargs.get("hidden_states")
        position_ids = kwargs.get("position_ids")
        cos = kwargs.get("cos")
        sin = kwargs.get("sin")
        kv_len = kwargs.get("kv_len")
        past_key_states = kwargs.get("past_key_states")
        rope_tile_shape_config = kwargs.get("rope_tile_shape_config")
        is_quant = kwargs.get("is_quant", False)

        b = hidden_states.shape[0]
        s = hidden_states.shape[1]
        bs = b * s
        d_type = hidden_states.dtype

        qkv = self.qkv_pre2(hidden_states, is_quant)
        q = qkv[0]
        kv_tmp = qkv[1]

        if is_quant:
            pto.set_vec_tile_shapes(min(NUM_32, bs), NUM_64)
            q_tmp_fp32 = pto.cast(q, pto.DT_FP32)
            q_tmp_scale_dequant = qkv[2]
            q_tmp_dequant_per_token = pto.mul(q_tmp_fp32, q_tmp_scale_dequant)
            q_tmp_dequant_channel = pto.mul(q_tmp_dequant_per_token, self.q_b_proj_w_scale)

            q = pto.cast(q_tmp_dequant_channel, d_type)
        q_tmp = pto.reshape(q, [b, s, self.num_heads, self.q_head_dim])

        q_nope = pto.view(q_tmp, [b, s, self.num_heads, self.qk_nope_head_dim], [0, 0, 0, 0])
        pto.set_vec_tile_shapes(NUM_32, 1, 1, NUM_128)
        q_nope_r = pto.reshape(q_nope, [bs, self.num_heads, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(NUM_2, NUM_32, self.qk_nope_head_dim)
        q_nope_t = pto.transpose(q_nope_r, [0, 1])

        c0 = NUM_16
        m = (min(NUM_32, bs) + c0 - 1) // c0 * c0
        pto.set_cube_tile_shapes([m, m], [NUM_128, NUM_128], [NUM_128, NUM_128])
        q_nope_new = pto.matmul(q_nope_t, self.kv_b_proj_wk, d_type)

        pto.set_vec_tile_shapes(NUM_16, NUM_2, self.kv_lora_rank)
        q_nope_new_t = pto.transpose(q_nope_new, [0, 1])
        q_nope_new_r = pto.reshape(q_nope_new_t, [b, s, self.num_heads, self.kv_lora_rank])
        pto.set_vec_tile_shapes(NUM_2, 1, NUM_32, self.kv_lora_rank)
        q_nope_new_t2 = pto.transpose(q_nope_new_r, [1, 2])

        compressd_kv = pto.view(kv_tmp, [b, s, self.kv_lora_rank], [0, 0, 0])
        pto.set_vec_tile_shapes(NUM_2, 1, NUM_512)
        compressd_kv_norm = pto.rms_norm(compressd_kv)
        k_nope = pto.reshape(compressd_kv_norm, [b, 1, s, self.kv_lora_rank])

        q_pe = pto.view(q_tmp, [b, s, self.num_heads, self.qk_rope_head_dim], [0, 0, 0, self.qk_nope_head_dim])
        pto.set_vec_tile_shapes(NUM_2, 1, NUM_32, self.qk_nope_head_dim)
        q_pe_t = pto.transpose(q_pe, [1, 2])

        k_pe = pto.view(kv_tmp, [b, s, self.qk_rope_head_dim], [0, 0, self.kv_lora_rank])
        pto.set_vec_tile_shapes(min(NUM_32, bs), 1, NUM_64)
        k_pe_r = pto.reshape(k_pe, [b, 1, s, self.qk_rope_head_dim])

        q_pe_rope = pto.tensor([b, self.num_heads, s, self.qk_rope_head_dim], q_pe_t.dtype, "q_pe_rope")
        k_pe_rope = pto.tensor([b, 1, s, self.qk_rope_head_dim], k_pe_r.dtype, "k_pe_rope")
        pto.apply_rotary_pos_emb(q_pe_t, k_pe_r, cos, sin, position_ids,
                     q_pe_rope, k_pe_rope, 1, rope_tile_shape_config)

        pto.set_vec_tile_shapes(NUM_2, NUM_32, 1, NUM_64)
        query_states = pto.concat([q_nope_new_t2, q_pe_rope], -1)

        pto.set_vec_tile_shapes(1, 1, 1, NUM_64)
        key_states = pto.concat([k_nope, k_pe_rope], -1)

        pto.set_vec_tile_shapes(1, 1, NUM_256, NUM_64)
        past_key_states = pto.scatter_update(past_key_states, kv_len, key_states, -2)

        res = [query_states, past_key_states]
        return res


def test_attention():
    b = NUM_2
    n1 = NUM_2
    n2 = 1
    s1 = NUM_128
    s2 = NUM_128
    d = NUM_576
    pto.set_platform_config(KEY_ONLY_HOST_COMPILE, True)

    query = pto.tensor([b, n1, s1, d], pto.DT_BF16)
    kv = pto.tensor([b, n2, s2, d], pto.DT_BF16)
    atten_mask = pto.tensor([b, n2, s1, s2], pto.DT_BF16)
    aw = AttentionW()
    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("attention", graph_type, func_type):
        atten = DeepSeekAttention(g_deepseek_config, aw, 1)
        res = atten.attention(query, kv, atten_mask)


def test_attention_post():
    b = NUM_2
    n1 = num_2
    n2 = 1
    s1 = NUM_128
    s2 = NUM_128
    d = NUM_576
    pto.set_platform_config(KEY_ONLY_HOST_COMPILE, True)

    query = pto.tensor([b, n1, s1, d], pto.DT_BF16)
    kv = pto.tensor([b, n2, s2, d], pto.DT_BF16)
    atten_mask = pto.tensor([b, n2, s1, s2], pto.DT_BF16)
    aw = AttentionW()
    aw.kv_b_proj_wv = pto.tensor([n1, NUM_512, NUM_128], pto.DT_BF16) # [n ,kvLoraRank, vHeadDim]
    aw.o_proj_w = pto.tensor([b, NUM_256, NUM_256], pto.DT_BF16) # [n * vHeadDim, h]

    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("attention_post", graph_type, func_type):
        atten = DeepSeekAttention(g_deepseek_config, aw, 1)
        atten_res = atten.attention(query, kv, atten_mask)
        res = atten.attention_post(atten_res)


def test_attention_post2():
    b = NUM_32
    n = NUM_32
    s = 1
    kv_lora_rank = NUM_512
    v_head_dim = NUM_128
    h = NUM_7168
    pto.set_platform_config(KEY_ONLY_HOST_COMPILE, True)
    atten_post_in = pto.tensor([b, n, s, kv_lora_rank], pto.DT_BF16, "attnPostIn")
    atten_ouput = pto.tensor()
    aw = AttentionW()
    aw.kv_b_proj_wv = pto.tensor([n, kv_lora_rank, v_head_dim], pto.DT_BF16, "kvBProjWV")
    aw.o_proj_w = pto.tensor([n * v_head_dim, h], pto.DT_BF16, "oProjW")

    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("attention_post2", graph_type, func_type):
        atten = DeepSeekAttention(g_deepseek_config, aw, 1)
        atten_output = atten.attention_post2(atten_post_in)


def test_qkv_pre():
    b = NUM_2
    s = NUM_128
    h = g_deepseek_config["hiddenSize"]
    num_heads = g_deepseek_config["numAttentionHeads"]
    q_lora_rank = g_deepseek_config["qLoraRank"]
    qk_rope_head_dim = g_deepseek_config["qkRopeHeadDim"]
    kv_lora_rank = g_deepseek_config["kvLoraRank"]
    v_head_dim = g_deepseek_config["vHeadDim"]
    qk_nope_head_dim = g_deepseek_config["qkNopeHeadDim"]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    pto.set_platform_config(KEY_ONLY_HOST_COMPILE, True)

    hidden_states = pto.tensor([b, s, h], pto.DT_BF16, "hidden_states")

    aw = AttentionW()
    aw.q_a_proj_w = pto.tensor([h, q_lora_rank], pto.DT_BF16, "qAProjW")
    aw.q_b_proj_w = pto.tensor([q_lora_rank, num_heads * q_head_dim], pto.DT_BF16, "qBProjW")
    aw.kv_a_proj_with_mqa_w = pto.tensor([h, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "kvAProjWithMqaW")
    aw.kv_b_proj_wk = pto.tensor([num_heads, qk_nope_head_dim, kv_lora_rank], pto.DT_BF16, "kvBProjWK")
    aw.o_proj_w = pto.tensor([num_heads * v_head_dim], pto.DT_BF16, "oProjW")

    res = []
    atten = DeepSeekAttention(g_deepseek_config, aw, 1)
    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("qkvpre", graph_type, func_type):
        res = atten.qkv_pre(hidden_states)


def test_qkv_pre_cv():
    b = NUM_2
    s = 1
    pto.set_platform_config(KEY_ONLY_HOST_COMPILE, True)
    h = g_deepseek_config["hiddenSize"]
    num_heads = g_deepseek_config["numAttentionHeads"]
    q_lora_rank = g_deepseek_config["qLoraRank"]
    qk_rope_head_dim = g_deepseek_config["qkRopeHeadDim"]
    kv_lora_rank = g_deepseek_config["kvLoraRank"]
    v_head_dim = g_deepseek_config["vHeadDim"]
    qk_nope_head_dim = g_deepseek_config["qkNopeHeadDim"]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    hidden_states = pto.tensor([b, s, h], pto.DT_BF16, "hidden_states")

    aw = AttentionW()
    aw.q_a_proj_w = pto.tensor([h, q_lora_rank], pto.DT_BF16, "qAProjW")
    aw.q_b_proj_w = pto.tensor([q_lora_rank, num_heads * q_head_dim], pto.DT_BF16, "qBProjW")
    aw.kv_a_proj_with_mqa_w = pto.tensor([h, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "kvAProjWithMqaW")
    aw.kv_b_proj_wk = pto.tensor([num_heads, qk_nope_head_dim, kv_lora_rank], pto.DT_BF16, "kvBProjWK")
    aw.kv_b_proj_wv = pto.tensor([num_heads, kv_lora_rank, v_head_dim], pto.DT_BF16, "kvBProjWV")
    aw.o_proj_w = pto.tensor([num_heads * v_head_dim], pto.DT_BF16, "oProjW")

    res = []
    atten = DeepSeekAttention(g_deepseek_config, aw, 1)
    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("qkvpre", graph_type, func_type):
        res = atten.qkv_pre_cv(hidden_states)


def test_qkv_pre2():
    pto.set_platform_config(KEY_ONLY_HOST_COMPILE, True)

    h = g_deepseek_config["hiddenSize"]
    num_heads = g_deepseek_config["numAttentionHeads"]
    q_lora_rank = g_deepseek_config["qLoraRank"]
    qk_rope_head_dim = g_deepseek_config["qkRopeHeadDim"]
    kv_lora_rank = g_deepseek_config["kvLoraRank"]
    v_head_dim = g_deepseek_config["vHeadDim"]
    qk_nope_head_dim = g_deepseek_config["qkNopeHeadDim"]

    b = NUM_2
    s = 1
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    is_quant = False
    hidden_states = pto.tensor([b, s, h], pto.DT_BF16, "hidden_states")

    aw = AttentionW()
    aw.q_a_proj_w = pto.tensor([h, q_lora_rank], pto.DT_BF16, "qAProjW")
    aw.q_b_proj_w = pto.tensor([q_lora_rank, num_heads * q_head_dim], pto.DT_BF16, "qBProjW")
    aw.kv_a_proj_with_mqa_w = pto.tensor([h, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "kvAProjWithMqaW")
    aw.kv_b_proj_wk = pto.tensor([num_heads, qk_nope_head_dim, kv_lora_rank], pto.DT_BF16, "kvBProjWK")
    aw.kv_b_proj_wv = pto.tensor([num_heads, kv_lora_rank, v_head_dim], pto.DT_BF16, "kvBProjWV")
    aw.o_proj_w = pto.tensor([num_heads * v_head_dim], pto.DT_BF16, "oProjW")

    res = []
    atten = DeepSeekAttention(g_deepseek_config, aw, 1)
    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("qkv_pre2", graph_type, func_type):
        res = atten.qkv_pre2(hidden_states, is_quant)


def test_forward():
    pto.set_platform_config(KEY_ONLY_HOST_COMPILE, True)

    b = NUM_2
    s = 1
    s2 = NUM_512
    h = g_deepseek_config["hiddenSize"]
    num_heads = g_deepseek_config["numAttentionHeads"]
    q_lora_rank = g_deepseek_config["qLoraRank"]
    qk_rope_head_dim = g_deepseek_config["qkRopeHeadDim"]
    kv_lora_rank = g_deepseek_config["kvLoraRank"]
    v_head_dim = g_deepseek_config["vHeadDim"]
    qk_nope_head_dim = g_deepseek_config["qkNopeHeadDim"]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    hidden_states = pto.tensor([b, s, h], pto.DT_BF16, "hidden_states")
    atten_mask = pto.tensor([b, 1, s, s2], pto.DT_FP32, "atten_mask")
    position_ids = pto.tensor([b, s], pto.DT_INT32, "position_ids")
    cos = pto.tensor([s, qk_rope_head_dim], pto.DT_BF16, "cos")
    sin = pto.tensor([s, qk_rope_head_dim], pto.DT_BF16, "sin")
    kv_len = pto.tensor([1, 1], pto.DT_INT32, "kv_len")
    past_key_states = pto.tensor([b, 1, s2, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "past_key_states")

    aw = AttentionW()
    aw.q_a_proj_w = pto.tensor([h, q_lora_rank], pto.DT_BF16, "qAProjW")
    aw.q_b_proj_w = pto.tensor([q_lora_rank, num_heads * q_head_dim], pto.DT_BF16, "qBProjW")
    aw.kv_a_proj_with_mqa_w = pto.tensor([h, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "kvAProjWithMqaW")
    aw.kv_b_proj_wk = pto.tensor([num_heads, qk_nope_head_dim, kv_lora_rank], pto.DT_BF16, "kvBProjWK")
    aw.kv_b_proj_wv = pto.tensor([num_heads, kv_lora_rank, v_head_dim], pto.DT_BF16, "kvBProjWV")
    aw.o_proj_w = pto.tensor([num_heads * v_head_dim, h], pto.DT_BF16, "oProjW")

    rope_tile_config = pto.rope_tile_shape_config()
    rope_tile_config.two_dims_tile_shape = [32, 32]
    rope_tile_config.three_dims_tile_shape = [1, 32, 32]
    rope_tile_config.four_dims_tile_shape = [1, 1, 32, 32]
    rope_tile_config.five_dims_tile_shape = [1, 1, 32, 32, 2]

    res = pto.tensor()
    atten = DeepSeekAttention(g_deepseek_config, aw, 1)
    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("forward", graph_type, func_type):
        res = atten.forward(hidden_states=hidden_states,
                            atten_mask=atten_mask,
                            position_ids=position_ids,
                            cos=cos,
                            sin=sin,
                            kv_len=kv_len,
                            past_key_states=past_key_states,
                            rope_tile_shape_config=rope_tile_config)


def test_attention_pre_forward():
    b = NUM_2
    s = 1
    s2 = NUM_256
    h = g_deepseek_config["hiddenSize"]
    num_heads = g_deepseek_config["numAttentionHeads"]
    q_lora_rank = g_deepseek_config["qLoraRank"]
    qk_rope_head_dim = g_deepseek_config["qkRopeHeadDim"]
    kv_lora_rank = g_deepseek_config["kvLoraRank"]
    v_head_dim = g_deepseek_config["vHeadDim"]
    qk_nope_head_dim = g_deepseek_config["qkNopeHeadDim"]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    hidden_states = pto.tensor([b, s, h], pto.DT_BF16, "hidden_states")
    atten_mask = pto.tensor([b, 1, s, s2], pto.DT_FP32, "atten_mask")
    position_ids = pto.tensor([b, s], pto.DT_INT32, "position_ids")
    cos = pto.tensor([s, qk_rope_head_dim], pto.DT_BF16, "cos")
    sin = pto.tensor([s, qk_rope_head_dim], pto.DT_BF16, "sin")
    kv_len = pto.tensor([1, 1], pto.DT_INT32, "kv_len")
    past_key_states = pto.tensor([b, 1, s2, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "past_key_states")

    aw = AttentionW()
    aw.q_a_proj_w = pto.tensor([h, q_lora_rank], pto.DT_BF16, "qAProjW")
    aw.q_b_proj_w = pto.tensor([q_lora_rank, num_heads * q_head_dim], pto.DT_BF16, "qBProjW")
    aw.kv_a_proj_with_mqa_w = pto.tensor([h, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "kvAProjWithMqaW")
    aw.kv_b_proj_wk = pto.tensor([num_heads, qk_nope_head_dim, kv_lora_rank], pto.DT_BF16, "kvBProjWK")
    aw.kv_b_proj_wv = pto.tensor([num_heads, kv_lora_rank, v_head_dim], pto.DT_BF16, "kvBProjWV")
    aw.o_proj_w = pto.tensor([num_heads * v_head_dim, h], pto.DT_BF16, "oProjW")

    rope_tile_config = pto.rope_tile_shape_config()
    rope_tile_config.two_dims_tile_shape = [32, 32]
    rope_tile_config.three_dims_tile_shape = [1, 32, 32]
    rope_tile_config.four_dims_tile_shape = [1, 1, 32, 32]
    rope_tile_config.five_dims_tile_shape = [1, 1, 32, 32, 2]

    res = pto.tensor()
    atten = DeepSeekAttention(g_deepseek_config, aw, 1)
    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("pre_forward", graph_type, func_type):
        res = atten.attention_pre_forward(hidden_states=hidden_states,
                                            atten_mask=atten_mask,
                                            position_ids=position_ids,
                                            cos=cos,
                                            sin=sin,
                                            kv_len=kv_len,
                                            past_key_states=past_key_states,
                                            rope_tile_shape_config=rope_tile_config)


def test_attention_pre_forward_cv():
    b = NUM_32
    s = 1
    s2 = NUM_256
    h = NUM_7168
    num_heads = NUM_128
    q_lora_rank = NUM_1536
    qk_rope_head_dim = NUM_64
    kv_lora_rank = NUM_512
    v_head_dim = NUM_128
    qk_nope_head_dim = NUM_128

    h = 1024
    num_heads = 2
    q_lora_rank = 512
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    hidden_states = pto.tensor([b, s, h], pto.DT_BF16, "hidden_states")
    atten_mask = pto.tensor([b, 1, s, s2], pto.DT_FP32, "atten_mask")
    position_ids = pto.tensor([b, s], pto.DT_INT32, "position_ids")
    cos = pto.tensor([s, qk_rope_head_dim], pto.DT_BF16, "cos")
    sin = pto.tensor([s, qk_rope_head_dim], pto.DT_BF16, "sin")
    kv_len = pto.tensor([1, 1], pto.DT_INT32, "kv_len")
    past_key_states = pto.tensor([b, 1, s2, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "past_key_states")

    aw = AttentionW()
    aw.q_a_proj_w = pto.tensor([h, q_lora_rank], pto.DT_BF16, "qAProjW")
    aw.q_b_proj_w = pto.tensor([q_lora_rank, num_heads * q_head_dim], pto.DT_BF16, "qBProjW")
    aw.kv_a_proj_with_mqa_w = pto.tensor([h, kv_lora_rank + qk_rope_head_dim], pto.DT_BF16, "kvAProjWithMqaW")
    aw.kv_b_proj_wk = pto.tensor([num_heads, qk_nope_head_dim, kv_lora_rank], pto.DT_BF16, "kvBProjWK")
    aw.kv_b_proj_wv = pto.tensor([num_heads, kv_lora_rank, v_head_dim], pto.DT_BF16, "kvBProjWV")
    aw.o_proj_w = pto.tensor([num_heads * v_head_dim, h], pto.DT_BF16, "oProjW")

    rope_tile_config = pto.rope_tile_shape_config()
    rope_tile_config.two_dims_tile_shape = [32, 32]
    rope_tile_config.three_dims_tile_shape = [1, 32, 32]
    rope_tile_config.four_dims_tile_shape = [1, 1, 32, 32]
    rope_tile_config.five_dims_tile_shape = [1, 1, 32, 32, 2]

    res = pto.tensor()
    atten = DeepSeekAttention(g_deepseek_config, aw, 1)
    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("attention_pre_forward_cv", graph_type, func_type):
        res = atten.attention_pre_forward_cv(hidden_states=hidden_states,
                                            atten_mask=atten_mask,
                                            position_ids=position_ids,
                                            cos=cos,
                                            sin=sin,
                                            kv_len=kv_len,
                                            past_key_states=past_key_states,
                                            rope_tile_shape_config=rope_tile_config)


def test_mla_prolog_forward():
    b = NUM_4
    s = 1
    s2 = NUM_1024
    h = NUM_1024
    n = NUM_32
    q_lora_rank = NUM_256
    qk_nope_head_dim = NUM_128
    qk_rope_head_dim = NUM_64
    kv_lora_rank = NUM_512
    v_head_dim = NUM_128
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    d_type = pto.DT_BF16
    x = pto.tensor([b, s, h], d_type, "x")
    w_qa = pto.tensor([h, q_lora_rank], d_type, "w_qa")
    w_qb = pto.tensor([q_lora_rank, n * q_head_dim], d_type, "w_qb")
    w_kv_a = pto.tensor([h, kv_lora_rank + qk_rope_head_dim], d_type, "w_kv_a")
    w_kv_b_k = pto.tensor([n, qk_nope_head_dim, kv_lora_rank], d_type, "w_kv_b_k")
    position_ids = pto.tensor([b, s], pto.DT_INT32, "position_ids")
    cos = pto.tensor([s, qk_rope_head_dim], d_type, "cos")
    sin = pto.tensor([s, qk_rope_head_dim], d_type, "sin")
    past_key_states = pto.tensor([b, 1, s2, kv_lora_rank + qk_rope_head_dim], d_type, "past_key_states")
    kv_len = pto.tensor([1, 1], pto.DT_INT32, "kv_len")
    output_q = pto.tensor([b, n, s, kv_lora_rank + qk_rope_head_dim], d_type, "output_q")

    aw = AttentionW()
    aw.q_a_proj_w = w_qa
    aw.q_b_proj_w = w_qb
    aw.kv_a_proj_with_mqa_w = w_kv_a
    aw.kv_b_proj_wk = w_kv_b_k

    rope_tile_config = pto.rope_tile_shape_config()
    rope_tile_config.two_dims_tile_shape = [32, 64]
    rope_tile_config.three_dims_tile_shape = [1, 32, 64]
    rope_tile_config.four_dims_tile_shape = [1, 32, 1, 64]
    rope_tile_config.five_dims_tile_shape = [1, 32, 1, 64, 64]

    atten = DeepSeekAttention(g_deepseek_config, aw, 1)
    atten.num_heads = NUM_32
    atten.q_lora_rank = NUM_1536
    atten.hidden_size = NUM_7168
    graph_type = pto.GraphType.TENSOR_GRAPH
    func_type = pto.FunctionType.STATIC
    with pto.pto_function("mla_prolog_forward", graph_type, func_type):
        q_kv = atten.mla_prolog_forward(hidden_states=x,
                                        position_ids=position_ids,
                                        cos=cos,
                                        sin=sin,
                                        kv_len=kv_len,
                                        past_key_states=past_key_states,
                                        rope_tile_shape_config=rope_tile_config)
        output_q = q_kv[0]
        past_key_states = q_kv[1]
