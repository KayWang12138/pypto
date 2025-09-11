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

from pypto.module import AscppModule
from pypto.utils import CustStruct, TensorMap, Tensor, Var
from pypto.utils import DATATYPE as DT
from pypto.stub_fun import set_c1_cube_config, set_vec_tile_shapes, set_tile_shape, \
    set_c2_cube_config, set_cube_tile_shapes, continue_loop, maximum, tensor_to_aggregation_vec, \
    assemble, cast, matmul, update_record_tile_op, rms_norm
from pypto.flowcontrol import Loop, If, Else


class AttentionDims(CustStruct):
    b: int
    n: int
    s: int
    single_m: int
    single_n: int


class AttentionVecTileConfig(CustStruct):
    softmaxTileX: int
    softmaxTileY: int
    defaultVecTileX: int
    defaultVecTileY: int
    castTileX: int
    castTileY: int


class AttentionCubeTileConfig(CustStruct):
    ...


class ModelActive(CustStruct):
    q: Tensor
    k: Tensor
    v: Tensor
    m: Tensor
    n: Tensor
    l: Tensor


class ModelWeights(CustStruct):
    hidden_states: Tensor
    attn_weight: Tensor
    dense_weight: Tensor
    ffn_weight: Tensor
    
    def __init__(self, hidden_states: Tensor, attn_weight: Tensor, dense_weight: Tensor, ffn_weight: Tensor):
        self.hidden_states = hidden_states
        self.attn_weight = attn_weight
        self.dense_weight = dense_weight
        self.ffn_weight = ffn_weight
        super().__init__()


class FlashAttention(AscppModule):
    def init(self):
        self.oi_offset = None
        self.mi_offset = None
        self.li_offset = None

        self.s1_loop = None
        self.s2_loop = None
        self.tilda_pij_fp16 = None
        self.tilda_lij = None
        self.tilda_mij = None

        self.oi_tmp = None
        self.li_new = None
        self.mi_new = None

        self.kj = None
        self.qi = None
        self.vj = None

        self.last_oi = TensorMap()
        self.last_mi = TensorMap()
        self.last_li = TensorMap()
        self.result = Tensor(force_declare=True)
        
        self.single_m = 0
        self.single_n = 0
    
    def forward_s2loop_pre(self, vec_cfg: AttentionVecTileConfig, cube_cfg: AttentionCubeTileConfig):
        set_c1_cube_config(cube_cfg)
        sij = self.qi @ self.kj.t()

        set_vec_tile_shapes([vec_cfg.softmaxTileX, vec_cfg.softmaxTileY])
        set_tile_shape(0, 64)  # AscendProgram::GetInstance().GetConfig().set_tile_shape(0, NUM_64);
        set_tile_shape(1, 128)  # AscendProgram::GetInstance().GetConfig().set_tile_shape(1, NUM_128);

        self.tilda_mij = sij.row_max_single()
        tsub = sij - self.tilda_mij
        tilda_pij = tsub.exp()
        self.tilda_pij_fp16 = tilda_pij.astype(DT.fp16)
        self.tilda_lij = tilda_pij.row_sum_single()

        set_c2_cube_config(cube_cfg)
        
    def forward_s2loop_post(self):
        oi = self.last_oi.retrieve(self.oi_offset)
        li = self.last_li.retrieve(self.li_offset)
        mi = self.last_mi.retrieve(self.mi_offset)

        self.mi_new = maximum(mi, self.tilda_mij)
        t1 = mi - self.mi_new
        t2 = t1.exp()
        t3 = self.tilda_mij - self.mi_new
        t4 = t3.exp()
        t5 = t4 * self.tilda_lij
        t6 = t2 * li
        self.li_new = t6 + t5

        q3 = oi * t2
        q1 = self.tilda_pij_fp16 @ self.vj
        q2 = q1 * t4
        self.oi_tmp = q3 + q2
        
    def s1_inner_loop(self, b_idx: int, n_idx: int, s2_idx: int, attn_dims: AttentionDims, model_active: ModelActive):
        _qshape = model_active.q.shape
        dim0 = _qshape[0]
        dim1 = _qshape[1]
        b = attn_dims.b
        n = attn_dims.n
        s = dim0 // b
        d = dim1 // n
        s1_loop = s // attn_dims.single_m
        s2_loop = s // attn_dims.single_n
        
        self.kj = model_active.k.view([attn_dims.single_n, d], [b_idx * s + s2_idx * attn_dims.single_n, n_idx * d])
        self.vj = model_active.v.view([attn_dims.single_n, d], [b_idx * s + s2_idx * attn_dims.single_n, n_idx * d])
        with Loop(0, s1_loop) as s1_idx:
            self.qi = model_active.q.view([attn_dims.single_m, d], [b_idx * s + s1_idx * attn_dims.single_m, n_idx * d])
            self.oi_offset = [b_idx * s + s1_idx * attn_dims.single_m, n_idx * d]
            self.li_offset = [(b_idx * n + n_idx) * s + s1_idx * attn_dims.single_m, 0]
            self.mi_offset = [(b_idx * n + n_idx) * s + s1_idx * attn_dims.single_m, 0]
            self.forward_s2loop_pre(vec_cfg, cube_cfg)

            with If(s2_idx == 0):
                oi_tmp = self.tilda_pij_fp16 @ self.vj
                with If(s2_loop == 1):
                    li_expand = self.tilda_lij.reciprocal()
                    self.last_oi.set(self.oi_offset, oi_tmp * li_expand)
                with Else():
                    self.last_oi.set(self.oi_offset, oi_tmp)
                self.last_li.set(self.li_offset, self.tilda_lij)
                self.last_mi.set(self.mi_offset, self.tilda_mij)
                continue_loop()

            self.forward_s2loop_post()
            with If(s2_idx == s2_loop - 1):
                self.last_oi.set(self.oi_offset, self.oi_tmp * self.li_new.reciprocal())
            with Else():
                self.last_oi.set(self.oi_offset, self.oi_tmp)
            self.last_li.set(self.li_offset, self.li_new)
            self.last_mi.set(self.mi_offset, self.mi_new)
        
    def forward(self, model_active: ModelActive = None, attn_dims: AttentionDims = None,
                vec_cfg: AttentionVecTileConfig = None, cube_cfg: AttentionCubeTileConfig = None):
        _qshape = model_active.q.shape
        dim0 = _qshape[0]
        b = attn_dims.b
        n = attn_dims.n
        s = dim0 // b
        s2_loop = s // attn_dims.single_n
        with Loop(0, b) as b_idx:  # LLAMA_FUNCTION: FlashAttention_L4
            with Loop(0, n) as n_idx:  # LLAMA_FUNCTION: FlashAttention_L3
                with Loop(0, s2_loop) as s2_idx:  # LLAMA_FUNCTION: lashAttention_L2, END TO BE CHANGED
                    self.s1_inner_loop(b_idx, n_idx, s2_idx, attn_dims, model_active)

            aggregation = tensor_to_aggregation_vec(self.last_oi)
            self.result.assign(assemble(aggregation))
        return self.result


class MultiAttention(AscppModule):
    def init(self):
        self.flash_attn = FlashAttention()

    def forward(self,
                attn_weight: ModelWeights = None,
                model_active: ModelActive = None,
                attn_dims: CustStruct = None,
                vec_cfg: AttentionVecTileConfig = None,
                cube_cfg: AttentionCubeTileConfig = None) -> Tensor:
        x = cast(model_weights.hidden_states, DT.fp16)
        qkv = matmul(x, model_weights.attn_weight, DT.fp16)
        model_active.q = qkv.view(model_weights.hidden_states.shape, [0, 0])
        model_active.k = qkv.view(model_weights.hidden_states.shape, [0, model_weights.hidden_states.shape[1]])
        model_active.v = qkv.view(model_weights.hidden_states.shape, [0, model_weights.hidden_states.shape[1] * 2])

        result = self.flash_attn(model_active, attn_dims, vec_cfg, cube_cfg)
        self.flash_attn.gen_code("./generatedcpp/flashattention_generated.cpp")
        return result


class SetDefaultL0CubeConfig(AscppModule):
    def init(self):
        ...

    def forward(self):
        set_cube_tile_shapes([128, 128], [128, 128], [128, 128])


class LlamaLayer(AscppModule):
    def init(self):
        self.attn = MultiAttention()
        self.defcfg = SetDefaultL0CubeConfig()

    def update_hidden(self, attention_out: Tensor,
                      residual: Tensor,
                      model_weights: ModelWeights,
                      vec_cfg: AttentionVecTileConfig) -> Tensor:
        attention_out_fp16 = attention_out.astype(DT.fp16)

        self.defcfg()
        dense_out = attention_out_fp16 @ model_weights.dense_weight
        set_vec_tile_shapes([vec_cfg.defaultVecTileX, vec_cfg.defaultVecTileY])
        set_tile_shape(0, vec_cfg.defaultVecTileX)
        set_tile_shape(1, vec_cfg.defaultVecTileY)
        hidden_states = residual + dense_out

        residual = hidden_states
        hidden_states = rms_norm(hidden_states)

        a = hidden_states.astype(DT.fp16)
        gate = a @ model_weights.ffn_weight

        swish = gate * -1.0
        swish = swish.exp()
        swish = swish + 1.0
        swish = gate / swish

        up = a @ model_weights.ffn_weight
        swish = swish * up
        swish_fp16 = swish.astype(DT.fp16)

        mlp_res = swish_fp16 @ model_weights.ffn_weight.t()
        hidden_states = residual + mlp_res

        return hidden_states

    def forward(self,
                model_weights: ModelWeights = None,
                attn_dims: CustStruct = None,
                vec_cfg: AttentionVecTileConfig = None,
                cube_cfg: AttentionCubeTileConfig = None,
                update_tiles: Var = None):
        update_record_tile_op(update_tiles)
        set_vec_tile_shapes([vec_cfg.defaultVecTileX, vec_cfg.defaultVecTileY])
        set_tile_shape(0, vec_cfg.defaultVecTileX)
        set_tile_shape(1, vec_cfg.defaultVecTileY)
        self.defcfg()

        residual = model_weights.hidden_states

        model_weights.hidden_states = rms_norm(model_weights.hidden_states)
        bns = attn_dims.b * attn_dims.n * attn_dims.s
        model_active = ModelActive()
        model_active.m = Tensor([bns, 1], DT.fp32)
        model_active.l = Tensor([bns, 1], DT.fp32)

        set_tile_shape(0, vec_cfg.castTileX)
        set_tile_shape(1, vec_cfg.castTileY)
        
        attention_out = self.attn(model_weights, model_active, attn_dims, vec_cfg, cube_cfg)
        self.attn.gen_code("./generatedcpp/multiattention_generated.cpp")

        model_weights.hidden_states = self.update_hidden(attention_out, residual, model_weights, vec_cfg)

        return model_weights.hidden_states


if __name__ == '__main__':
    hidden = Tensor()
    attn_weight = Tensor()
    dense_weight = Tensor()
    ffn_weight = Tensor()
    model_weights = ModelWeights(hidden, attn_weight, dense_weight, ffn_weight)
    attn_dims = AttentionDims()
    vec_cfg = AttentionVecTileConfig()
    cube_cfg = AttentionCubeTileConfig()
    update_tile = Var('bool', is_declare=False)

    mod = LlamaLayer()
    mod(model_weights, attn_dims, vec_cfg, cube_cfg, update_tile)

    mod.gen_code('./generatedcpp/llamalayer_all_generated.cpp')

