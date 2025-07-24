/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_nsa.h
 * \brief
 */

#pragma once
#ifndef DYNAMIC_NSA
#define DYNAMIC_NSA

#include "common/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_storage.h"
#include "models/nsa/selected_attention.h"
#include "models/deepseek/gen_kv_slc.h"
#include "models/nsa/attention_post.h"

namespace npu::tile_fwk {
constexpr int NUM_1 = 1;
constexpr int NUM_3 = 3;
constexpr int NUM_65536 = 65536;

enum GateMode {
    standard,
    simple
};

struct NSASimpleParams {
    int b;
    int s1;
    int s2;
    int n1;
    int n2;
    int h;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_head_dim;
    int qk_nope_head_dim;
    int q_head_dim;
    int rope_dim;
    int cmpBlockSize;
    int cmpStride;
    int slcBlockSize;
    int front;
    int near;
    int topk;
    std::string cacheMode;
    int blockSize;
    int vHeadDim;
    static NSASimpleParams getCommonParams() {
        NSASimpleParams params;
        params.h = NUM_7168;
        params.q_lora_rank = NUM_1536;
        params.kv_lora_rank = NUM_512;
        params.qk_rope_head_dim = NUM_64;
        params.qk_nope_head_dim = NUM_128;
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim;
        params.rope_dim = NUM_64;
        params.cmpBlockSize = NUM_32;
        params.cmpStride = NUM_16;
        params.slcBlockSize = NUM_64;
        params.front = NUM_1;
        params.near = NUM_2;
        params.topk = NUM_16;
        params.cacheMode = "BSND";
        params.blockSize = NUM_128;
        params.vHeadDim = NUM_128;
        return params;
    }

    static NSASimpleParams getDecodeParams() {
        NSASimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_1;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }

    static NSASimpleParams getMTPParams() {
        NSASimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.s1 = NUM_2;
        params.s2 = NUM_65536;
        params.n1 = NUM_128;
        params.n2 = NUM_1;
        return params;
    }
};

void GenGatedScore(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor& gatingScore, GateMode gateMode);

void GenAttn(Tensor &gatingScore, Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &attentionOut);

void DynamicNsa(Tensor &topkIndices, Tensor &topkTensorShape, Tensor &kvNopeCache, Tensor &kRopeCache, Tensor &kvActSeqs, Tensor &blockTable,
    int front, int near, int topk, int slcBlockSize, int blockSize, KvSlcTileShapeConfig &kvSlcTileConfig,
    const Tensor &qNope, const Tensor &qRope, Tensor &kvSlcActSeqs, float softmaxScale, SaTileShapeConfig saTileConfig,
    const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1, GateMode gateMode,
    Tensor &cmpAtten, Tensor &winAtten,
    Tensor &weightUV, Tensor &weightO, Tensor &weightOScale, Tensor &smoothScalesWo, const PostTileConfig &postConfig,
    Tensor &kvSlcActSeqOut, Tensor &attentionOut, Tensor &postOut);

} // namespace npu::tile_fwk

#endif // DYNAMIC_NSA
