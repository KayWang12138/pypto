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
#include "models/deepseek/dynamic_mla.h"
#include "models/nsa/win_attention.h"
#include "models/nsa/attention_post.h"

namespace npu::tile_fwk {
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
    int winSize;
    int vHeadDim;
    float eps;
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
        params.winSize = NUM_512;
        params.vHeadDim = NUM_128;
        params.eps = 1e-5f;
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

void DynamicNsa(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk,
    const Tensor &wDkvKr, const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos,
    const Tensor &cacheIndex, Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs,
    const MlaTileConfig &tileConfig, float epsilonCq, float epsilonCkv, std::string cacheMode,
    Tensor &topkIndices, Tensor &topkTensorShape, Tensor &kvActSeqs, Tensor &blockTable,
    int front, int near, int topk, int slcBlockSize, int blockSize, KvSlcTileShapeConfig &kvSlcTileConfig,
    Tensor &kvSlcActSeqs, float softmaxScale, SaTileShapeConfig saTileConfig,
    const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1, GateMode gateMode,
    Tensor &cmpAtten, Tensor &winAtten, int winSize, WinAttenTileShapeConfig &winAttntileConfig,
    Tensor &weightUV, Tensor &weightO, Tensor &weightOScale, Tensor &smoothScalesWo, const PostTileConfig &postConfig,
    Tensor &queryOut, Tensor &queryRopeOut, Tensor &kvCacheOut, Tensor &krCacheOut, Tensor &qNope, Tensor &qRope,
    Tensor &kvSlcActSeqOut, Tensor &kSlc, Tensor &vSlc, Tensor &slcAttn, Tensor &attentionOut, Tensor &postOut);

} // namespace npu::tile_fwk

#endif // DYNAMIC_NSA
