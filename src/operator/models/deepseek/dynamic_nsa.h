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
 * \file dynamic_mla.h
 * \brief
 */

#pragma once
#ifndef MLA_DYNAMIC
#define MLA_DYNAMIC

#include "common/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "operator/models/deepseek/deepseek_mla.h"

namespace npu::tile_fwk {

enum GateMode { standard, simple };

struct MlaQuantInputs {
    Tensor dequantScaleX;
    Tensor dequantScaleWDq;
    Tensor dequantScaleWUqQr;
    Tensor dequantScaleWDkvKr;
    Tensor quantScaleCkv;
    Tensor quantScaleCkr;
    Tensor smoothScalesCq;
};

struct SimpleParams {
    int b;
    int s;
    int s2;
    int d;
    int m;
    int k;
    int n;
    int right;
    int h;
    int n2;
    int q_lora_rank;
    int kv_lora_rank;
    int qk_rope_head_dim;
    int qk_nope_head_dim;
    int q_head_dim;
    std::string cacheMode;
    int blockSize;
    std::vector<int> vecTile;
    std::vector<int> cubeMTile;
    std::vector<int> cubeKTile;
    std::vector<int> cubeNTile;
    int tileB;
    static SimpleParams getCommonParams() {
        SimpleParams params;
        params.s = 1;
        params.h = NUM_7168;
        params.q_lora_rank = NUM_1536;
        params.kv_lora_rank = NUM_512;
        params.qk_rope_head_dim = NUM_64;
        params.qk_nope_head_dim = NUM_128;
        params.q_head_dim = params.qk_rope_head_dim + params.qk_nope_head_dim;
        params.cacheMode = "BNSD";
        params.blockSize = NUM_128;
        params.n2 = 1;
        return params;
    }

    static SimpleParams getLowParams() {
        SimpleParams params = getCommonParams();
        params.b = NUM_4;
        params.n = NUM_32;
        params.s2 = NUM_256;
        return params;
    }

    static SimpleParams getHighParams() {
        SimpleParams params = getCommonParams();
        params.b = NUM_32;
        params.n = NUM_128;
        params.s2 = NUM_4096;
        return params;
    }
};

void GenGatedScore(const Tensor &x, const Tensor &gateW1, const Tensor &gateW2, const Tensor &gateSimW1,
    Tensor &gatingScore, Tensor &mm1, Tensor &tempOut, GateMode gateMode = standard);

void GenSlc(const Tensor &x, Tensor &trans0res, Tensor &reduce0res, Tensor &trans1res, Tensor &reduce1res,
    Tensor &topkInd, Tensor &topkVal, Tensor &out, int actualLen, int l_prime = 64, int d = 16, int front = 1,
    int near = 2, int topk = 16);

void GenTopkIndicesFun(const Tensor &x, Tensor &trans0res, Tensor &reduce0res, Tensor &trans1res, Tensor &reduce1res,
    Tensor &topkInd, Tensor &topkVal, Tensor &out, int actualLen, int front = 1, int near = 2);

} // namespace npu::tile_fwk

#endif // MLA_DYNAMIC
