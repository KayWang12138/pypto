/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_compress_attention_with_topk.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "interface/tensor/float.h"
#include "tilefwk/data_type.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "operator/models/nsa/compress_attention_with_topk.h"

using namespace npu::tile_fwk;

class CmpAttnTopk : public testing::Test {
};

template <typename T>
inline DataType GetDataType() {
    if (std::is_same<T, npu::tile_fwk::bfloat16>::value) {
        return DT_BF16;
    } else if (std::is_same<T, npu::tile_fwk::float16>::value) {
        return DT_FP16;
    }
    return DT_FP32;
}

struct CmpAttnTopkParams {
    int b;
    int s1;
    int n1;
    int dn;
    int dr;
    int n2;
    int blockSize;
    int cmpBlockSize;
    int cmpStride;
    int slcBlockSize;
    int topk;
    int front;
    int near;
    float softmaxScale;
};

inline CmpAttnTopkParams CalculateCmpAttnTopkParams(const std::vector<int> &input_param) {
    const int dn = input_param[3];
    const int dr = input_param[4];
    return {input_param[0], input_param[1], input_param[2], dn, dr,
            input_param[5], input_param[6], input_param[7], input_param[8],
            input_param[9], input_param[10], input_param[11], input_param[12],
            static_cast<float>(1.0 / sqrtf((dn + dr)))};
}

inline std::vector<int> CalculateActCmpSeq(const std::vector<int> &actSeqLen, int cmpBlockSize, int cmpStride) {
    std::vector<int> actCmpSeq;
    actCmpSeq.reserve(actSeqLen.size());
    for (auto curSeq : actSeqLen) {
        actCmpSeq.emplace_back((curSeq - cmpBlockSize) / cmpStride + 1);
    }
    return actCmpSeq;
}

inline int CalculateCmpBlockNum(const std::vector<int> &actCmpSeq, int blockSize, int s1) {
    int cmpBlockNum = 0;
    for (auto s : actCmpSeq) {
        cmpBlockNum += CeilDiv(s, blockSize);
    }
    return cmpBlockNum;
}

template <typename T = npu::tile_fwk::bfloat16>
void TestCmpAttnTopk(CmpAttnTopkTile &tileConfig, std::vector<int> input_param, std::vector<int> actSeqLen) {
    DataType dType = GetDataType<T>();

    auto params = CalculateCmpAttnTopkParams(input_param);
    auto actCmpSeq = CalculateActCmpSeq(actSeqLen, params.cmpBlockSize, params.cmpStride);
    int cmpBlockNum = CalculateCmpBlockNum(actCmpSeq, params.blockSize, params.s1);
    int maxCmpSeq = *(std::max_element(actCmpSeq.begin(), actCmpSeq.end()));
    int maxCmpBlockNum = CeilDiv(maxCmpSeq, params.blockSize);
    const int slcSize = params.slcBlockSize / params.cmpStride;

    Tensor qNope(dType, {params.b * params.s1 * params.n1, params.dn}, "qNope");
    Tensor qRope(dType, {params.b * params.s1 * params.n1, params.dr}, "qRope");
    Tensor cmpKvCache(dType, {cmpBlockNum, params.blockSize, params.n2, params.dn}, "cmpKvCache");
    Tensor cmpKrCache(dType, {cmpBlockNum, params.blockSize, params.n2, params.dr}, "cmpKrCache");
    Tensor cmpBlockTable(DT_INT32, {params.b, maxCmpBlockNum}, "cmpBlockTable");
    Tensor actSeq(DT_INT32, {params.b}, "actSeq");
    Tensor auxTensor(DT_FP32, {slcSize + params.cmpBlockSize / params.cmpStride - 1, params.n1}, "auxTensor");

    Tensor cmpAttn(DT_FP32, {params.b, params.s1, params.n1, params.dn}, "cmpAttnOut");
    Tensor topkRes(DT_INT32, {params.b, params.s1, params.topk}, "topkRes");

    FUNCTION("CompressAttentionWithTopK",
        {qNope, qRope, cmpKvCache, cmpKrCache, cmpBlockTable, actSeq, auxTensor},
        {cmpAttn, topkRes}) {
        CompressAttentionWithTopK(qNope, qRope, cmpKvCache, cmpKrCache, cmpBlockTable, actSeq, auxTensor, cmpAttn,
            topkRes, params.blockSize, params.cmpBlockSize, params.cmpStride, params.slcBlockSize, params.softmaxScale,
            params.n1, params.topk, params.front, params.near, tileConfig);
    }
}

void CommonTestConfig() {

    CmpAttnTopkTile config;
    config.topkTile = {1, 1, 128};
    config.cmpTile.c1Tile = {128, 128, 128, 128, 128, 128};
    config.cmpTile.v1Tile = {128, 128};
    config.cmpTile.c2Tile = {128, 128, 128, 128, 128, 128};
    config.cmpTile.v2Tile = {128, 64};

    std::vector<int> inputParam = {2, 1, 128, 512, 64, 1, 128, 32, 16, 64, 16, 1, 2};
    std::vector<int> actSeqLen = {8192, 8192};

    TestCmpAttnTopk<npu::tile_fwk::bfloat16>(config, inputParam, actSeqLen);
}

TEST_F(CmpAttnTopk, cmp_attn_with_topk_singleop_bf16) {
    CommonTestConfig();
}