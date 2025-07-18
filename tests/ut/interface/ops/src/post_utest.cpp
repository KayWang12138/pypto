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
 * \file post_utest.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "models/nsa/attention_post.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/float.h"

using namespace npu::tile_fwk;

class AttentionPostUTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
    }

    void TearDown() override {}
};

struct TestPostParams {
    int b;
    int n;
    int s;
    int h;
    int kvLoraRank;
    int vHeadDim;
};

template <typename T = npu::tile_fwk::float16, bool nz = true, typename wDtype = int8_t, bool isSmooth = false>
void TestAttentionPostUt(const TestPostParams &params, const PostTileConfig &tileConfig) {
    int b = params.b;
    int n = params.n;
    int s = params.s;
    int h = params.h;
    int kvLoraRank = params.kvLoraRank;
    int vHeadDim = params.vHeadDim;

    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuant = std::is_same<wDtype, int8_t>::value;
    DataType dTypeQuant = isQuant ? DT_INT8 : dType;

    std::vector<int> xShape = {b, s, n, kvLoraRank};
    std::vector<int> wUvShape = {n, kvLoraRank, vHeadDim};
    std::vector<int> woShape = {n * vHeadDim, h};
    std::vector<int> woScaleShape = {1, h};
    std::vector<int> smoothWoShape = {1, n * vHeadDim};
    std::vector<int> outShape = {b, s, h};

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    Tensor x(dType, xShape, "x");
    Tensor wUv(dType, wUvShape, "wUv");
    Tensor wo(dTypeQuant, woShape, "wo", NodeType::LOCAL, weightFormat);
    Tensor woScale;
    Tensor smoothWo;
    Tensor postOut(dType, outShape, "postOut");

    if (isQuant) {
        Tensor scale(DT_FP32, woScaleShape, "woScale");
        woScale = scale;
        if (isSmooth) {
            Tensor smooth(DT_FP32, smoothWoShape, "smoothWo");
            smoothWo = smooth;
        }
    }

    AttentionPost(x, wUv, wo, woScale, smoothWo, tileConfig, postOut);
}

TEST_F(AttentionPostUTest, b32_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    TestAttentionPostUt<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig);
}
