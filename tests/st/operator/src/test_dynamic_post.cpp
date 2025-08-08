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
 * \file test_dynamic_post.cpp
 * \brief
 */

#include "test_dynamic.h"
#include "test_suite_stest_ops.h"
#include "operator/models/nsa/attention_post.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class AttentionPostSTest : public npu::tile_fwk::stest::TestSuite_STest_Ops_Aihac {};

namespace {

struct TestPostParams {
    int b;
    int n;
    int s;
    int h;
    int kvLoraRank;
    int vHeadDim;
};

template <typename T>
static std::shared_ptr<RawTensorData> CreateTensorData(Tensor tensor, std::vector<int> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> values(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, values);
    return RawTensorData::CreateTensor<T>(tensor, values);
}

template <typename T>
static std::vector<T> getGoldenVec(std::vector<int> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

template <typename T = npu::tile_fwk::float16, bool nz = true, typename wDtype = int8_t, bool isSmooth = false>
void TestAttentionPost(const TestPostParams &params, const PostTileConfig &tileConfig, float precision) {
    config::SetHostConfig(npu::tile_fwk::KEY_ONLY_CODEGEN, true);
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

    std::vector<T> goldenDate = getGoldenVec<T>(outShape, "/golden_output.bin");

    auto xData = CreateTensorData<T>(x, xShape, "/x.bin");
    auto wUvData = CreateTensorData<T>(wUv, wUvShape, "/w_uv.bin");
    auto woData = CreateTensorData<wDtype>(wo, woShape, "/w_o.bin");
    auto outputData = RawTensorData::CreateConstantTensor<T>(postOut, 0.0);

    std::vector<RawTensorDataPtr> outputDataList = {outputData};
    std::vector<RawTensorDataPtr> inputDataList = {xData, wUvData, woData};
    if (isQuant) {
        Tensor scale(DT_FP32, woScaleShape, "woScale");
        woScale = scale;
        auto woScaleData = CreateTensorData<float>(woScale, woScaleShape, "/w_o_scale.bin");
        inputDataList.emplace_back(woScaleData);
        if (isSmooth) {
            Tensor smooth(DT_FP32, smoothWoShape, "smoothWo");
            smoothWo = smooth;
            auto smoothWoData = CreateTensorData<float>(smoothWo, smoothWoShape, "/smooth_wo.bin");
            inputDataList.emplace_back(smoothWoData);
        }
    } else {
        inputDataList.emplace_back(nullptr); // woScaleData
        inputDataList.emplace_back(nullptr); // smoothWoData
    }

    AttentionPost(x, wUv, wo, woScale, smoothWo, tileConfig, postOut);

    auto funcOp = Program::GetInstance().GetLastFunction()->GetDyndevAttribute();
#ifndef AC_ENABLE_FRAMEWORK_WITHOUT_CANN
    DynFuncRunner::Run(funcOp, inputDataList, outputDataList);

    std::cout << "postOut ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(goldenDate, (T *)outputData->data(), precision));
    resultCmp<T>(goldenDate, (T *)outputData->data(), precision, int(b*s*h*precision), 1000, false, false, 16);
#endif
}

void PerformanceConfig() {
    Program::GetInstance().GetConfig().Set<int>(CYCLE_UPPER_BOUND, 500000);
    Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{0, 4}});
}

////// fp16, nz, quant
TEST_F(AttentionPostSTest, b16_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {64, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true>(params, tileConfig, 0.006f);
}

////// bf16, nz, quant
TEST_F(AttentionPostSTest, b16_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {64, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true>(params, tileConfig, 0.006f);
}

////// fp16, nd, quant
TEST_F(AttentionPostSTest, b16_s1_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

////// fp16, nz, no quant
TEST_F(AttentionPostSTest, b32_s1_nz_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

////// fp16, nd, no quant
TEST_F(AttentionPostSTest, b16_s1_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b16_s2_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s1_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s2_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

} // namespace
