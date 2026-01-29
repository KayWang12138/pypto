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
 * \file test_dynamic_post.cpp
 * \brief
 */

#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/float.h"
#include "test_dev_func_runner.h"
#include "test_data_prepare.h"
#include "test_suite_stest_ops.h"
#include "operator/models/nsa/attention_post.h"
#include "tilefwk/tensor.h"

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
static std::vector<T> getGoldenVec(std::vector<int64_t> shape, std::string fileName) {
    int capacity = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> golden(capacity, 0);
    readInput<T>(GetGoldenDir() + fileName, golden);
    return golden;
}

struct AttentionPostShapes {
    std::vector<int64_t> xShape;
    std::vector<int64_t> wUvShape;
    std::vector<int64_t> wUvScaleShape;
    std::vector<int64_t> smoothWUvShape;
    std::vector<int64_t> woShape;
    std::vector<int64_t> woScaleShape;
    std::vector<int64_t> smoothWoShape;
    std::vector<int64_t> outShape;
};

inline AttentionPostShapes CalculateAttentionPostShapes(const TestPostParams &params) {
    int b = params.b;
    int n = params.n;
    int s = params.s;
    int h = params.h;
    int kvLoraRank = params.kvLoraRank;
    int vHeadDim = params.vHeadDim;

    return {
        {b, s, n, kvLoraRank},
        {n, kvLoraRank, vHeadDim},
        {n, 1, vHeadDim},
        {1, kvLoraRank},
        {n * vHeadDim, h},
        {1, h},
        {1, n * vHeadDim},
        {b, s, h}
    };
}

template <typename T>
struct AttentionPostTensors {
    Tensor x;
    Tensor wUv;
    Tensor wo;
    Tensor postOut;
};

template <typename T, bool nz, typename wUvDType, typename wODType>
AttentionPostTensors<T> CreateAttentionPostTensors(const AttentionPostShapes &shapes) {
    DataType dType = (std::is_same<T, npu::tile_fwk::float16>::value) ? DT_FP16 : DT_BF16;
    bool isQuantWUv = std::is_same<wUvDType, int8_t>::value;
    bool isQuantWo = std::is_same<wODType, int8_t>::value;

    TileOpFormat weightFormat = nz ? TileOpFormat::TILEOP_NZ : TileOpFormat::TILEOP_ND;
    
    return {
        Tensor(dType, shapes.xShape, "x"),
        Tensor(isQuantWUv ? DT_INT8 : dType, shapes.wUvShape, "wUv"),
        Tensor(isQuantWo ? DT_INT8 : dType, shapes.woShape, "wo", weightFormat),
        Tensor(dType, shapes.outShape, "postOut")
    };
}

template <typename T>
struct AttentionPostData {
    std::vector<T> golden;
    std::vector<RawTensorDataPtr> inputDataList;
    std::vector<RawTensorDataPtr> outputDataList;
};

template <typename T, typename wUvDType, typename wODType>
AttentionPostData<T> PrepareAttentionPostData(const AttentionPostTensors<T> &tensors,
                                              const AttentionPostShapes &shapes,
                                              bool isQuantWUv, bool isSmoothWUv,
                                              bool isQuantWo, bool isSmoothWo) {
    auto golden = getGoldenVec<T>(shapes.outShape, "/golden_output.bin");

    auto xData = CreateTensorData<T>(tensors.x, "/x.bin");
    auto wUvData = CreateTensorData<wUvDType>(tensors.wUv, "/w_uv.bin");
    auto woData = CreateTensorData<wODType>(tensors.wo, "/w_o.bin");
    auto outputData = RawTensorData::CreateConstantTensor<T>(tensors.postOut, 0.0);

    std::vector<RawTensorDataPtr> inputDataList = {xData, wUvData, woData};
    std::vector<RawTensorDataPtr> outputDataList = {outputData};

    QuantTensorWithData wUvQuant{isQuantWUv, isSmoothWUv, shapes.wUvScaleShape, shapes.smoothWUvShape,
        "wUvScale", "smoothWUv", "/w_uv_scale.bin", "/smooth_w_uv.bin"};
    CreateQuantTensorAndData(wUvQuant);
    inputDataList.emplace_back(wUvQuant.scale.dataPtr);
    inputDataList.emplace_back(wUvQuant.smooth.dataPtr);

    QuantTensorWithData wOQuant{isQuantWo, isSmoothWo, shapes.woScaleShape, shapes.smoothWoShape,
        "woScale", "smoothWo", "/w_o_scale.bin", "/smooth_w_o.bin"};
    CreateQuantTensorAndData(wOQuant);
    inputDataList.emplace_back(wOQuant.scale.dataPtr);
    inputDataList.emplace_back(wOQuant.smooth.dataPtr);

    return {golden, inputDataList, outputDataList};
}

inline void SetupAttentionPostProgram(const AttentionPostTensors<npu::tile_fwk::float16> &tensors,
                                       const std::vector<npu::tile_fwk::float16> &golden,
                                       const std::vector<RawTensorDataPtr> &inputDataList,
                                       const std::vector<RawTensorDataPtr> &outputDataList) {
    ProgramData::GetInstance().AppendInputs({inputDataList});
    ProgramData::GetInstance().AppendOutputs({outputDataList});
    ProgramData::GetInstance().AppendGoldens({
        RawTensorData::CreateTensor<npu::tile_fwk::float16>(tensors.postOut, golden),
    });
}

inline void BuildAttentionPostGraph(const AttentionPostTensors<npu::tile_fwk::float16> &tensors,
                                     const PostTileConfig &tileConfig,
                                     const QuantTensorWithData &wUvQuant,
                                     const QuantTensorWithData &wOQuant) {
    PostTensors postTensors{tensors.wUv, tensors.wo, wUvQuant.scale.tensor, wUvQuant.smooth.tensor,
                            wOQuant.scale.tensor, wOQuant.smooth.tensor};
    AttentionPostStandalone(tensors.x, postTensors, tileConfig, tensors.postOut);
}

template <typename T>
inline void VerifyAttentionPostResult(const std::vector<T> &golden, RawTensorDataPtr outputData,
                                      const TestPostParams &params, float precision) {
    int b = params.b;
    int s = params.s;
    int h = params.h;

#ifdef BUILD_WITH_CANN
    std::cout << "postOut ====== " << std::endl;
    EXPECT_TRUE(resultCmp<T>(golden, (T *)outputData->data(), precision));
    resultCmp<T>(golden, (T *)outputData->data(), precision, int(b * s * h * precision), 1000, false, false, 16);
#endif
}

template <typename T = npu::tile_fwk::float16, bool nz = true, typename wUvDType = int8_t, bool isSmoothWUv = false,
    typename wODType = int8_t, bool isSmoothWo = false>
void TestAttentionPost(const TestPostParams &params, const PostTileConfig &tileConfig, float precision) {
    SetInterpreterConfig();
    
    bool isQuantWUv = std::is_same<wUvDType, int8_t>::value;
    bool isQuantWo = std::is_same<wODType, int8_t>::value;

    auto shapes = CalculateAttentionPostShapes(params);
    auto tensors = CreateAttentionPostTensors<T, nz, wUvDType, wODType>(shapes);
    
    auto testData = PrepareAttentionPostData<T, wUvDType, wODType>(tensors, shapes, isQuantWUv, isSmoothWUv,
                                                                     isQuantWo, isSmoothWo);
    
    SetupAttentionPostProgram(tensors, testData.golden, testData.inputDataList, testData.outputDataList);

    QuantTensorWithData wUvQuant{isQuantWUv, isSmoothWUv, shapes.wUvScaleShape, shapes.smoothWUvShape,
        "wUvScale", "smoothWUv", "/w_uv_scale.bin", "/smooth_w_uv.bin"};
    CreateQuantTensorAndData(wUvQuant);
    QuantTensorWithData wOQuant{isQuantWo, isSmoothWo, shapes.woScaleShape, shapes.smoothWoShape,
        "woScale", "smoothWo", "/w_o_scale.bin", "/smooth_w_o.bin"};
    CreateQuantTensorAndData(wOQuant);

    BuildAttentionPostGraph(tensors, tileConfig, wUvQuant, wOQuant);

#ifdef BUILD_WITH_CANN
    DevFuncRunner::Run(Program::GetInstance().GetLastFunction(), testData.inputDataList, testData.outputDataList);
#endif

    VerifyAttentionPostResult(testData.golden, testData.outputDataList[0], params, precision);
}

void PerformanceConfig() {
    const int pg_upper_bound = 500000;
    config::SetPassOption(SG_PG_UPPER_BOUND, pg_upper_bound);
    config::SetPassOption(CUBE_NBUFFER_SETTING, std::map<int64_t, int64_t>{{0, 4}});
}

////// fp16, nz, quant
TEST_F(AttentionPostSTest, b16_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {64, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s1_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s2_nz_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

////// bf16, nz, quant
TEST_F(AttentionPostSTest, b16_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {64, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b64_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {64, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b24_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {24, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {24, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s1_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b96_s2_nz_bf16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {96, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, npu::tile_fwk::bfloat16, false, int8_t, true>(params, tileConfig, 0.006f);
}

////// fp16, nd, quant
TEST_F(AttentionPostSTest, b16_s1_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b16_s2_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s1_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nd_fp16_quant) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, int8_t, true>(params, tileConfig, 0.006f);
}

////// fp16, nz, no quant
TEST_F(AttentionPostSTest, b32_s1_nz_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

////// fp16, nd, no quant
TEST_F(AttentionPostSTest, b16_s1_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b16_s2_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s1_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b32_s2_nd_fp16) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, false, npu::tile_fwk::float16, false, npu::tile_fwk::float16, false>(params, tileConfig, 0.002f);
}

TEST_F(AttentionPostSTest, b16_s1_nz_fp16_quant_all) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {16, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {16, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b32_s2_nz_bf16_quant_all) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {32, 128, 2, 7168, 512, 128};
    PostTileConfig tileConfig = {32, 2};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::bfloat16, true, int8_t, true, int8_t, true>(params, tileConfig, 0.006f);
}

TEST_F(AttentionPostSTest, b48_s1_nz_fp16_quant_all) {
    // b, n, s, h, kvLoraRank, vHeadDim
    TestPostParams params = {48, 128, 1, 7168, 512, 128};
    PostTileConfig tileConfig = {48, 1};

    PerformanceConfig();
    TestAttentionPost<npu::tile_fwk::float16, true, int8_t, true, int8_t, true>(params, tileConfig, 0.006f);
}

} // namespace
