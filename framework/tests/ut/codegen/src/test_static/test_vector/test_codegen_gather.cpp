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
 * \file test_codegen_gather.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenGather : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

constexpr const int GATHER_SHAPE0 = 16;
constexpr const int GATHER_SHAPE1 = 32;

TEST_F(TestCodegenGather, TestGather) {
    constexpr const int S2 = 32;
    constexpr const int D = 64;
    constexpr const int B = 1;
    constexpr const int S = 32;
    std::vector<int64_t> shape0 = {S2, D};
    std::vector<int64_t> shape1 = {B, S};
    int axis = 0;
    std::vector<int64_t> shape2 = {B, S, D};

    TileShape::Current().SetVecTile({1, GATHER_SHAPE0, GATHER_SHAPE1});

    Tensor inputSrc0(DT_FP32, shape0, "x");
    Tensor inputSrc1(DT_INT32, shape1, "indices");
    Tensor output(DT_FP32, shape2, "output");

    ConfigManager::Instance();
    std::string funcName = "GATHER_T";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {inputSrc0, inputSrc1, output}) {
        output = Gather(inputSrc0, inputSrc1, axis);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

Function& testGatherEle(bool isSupportTileTensor){
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
        config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    }
    constexpr const int32_t nRoutedExperts = 32;
    constexpr const int32_t numExpertsPerTopk = 8;
    constexpr const int32_t S = 1;
    constexpr const int32_t B = 2;

    std::vector<int64_t> inputShape = {B * S, nRoutedExperts};
    std::vector<int64_t> outputShape = {B * S, numExpertsPerTopk};
    TileShape::Current().SetVecTile({GATHER_SHAPE0, GATHER_SHAPE1});
    Tensor inputScores(DT_FP32, inputShape, "input_scores");
    Tensor inputTmpScores(DT_FP32, inputShape, "input_tmp_scores");
    Tensor outputTensor(DT_FP32, outputShape, "output_tensor");

    std::string funcName = "GATHER_ELEMET_T";
    config::SetBuildStatic(true);
    FUNCTION(funcName, {inputScores, inputTmpScores, outputTensor}) {
        auto topkIdx = std::get<1>(TopK(inputScores, numExpertsPerTopk, -1));       // [b*s,256]->[b*s,8]
        auto topkWeight = GatherElements(inputTmpScores, topkIdx, 1);                // [b*s,8]
        auto topkWeightSum = Sum(topkWeight, 1);                           // [b*s,8]->[b*s,1]
        auto denominator = Add(topkWeightSum, Element(DataType::DT_FP32, 1e-20f)); // [b*s,1]
        outputTensor = Div(topkWeight, denominator);                                // [b*s,numExpertsPerTok]
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    return *function;
}
TEST_F(TestCodegenGather, TestGatherEle) {
    testGatherEle(false);
}

TEST_F(TestCodegenGather, TestGatherEleTileTensor) {
    Function& func = testGatherEle(true);
    std::string res = GetResultFromCpp(func);
    std::string expect = R"!!!(#include "TileOpImpl.h"

// funcHash: 2101778897941530938

extern "C" [aicore] void TENSOR_GATHER_ELEMET_T_2_0_4503599627370496(__gm__ GMTensorInfo* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
float __ubuf__ *UB_S0_E256 = (float __ubuf__ *)get_imm(0x0); // size: 0x100
float *UB_S0_E256_T = (float *)get_imm(0x0); // size: 0x100
float __ubuf__ *UB_S1472_E1728 = (float __ubuf__ *)get_imm(0x5c0); // size: 0x100
float *UB_S1472_E1728_T = (float *)get_imm(0x5c0); // size: 0x100
float __ubuf__ *UB_S256_E1280 = (float __ubuf__ *)get_imm(0x100); // size: 0x400
float *UB_S256_E1280_T = (float *)get_imm(0x100); // size: 0x400
float __ubuf__ *UB_S1280_E1408 = (float __ubuf__ *)get_imm(0x500); // size: 0x80
float *UB_S1280_E1408_T = (float *)get_imm(0x500); // size: 0x80
int32_t __ubuf__ *UB_S1408_E1472 = (int32_t __ubuf__ *)get_imm(0x580); // size: 0x40
int32_t *UB_S1408_E1472_T = (int32_t *)get_imm(0x580); // size: 0x40
float __ubuf__ *UB_S1728_E1792 = (float __ubuf__ *)get_imm(0x6c0); // size: 0x40
float *UB_S1728_E1792_T = (float *)get_imm(0x6c0); // size: 0x40
float __ubuf__ *UB_S1792_E1856 = (float __ubuf__ *)get_imm(0x700); // size: 0x40
float *UB_S1792_E1856_T = (float *)get_imm(0x700); // size: 0x40
float __ubuf__ *UB_S1856_E1888 = (float __ubuf__ *)get_imm(0x740); // size: 0x20
float *UB_S1856_E1888_T = (float *)get_imm(0x740); // size: 0x20
using GMTileTensorFP32Dim2_8 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_7 = TileTensor<float, StaticLayout2Dim<1, 8, 1, 8>, Hardware::UB>;
using UBTileTensorFP32Dim2_6 = TileTensor<float, StaticLayout2Dim<2, 1, 2, 8>, Hardware::UB>;
using UBTileTensorFP32Dim2_5 = TileTensor<float, StaticLayout2Dim<2, 8, 2, 8>, Hardware::UB>;
using UBTileTensorINT32Dim2_4 = TileTensor<int32_t, StaticLayout2Dim<2, 8, 2, 8>, Hardware::UB>;
using UBTileTensorFP32Dim2_3 = TileTensor<float, StaticLayout2Dim<2, 16, 2, 16>, Hardware::UB>;
using UBTileTensorFP32Dim2_2 = TileTensor<float, StaticLayout2Dim<2, 128, 2, 128>, Hardware::UB>;
using GMTileTensorFP32Dim2_1 = TileTensor<__gm__ float, DynLayout2Dim, Hardware::GM>;
using UBTileTensorFP32Dim2_0 = TileTensor<float, StaticLayout2Dim<2, 32, 2, 32>, Hardware::UB>;
GMTileTensorFP32Dim2_8 gmTensor_18((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 2)->Addr, DynLayout2Dim(Shape2Dim(2, 8), Stride2Dim(8, 1)));
UBTileTensorFP32Dim2_7 ubTensor_14((uint64_t)UB_S1856_E1888_T);
UBTileTensorINT32Dim2_4 ubTensor_8((uint64_t)UB_S1408_E1472_T);
UBTileTensorFP32Dim2_5 ubTensor_10((uint64_t)UB_S1728_E1792_T);
UBTileTensorFP32Dim2_3 ubTensor_6((uint64_t)UB_S1280_E1408_T);
UBTileTensorFP32Dim2_5 ubTensor_16((uint64_t)UB_S1792_E1856_T);
UBTileTensorFP32Dim2_2 ubTensor_4((uint64_t)UB_S256_E1280_T);
GMTileTensorFP32Dim2_1 gmTensor_3((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 1)->Addr, DynLayout2Dim(Shape2Dim(2, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_6 ubTensor_13((uint64_t)UB_S1792_E1856_T);
UBTileTensorFP32Dim2_0 ubTensor_2((uint64_t)UB_S1472_E1728_T);
GMTileTensorFP32Dim2_1 gmTensor_1((__gm__ float*)((__gm__ GMTensorInfo*)(param) + 0)->Addr, DynLayout2Dim(Shape2Dim(2, 32), Stride2Dim(32, 1)));
UBTileTensorFP32Dim2_0 ubTensor_0((uint64_t)UB_S0_E256_T);
SUBKERNEL_PHASE1
TLoad(ubTensor_0, gmTensor_1, Coord2Dim(0, 0));
set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
TLoad(ubTensor_2, gmTensor_3, Coord2Dim(0, 0));
SUBKERNEL_PHASE2
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
TBitSort<1, 0, 1>(ubTensor_4, ubTensor_0);
pipe_barrier(PIPE_V);
TMrgSort<1, 8, 1>(ubTensor_6, ubTensor_4);
pipe_barrier(PIPE_V);
TExtract<8, 1, 1>(ubTensor_8, ubTensor_6);
pipe_barrier(PIPE_V);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
TgatherElement<4>(ubTensor_10, ubTensor_2, ubTensor_8);
pipe_barrier(PIPE_V);
TRowSumSingle(ubTensor_13, ubTensor_10, ubTensor_14);
pipe_barrier(PIPE_V);
TileOp::Tadds_<float, 1, 1, 2, 1, /*DS*/ 1, 2, 8, /*S0S*/ 1, 2, 8>((__ubuf__ float*)UB_S1792_E1856, (__ubuf__ float*)UB_S1792_E1856, (float)9.99999968e-21);
pipe_barrier(PIPE_V);
TExpand(ubTensor_16, ubTensor_13, 3);
pipe_barrier(PIPE_V);
TileOp::Tdiv_<float, /*OS0*/ 1, 1, 2, 8, /*OS1*/ 1, 1, 2, 8, /*DS*/ 1, 1, 2, 8, /*S0*/ 1, 1, 2, 8, /*S1*/ 1, 1, 2, 8>((__ubuf__ float*)UB_S1792_E1856, (__ubuf__ float*)UB_S1728_E1792, (__ubuf__ float*)UB_S1792_E1856);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
TStore(gmTensor_18, ubTensor_16, Coord2Dim(0, 0));
}
)!!!";

    EXPECT_EQ(res, expect);
}

} // namespace npu::tile_fwk