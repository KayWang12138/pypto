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
    FunctionConfig funConfig(FunctionType::STATIC);
    ;
    FUNCTION(funcName, funConfig, {inputSrc0, inputSrc1, output}) {
        output = Gather(inputSrc0, inputSrc1, axis);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenGather, TestGatherEle) {
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
    FunctionConfig funConfig(FunctionType::STATIC);
    ;
    FUNCTION(funcName, funConfig, {inputScores, inputTmpScores, outputTensor}) {
        auto topkIdx = std::get<1>(TopK(inputScores, numExpertsPerTopk, -1));       // [b*s,256]->[b*s,8]
        auto topkWeight = GatherElement(inputTmpScores, topkIdx, 1);                // [b*s,8]
        auto topkWeightSum = RowSumSingle(topkWeight, 1);                           // [b*s,8]->[b*s,1]
        auto denominator = AddS(topkWeightSum, Element(DataType::DT_FP32, 1e-20f)); // [b*s,1]
        outputTensor = Div(topkWeight, denominator);                                // [b*s,numExpertsPerTok]
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk