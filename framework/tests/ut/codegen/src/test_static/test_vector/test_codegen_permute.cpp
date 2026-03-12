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
 * \file test_codegen_permute.cpp
 * \brief Unit test for codegen OP_PERMUTE and OP_PERMUTE_ELEMENT.
 */
#include <vector>
#include <string>
using std::string;
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenPermute : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override
    {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetBuildStatic(true);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }

    void TearDown() override {}
};

Function& testPermute(bool isSupportTileTensor, string funcName, std::vector<int> perm)
{
    if (isSupportTileTensor) {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    } else {
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
    }
    constexpr const int DIM0 = 4;
    constexpr const int DIM1 = 8;
    constexpr const int DIM2 = 16;
    std::vector<int64_t> shape = {DIM0, DIM1, DIM2};

    TileShape::Current().SetVecTile({DIM0, DIM1, DIM2});
    Tensor inputTensor(DT_FP32, shape, "input_tensor");
    Tensor outputTensor(DT_FP32, shape, "output_tensor");

    FUNCTION(funcName, {inputTensor}, {outputTensor}) { outputTensor = Permute(inputTensor, perm); }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
    return *function;
}

TEST_F(TestCodegenPermute, TestPermuteStatic) { testPermute(false, "PERMUTE_STATIC_T", {1, 0, 2}); }

TEST_F(TestCodegenPermute, TestPermuteTileTensor)
{
    Function& func = testPermute(true, "PERMUTE_TILETENSOR_T", {1, 0, 2});
    std::string res = GetResultFromCpp(func);
    std::string expect = "TPermute<1, 0, 2, -1, -1, 3>";
    CheckStringExist(expect, res);
}

TEST_F(TestCodegenPermute, TestPermuteElementStatic) { testPermute(false, "PERMUTE_ELEM_STATIC_T", {0, 2, 1}); }

TEST_F(TestCodegenPermute, TestPermuteElementTileTensor)
{
    Function& func = testPermute(true, "PERMUTE_ELEM_TILETENSOR_T", {0, 2, 1});
    std::string res = GetResultFromCpp(func);
    std::string expect = "TPermuteElewise<0, 2, 1, -1, -1, 3>";
    CheckStringExist(expect, res);
}

} // namespace npu::tile_fwk
