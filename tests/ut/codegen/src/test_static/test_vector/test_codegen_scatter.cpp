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
 * \file test_codegen_scatter.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include "codegen/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;

class TestCodegenScatter : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }

    void TearDown() override {}
};

constexpr const int SCATER_SHAPE0 = 128;
constexpr const int SCATER_SHAPE1 = 8;
TEST_F(TestCodegenScatter, TestScatterElement) {
    constexpr const int b = 2;
    constexpr const int s = 512;
    constexpr const int nRoutedExperts = 256;
    constexpr const int numExpertsPerTok = 8;
    Program::GetInstance().GetTileShape().SetVecTileShapes(SCATER_SHAPE0, SCATER_SHAPE1);

    Tensor cnts(DT_FP32, {b * s, nRoutedExperts}, "cnts");
    Tensor topkIds(DT_INT32, {b * s, numExpertsPerTok}, "topkIds");

    Tensor res;

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::string funcName = "SCATTER_T";
    FUNCTION(funcName) {
        res = ScatterElement(cnts, topkIds, Element(DataType::DT_FP32, 1.0), 1); // (b*s, nRoutedExperts)
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + funcName);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
