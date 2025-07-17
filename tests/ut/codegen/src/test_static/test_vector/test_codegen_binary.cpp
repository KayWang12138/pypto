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
 * \file test_codegen_binary.cpp
 * \brief Unit test for codegen.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include <vector>
#include <string>
#include "codegen/cloudnpu/codegen_cloudnpu.h"

using namespace npu::tile_fwk;

class TestCodegenBinary : public ::testing::Test {
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

void TestAddBody(std::vector<int> shape, std::vector<int> tile_shape, std::string name) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(tile_shape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    FUNCTION(name, FunctionType::STATIC, {input_a, input_b, output}) {
        output = Add(input_a, input_b);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenBinary, TestCodegenAddDim2) {
    TestAddBody({64, 64}, {64, 64}, "ADD_DIM2");
}

TEST_F(TestCodegenBinary, TestCodegenAddDim3) {
    TestAddBody({8, 8, 8}, {8, 8, 8}, "ADD_DIM3");
}

TEST_F(TestCodegenBinary, TestCodegenAddDim4) {
    TestAddBody({2, 2, 16, 16}, {1, 1, 8, 8}, "ADD_DIM4");
}

TEST_F(TestCodegenBinary, TestCodegenAddDim2ByJson) {
    std::vector<int> shape = {64, 64};
    std::vector<int> tile_shape = {64, 64};
    Program::GetInstance().GetTileShape().SetVecTileShapes(tile_shape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    std::string name = "ADD_DIM2_BY_JSON";
    FUNCTION(name, FunctionType::STATIC, {input_a, input_b, output}) {
        output = Add(input_a, input_b);
    }

    std::string jsonPath = config::LogTopFolder() + "/program.json";
    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(jsonPath, {});
}

void TestAddSBody(std::vector<int> shape, std::vector<int> tile_shape, std::string name) {
    Program::GetInstance().GetTileShape().SetVecTileShapes(tile_shape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, shape, "C");
    Element value(DataType::DT_FP32, 1.5);
    FUNCTION(name, FunctionType::STATIC, {input_a, output}) {
        output = AddS(input_a, value);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCloudNPU codeGen;
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenBinary, TestCodegenAddSDim2) {
    TestAddSBody({19, 90}, {8, 128}, "ADD_DIM2");
}

TEST_F(TestCodegenBinary, TestCodegenAddSDim3) {
    TestAddSBody({5, 19, 90}, {8, 8, 128}, "ADD_DIM3");
}

TEST_F(TestCodegenBinary, TestCodegenAddSDim4) {
    TestAddSBody({2, 2, 20, 20}, {1, 1, 8, 8}, "ADD_DIM4");
}
