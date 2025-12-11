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
 * \file test_codegen_binary.cpp
 * \brief Unit test for codegen.
 */

#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "codegen/codegen.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "test_codegen_utils.h"

namespace npu::tile_fwk {

class TestCodegenBinary : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() { config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true); }

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, false);
        IdGen<IdType::CG_USING_NAME>::Inst().SetId(DummyFuncMagic);
        IdGen<IdType::CG_VAR_NAME>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

void TestAddBody(std::vector<int64_t> shape, std::vector<int64_t> tile_shape, std::string name, bool withBrc = false) {
    TileShape::Current().SetVecTile(tile_shape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "C");

    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, input_b, output}) {
        output = Add(input_a, input_b);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);

    if (withBrc) {
        for (auto &subProgram : function->rootFunc_->programs_) {
            for (auto &op : subProgram.second->Operations(false)) {
                if (op.GetOpcode() == Opcode::OP_ADD) {
                    op.SetAttribute(OpAttributeKey::brcbIdx, 0);
                }
            }
        }
    }
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenBinary, TestCodegenAddDim2) {
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE,false);
    TestAddBody({64, 64}, {64, 64}, "ADD_DIM2", true);
}

TEST_F(TestCodegenBinary, TestCodegenAddDim3) {
    TestAddBody({8, 8, 8}, {8, 8, 8}, "ADD_DIM3");
}

TEST_F(TestCodegenBinary, TestCodegenAddDim4) {
    TestAddBody({2, 2, 16, 16}, {1, 1, 8, 8}, "ADD_DIM4");
}

void TestAddSBody(std::vector<int64_t> shape, std::vector<int64_t> tile_shape, std::string name) {
    TileShape::Current().SetVecTile(tile_shape);
    Tensor input_a(DT_FP32, shape, "A");
    Tensor output(DT_FP32, shape, "C");
    Element value(DataType::DT_FP32, 1.5);
    config::SetBuildStatic(true);
    FUNCTION(name, {input_a, output}) {
        output = Add(input_a, value);
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
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

TEST_F(TestCodegenBinary, TestCodegenAddMulDim4TileTensor) {
    config::SetCodeGenConfig(KEY_CODEGEN_SUPPORT_TILE_TENSOR, true);
    config::SetCodeGenConfig(KEY_CODEGEN_NEED_COMPILE, false);
    TileShape::Current().SetVecTile(1, 1, 16, 16);
    std::vector<int64_t> shape = {1, 1, 16, 16};
    Tensor input_a(DT_FP32, shape, "A");
    Tensor input_b(DT_FP32, shape, "B");
    Tensor output(DT_FP32, shape, "OUT");

    config::SetBuildStatic(true);
    std::string name = "AddMulDim4_TILETENSOR";
    FUNCTION(name, {input_a, input_b, output}) {
        Tensor tmp_c(DT_FP32, shape, "TEMP_C");
        tmp_c = Add(input_a, input_b);
        output = Mul(input_a, tmp_c);
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name);
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
} // namespace npu::tile_fwk
