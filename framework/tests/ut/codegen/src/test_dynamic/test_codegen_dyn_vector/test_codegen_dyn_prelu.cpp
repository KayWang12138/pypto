/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_prelu.cpp
 * \brief Unit test for prelu codegen.
 */

#include "gtest/gtest.h"

#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation.h"
#include "tilefwk/data_type.h"
#include "codegen/codegen.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/cloudnpu/codegen_op_cloudnpu.h"
#include "test_codegen_utils.h"
#include "test_codegen_common.h"

namespace npu::tile_fwk {

class TestCodegenDynPrelu : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }

    void TearDown() override {}
};

TEST_F(TestCodegenDynPrelu, PreluNormal) {
    std::vector<int64_t> shape = {32, 256};
    TileShape::Current().SetVecTile(shape);
    
    Tensor input(DT_FP32, shape, "input");
    Tensor weight(DT_FP32, {shape[1]}, "weight");
    Tensor output(DT_FP32, shape, "output");

    std::string funcName = "PRELU_NORMAL";

    FUNCTION(funcName, {input, weight}, {output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = PReLU(input, weight);
        }
    }

    auto rawName = FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX;
    auto function = Program::GetInstance().GetFunctionByRawName(rawName);
    
    ASSERT_NE(function, nullptr) << "Failed to find function: " << rawName;

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynPrelu, PreluFP16) {
    std::vector<int64_t> shape = {16, 128};
    TileShape::Current().SetVecTile(shape);
    
    Tensor input(DT_FP16, shape, "input");
    Tensor weight(DT_FP16, {shape[1]}, "weight");
    Tensor output(DT_FP16, shape, "output");

    std::string funcName = "PRELU_FP16";

    FUNCTION(funcName, {input, weight}, {output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = PReLU(input, weight);
        }
    }

    auto rawName = FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX;
    auto function = Program::GetInstance().GetFunctionByRawName(rawName);
    
    ASSERT_NE(function, nullptr) << "Failed to find function: " << rawName;

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynPrelu, PreluBF16) {
    std::vector<int64_t> shape = {8, 64};
    TileShape::Current().SetVecTile(shape);
    
    Tensor input(DT_BF16, shape, "input");
    Tensor weight(DT_BF16, {shape[1]}, "weight");
    Tensor output(DT_BF16, shape, "output");

    std::string funcName = "PRELU_BF16";

    FUNCTION(funcName, {input, weight}, {output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = PReLU(input, weight);
        }
    }

    auto rawName = FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX;
    auto function = Program::GetInstance().GetFunctionByRawName(rawName);
    
    ASSERT_NE(function, nullptr) << "Failed to find function: " << rawName;

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynPrelu, Prelu4D) {
    std::vector<int64_t> shape = {2, 64, 8, 8};
    TileShape::Current().SetVecTile(shape);
    
    Tensor input(DT_FP32, shape, "input");
    Tensor weight(DT_FP32, {shape[1]}, "weight");
    Tensor output(DT_FP32, shape, "output");

    std::string funcName = "PRELU_4D";

    FUNCTION(funcName, {input, weight}, {output}) {
        LOOP(funcName, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            output = PReLU(input, weight);
        }
    }

    auto rawName = FUNCTION_PREFIX + funcName + SUB_FUNC_SUFFIX + HIDDEN_FUNC_SUFFIX;
    auto function = Program::GetInstance().GetFunctionByRawName(rawName);
    
    ASSERT_NE(function, nullptr) << "Failed to find function: " << rawName;

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

} // namespace npu::tile_fwk
