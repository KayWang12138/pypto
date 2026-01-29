/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_codegen_dyn_bitwisebinary.cpp
 * \brief Unit test for codegen.
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

class TestCodegenDynBitwiseBinary : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, HOST_COMPILE_END);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        IdGen<IdType::FUNCTION>::Inst().SetId(DummyFuncMagic);
    }

    void TearDown() override {}
};

void TestBitwiseTensorDynBody(const std::vector<int64_t> &shape, 
                          const std::vector<int64_t> &tile_shape, 
                          const std::string &name) {
    // 设置Tile形状
    TileShape::Current().SetVecTile(tile_shape);
    
    Tensor input_a(DT_INT16, shape, "A");
    Tensor input_b(DT_INT16, shape, "B");
    Tensor output(DT_INT16, shape, "C");

    FUNCTION(name, {input_a, input_b}, {output}) {
        LOOP(name, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
             if (name == "BitwiseAnd") {
                output = BitwiseAnd(input_a, input_b);
            } else if (name == "BitwiseOr") {
                output = BitwiseOr(input_a, input_b);
            } else if (name == "BitwiseXor") {
                output = BitwiseXor(input_a, input_b);
            }
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name + SUB_FUNC_SUFFIX + "_hiddenfunc0");
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

void TestBitwiseScalarDynBody(const std::vector<int64_t> &shape, 
                          const std::vector<int64_t> &tile_shape, 
                          const std::string &name) {
    // 设置Tile形状
    TileShape::Current().SetVecTile(tile_shape);
    
    Tensor input_a(DT_INT16, shape, "A");
    Tensor output(DT_INT16, shape, "B");
     Element scalar_element(DT_INT16, static_cast<int16_t>(2));

    FUNCTION(name, {input_a}, {output}) {
        LOOP(name, FunctionType::DYNAMIC_LOOP, i, LoopRange(1)) {
            (void)i;
            if (name == "BitwiseAnds") {
                output = BitwiseAnd(input_a, scalar_element);
            } else if (name == "BitwiseOrs") {
                output = BitwiseOr(input_a, scalar_element);
            } else if (name == "BitwiseXors") {
                output = BitwiseXor(input_a, scalar_element);
            }
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + name + SUB_FUNC_SUFFIX + "_hiddenfunc0");
    config::SetCodeGenOption(SUPPORT_DYNAMIC_ALIGNED, true);

    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenCloudNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(TestCodegenDynBitwiseBinary, BitwiseAndLayout) {
    TestBitwiseTensorDynBody({32, 32}, {16, 16}, "BitwiseAnd");
}

TEST_F(TestCodegenDynBitwiseBinary, BitwiseOrLayout) {
    TestBitwiseTensorDynBody({32, 32}, {16, 16}, "BitwiseOr");
}

TEST_F(TestCodegenDynBitwiseBinary, BitwiseXorLayout) {
    TestBitwiseTensorDynBody({32, 32}, {16, 16}, "BitwiseXor");
}

TEST_F(TestCodegenDynBitwiseBinary, BitwiseAndsLayout) {
    TestBitwiseScalarDynBody({32, 32}, {16, 16}, "BitwiseAnds");
}

TEST_F(TestCodegenDynBitwiseBinary, BitwiseOrsLayout) {
    TestBitwiseScalarDynBody({32, 32}, {16, 16}, "BitwiseOrs");
}

TEST_F(TestCodegenDynBitwiseBinary, BitwiseXorsLayout) {
    TestBitwiseScalarDynBody({32, 32}, {16, 16}, "BitwiseXors");
}


} // namespace npu::tile_fwk