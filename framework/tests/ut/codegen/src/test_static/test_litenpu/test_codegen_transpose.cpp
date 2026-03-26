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
 * \file test_codegen_compare.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "interface/interpreter/calc.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/interpreter/calc.h"
#include "codegen/codegen.h"
#include "codegen/litenpu/codegen_litenpu.h"

using namespace npu::tile_fwk;

class LiteNPUCodeGenTranspose : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

TEST_F(LiteNPUCodeGenTranspose, Test_Transpose_View_Assemble_Unsqueeze_Reshape_Lite) {
    PROGRAM("TestTranspose"){
        TileShape::Current().SetVecTile(8, 8);
        Tensor operand(DT_FP32, {32, 32}, "operand");
        Tensor result;
        std::vector<int64_t> new_shape = {16, 16};
        std::vector<int64_t> offsetsView = {0, 4};
        Tensor resultAssemble(DataType::DT_FP32, {32, 32}, "resultAssemble")
        FUNCTION("TestTranspose") {
            result = Transpose(operand, {-1, 0});
            result = View(result, new_shape, offsetsView);
            std::vector<SymbolicScalar> offsetsAssemble(2);
            offsetsAssemble[0] = 1;
            offsetsAssemble[1] = 1;
            Assemble(result, offsetsAssemble, resultAssemble);
            result = Unsqueeze(resultAssemble, 0);
            result = Reshape(result, {1024});
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TestTranspose");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::codegen_litenpu codeGen(ctx);
    codeGen.GenCode(*function, {});

}