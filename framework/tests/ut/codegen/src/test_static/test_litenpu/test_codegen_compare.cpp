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

class LiteNPUCodeGenSigmoid : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }

    void TearDown() override {}
};

TEST_F(LiteNPUCodeGenCompare, test_compare_eq) {
    PROGRAM("COMPARE_EQ") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_EQ") {
            output = Compare(input, other, OpType::EQ, OutType::BOOL);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_ge) {
    PROGRAM("COMPARE_GE") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_GE") {
            output = Compare(input, other, OpType::GE, OutType::BOOL);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_GE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_gt) {
    PROGRAM("COMPARE_GT") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_GT") {
            output = Compare(input, other, OpType::GT, OutType::BOOL);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_GT");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_le) {
    PROGRAM("COMPARE_LE") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_LE") {
            output = Compare(input, other, OpType::LE, OutType::BOOL);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_LE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_lt) {
    PROGRAM("COMPARE_LT") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_LT") {
            output = Compare(input, other, OpType::LT, OutType::BOOL);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_LT");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_ne) {
    PROGRAM("COMPARE_NE") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_NE") {
            output = Compare(input, other, OpType::NE, OutType::BOOL);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_NE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}