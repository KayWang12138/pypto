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
 * \file test_operation_impl.cpp
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
#include "codegen/cloudnpu/codegen_cloudnpu.h"
#include "codegen/litenpu/codegen_litenpu.h"
using namespace npu::tile_fwk;


class LiteNPUCodegenUnary : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SwtBuildStatic(true);
    }

    void TearDown() override {}
};


TEST_F(LiteNPUCodegenUnary, test_exp_lite_002) {
    PROGRAM("EXP_002") {
        Tensor input(DataType::DT_FP16, {100}, "input");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("EXP_002") {           
        TileShape::Current().SetVecTile({100});
            output = Exp(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "EXP_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodegenUnary, test_Neg_lite_002) {
    PROGRAM("NEG_002") {
        Tensor input(DataType::DT_FP16, {100}, "input");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("NEG_002") {           
        TileShape::Current().SetVecTile({100});
            output = Neg(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "NEG_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}


TEST_F(LiteNPUCodegenUnary, test_Abs_lite_002) {
    PROGRAM("ABS_002") {
        Tensor input(DataType::DT_FP16, {100}, "input");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("ABS_002") {           
        TileShape::Current().SetVecTile({100});
            output = Abs(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ABS_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}


TEST_F(LiteNPUCodegenUnary, test_Sqrt_lite_002) {
    PROGRAM("SQRT_002") {
        Tensor input(DataType::DT_FP16, {100}, "input");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("SQRT_002") {           
        TileShape::Current().SetVecTile({100});
            output = Sqrt(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "SQRT_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodegenUnary, test_Reciprocal_lite_002) {
    PROGRAM("RECIPROCAL_002") {
        Tensor input(DataType::DT_FP16, {100}, "input");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("RECIPROCAL_002") {           
        TileShape::Current().SetVecTile({100});
            output = Reciprocal(input);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "RECIPROCAL_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}



