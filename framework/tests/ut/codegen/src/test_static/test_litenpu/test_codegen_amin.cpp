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
#include "codegen/npu/litenpu/codegen_litenpu.h"

using namespace npu::tile_fwk;

class LiteNPUCodeGenAmin : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetBuildStatic(true);
    }

    void TearDown() override {}
};

TEST_F(LiteNPUCodeGenAmin, test_amin_fp32_2d_axis_neg1_keepdims_true) {
    PROGRAM("AMIN_FP32_2D_AXIS_NEG1_KEEPDIMS_TRUE") {
        TileShape::Current().SetVecTile({8, 8});
        Tensor operand(DT_FP32, {16, 16}, "operand");
        Tensor result;
        FUNCTION("AMIN_FP32_2D_AXIS_NEG1_KEEPDIMS_TRUE") {
            result = Amin(operand, -1, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMIN_FP32_2D_AXIS_NEG1_KEEPDIMS_TRUE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenAmin, test_amin_fp32_2d_axis_0_keepdims_false) {
    PROGRAM("AMIN_FP32_2D_AXIS_0_KEEPDIMS_FALSE") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {8, 8}, "operand");
        Tensor result;
        FUNCTION("AMIN_FP32_2D_AXIS_0_KEEPDIMS_FALSE") {
            result = Amin(operand, 0, false);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMIN_FP32_2D_AXIS_0_KEEPDIMS_FALSE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenAmin, test_amin_fp32_2d_axis_1_keepdims_true) {
    PROGRAM("AMIN_FP32_2D_AXIS_1_KEEPDIMS_TRUE") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {8, 8}, "operand");
        Tensor result;
        FUNCTION("AMIN_FP32_2D_AXIS_1_KEEPDIMS_TRUE") {
            result = Amin(operand, 1, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMIN_FP32_2D_AXIS_1_KEEPDIMS_TRUE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenAmin, test_amin_fp16_2d_axis_neg1_keepdims_false) {
    PROGRAM("AMIN_FP16_2D_AXIS_NEG1_KEEPDIMS_FALSE") {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP16, {8, 8}, "operand");
        Tensor result;
        FUNCTION("AMIN_FP16_2D_AXIS_NEG1_KEEPDIMS_FALSE") {
            result = Amin(operand, -1, false);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMIN_FP16_2D_AXIS_NEG1_KEEPDIMS_FALSE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenAmin, test_amin_fp32_1d_axis_neg1_keepdims_true) {
    PROGRAM("AMIN_FP32_1D_AXIS_NEG1_KEEPDIMS_TRUE") {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_FP32, {32}, "operand");
        Tensor result;
        FUNCTION("AMIN_FP32_1D_AXIS_NEG1_KEEPDIMS_TRUE") {
            result = Amin(operand, -1, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMIN_FP32_1D_AXIS_NEG1_KEEPDIMS_TRUE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenAmin, test_amin_fp32_3d_axis_1_keepdims_true) {
    PROGRAM("AMIN_FP32_3D_AXIS_1_KEEPDIMS_TRUE") {
        TileShape::Current().SetVecTile({2, 4, 4});
        Tensor operand(DT_FP32, {4, 8, 8}, "operand");
        Tensor result;
        FUNCTION("AMIN_FP32_3D_AXIS_1_KEEPDIMS_TRUE") {
            result = Amin(operand, 1, true);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMIN_FP32_3D_AXIS_1_KEEPDIMS_TRUE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenAmin, test_amin_fp32_4d_axis_2_keepdims_false) {
    PROGRAM("AMIN_FP32_4D_AXIS_2_KEEPDIMS_FALSE") {
        TileShape::Current().SetVecTile({1, 1, 4, 4});
        Tensor operand(DT_FP32, {2, 2, 8, 8}, "operand");
        Tensor result;
        FUNCTION("AMIN_FP32_4D_AXIS_2_KEEPDIMS_FALSE") {
            result = Amin(operand, 2, false);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "AMIN_FP32_4D_AXIS_2_KEEPDIMS_FALSE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}