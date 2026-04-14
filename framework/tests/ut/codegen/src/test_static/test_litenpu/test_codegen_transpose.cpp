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

class LiteNPUCodeGenTranspose : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

<<<<<<< HEAD
    void SetUp() override{
=======
    void SetUp() override
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        config::Reset();
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    config::SetBuildStatic(true);
}

    void TearDown() override
{}
}
;

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_2d_axis_neg1_0)
{
    PROGRAM("TRANSPOSE_FP32_2D_AXIS_NEG1_0")
    {
        TileShape::Current().SetVecTile({8, 8});
        Tensor operand(DT_FP32, {32, 32}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_2D_AXIS_NEG1_0") { result = Transpose(operand, {-1, 0}); }
=======
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_2d_axis_neg1_0)
{
    PROGRAM("TRANSPOSE_FP32_2D_AXIS_NEG1_0")
    {
        TileShape::Current().SetVecTile({8, 8});
        Tensor operand(DT_FP32, {32, 32}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_2D_AXIS_NEG1_0") { result = Transpose(operand, {-1, 0}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_2D_AXIS_NEG1_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_2d_axis_1_0)
{
    PROGRAM("TRANSPOSE_FP32_2D_AXIS_1_0")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {16, 16}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_2D_AXIS_1_0") { result = Transpose(operand, {1, 0}); }
=======
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_2d_axis_1_0)
{
    PROGRAM("TRANSPOSE_FP32_2D_AXIS_1_0")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {16, 16}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_2D_AXIS_1_0") { result = Transpose(operand, {1, 0}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_2D_AXIS_1_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp16_2d_axis_1_0)
{
    PROGRAM("TRANSPOSE_FP16_2D_AXIS_1_0")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP16, {16, 16}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_2D_AXIS_1_0") { result = Transpose(operand, {1, 0}); }
=======
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp16_2d_axis_1_0)
{
    PROGRAM("TRANSPOSE_FP16_2D_AXIS_1_0")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP16, {16, 16}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP16_2D_AXIS_1_0") { result = Transpose(operand, {1, 0}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP16_2D_AXIS_1_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_3d_axis_0_2_1)
{
    PROGRAM("TRANSPOSE_FP32_3D_AXIS_0_2_1")
    {
        TileShape::Current().SetVecTile({2, 4, 4});
        Tensor operand(DT_FP32, {2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_3D_AXIS_0_2_1") { result = Transpose(operand, {0, 2, 1}); }
=======
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_3d_axis_0_2_1)
{
    PROGRAM("TRANSPOSE_FP32_3D_AXIS_0_2_1")
    {
        TileShape::Current().SetVecTile({2, 4, 4});
        Tensor operand(DT_FP32, {2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_3D_AXIS_0_2_1") { result = Transpose(operand, {0, 2, 1}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_3D_AXIS_0_2_1");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_3d_axis_2_1_0)
{
    PROGRAM("TRANSPOSE_FP32_3D_AXIS_2_1_0")
    {
        TileShape::Current().SetVecTile({2, 4, 4});
        Tensor operand(DT_FP32, {2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_3D_AXIS_2_1_0") { result = Transpose(operand, {2, 1, 0}); }
=======
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_3d_axis_2_1_0)
{
    PROGRAM("TRANSPOSE_FP32_3D_AXIS_2_1_0")
    {
        TileShape::Current().SetVecTile({2, 4, 4});
        Tensor operand(DT_FP32, {2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_3D_AXIS_2_1_0") { result = Transpose(operand, {2, 1, 0}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_3D_AXIS_2_1_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_4d_axis_0_1_3_2)
{
    PROGRAM("TRANSPOSE_FP32_4D_AXIS_0_1_3_2")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 4});
        Tensor operand(DT_FP32, {2, 2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_4D_AXIS_0_1_3_2") { result = Transpose(operand, {0, 1, 3, 2}); }
=======
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_4d_axis_0_1_3_2)
{
    PROGRAM("TRANSPOSE_FP32_4D_AXIS_0_1_3_2")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 4});
        Tensor operand(DT_FP32, {2, 2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_4D_AXIS_0_1_3_2") { result = Transpose(operand, {0, 1, 3, 2}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_4D_AXIS_0_1_3_2");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_4d_axis_3_2_1_0)
{
    PROGRAM("TRANSPOSE_FP32_4D_AXIS_3_2_1_0")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 4});
        Tensor operand(DT_FP32, {2, 2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_4D_AXIS_3_2_1_0") { result = Transpose(operand, {3, 2, 1, 0}); }
=======
TEST_F(LiteNPUCodeGenTranspose, test_transpose_fp32_4d_axis_3_2_1_0)
{
    PROGRAM("TRANSPOSE_FP32_4D_AXIS_3_2_1_0")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 4});
        Tensor operand(DT_FP32, {2, 2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("TRANSPOSE_FP32_4D_AXIS_3_2_1_0") { result = Transpose(operand, {3, 2, 1, 0}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "TRANSPOSE_FP32_4D_AXIS_3_2_1_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
<<<<<<< HEAD
}
=======
}
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
