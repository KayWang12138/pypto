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
 * \file test_codegen_unsqueeze.cpp
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

class LiteNPUCodeGenUnsqueeze : public testing::Test {
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
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_1d_axis_0)
{
    PROGRAM("UNSQUEEZE_FP32_1D_AXIS_0")
    {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_FP32, {16}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_1D_AXIS_0") { result = Unsqueeze(operand, 0); }
=======
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_1d_axis_0)
{
    PROGRAM("UNSQUEEZE_FP32_1D_AXIS_0")
    {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_FP32, {16}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_1D_AXIS_0") { result = Unsqueeze(operand, 0); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "UNSQUEEZE_FP32_1D_AXIS_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_1d_axis_neg1)
{
    PROGRAM("UNSQUEEZE_FP32_1D_AXIS_NEG1")
    {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_FP32, {16}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_1D_AXIS_NEG1") { result = Unsqueeze(operand, -1); }
=======
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_1d_axis_neg1)
{
    PROGRAM("UNSQUEEZE_FP32_1D_AXIS_NEG1")
    {
        TileShape::Current().SetVecTile({16});
        Tensor operand(DT_FP32, {16}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_1D_AXIS_NEG1") { result = Unsqueeze(operand, -1); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "UNSQUEEZE_FP32_1D_AXIS_NEG1");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_2d_axis_0)
{
    PROGRAM("UNSQUEEZE_FP32_2D_AXIS_0")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_2D_AXIS_0") { result = Unsqueeze(operand, 0); }
=======
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_2d_axis_0)
{
    PROGRAM("UNSQUEEZE_FP32_2D_AXIS_0")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_2D_AXIS_0") { result = Unsqueeze(operand, 0); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "UNSQUEEZE_FP32_2D_AXIS_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_2d_axis_1)
{
    PROGRAM("UNSQUEEZE_FP32_2D_AXIS_1")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_2D_AXIS_1") { result = Unsqueeze(operand, 1); }
=======
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_2d_axis_1)
{
    PROGRAM("UNSQUEEZE_FP32_2D_AXIS_1")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_2D_AXIS_1") { result = Unsqueeze(operand, 1); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "UNSQUEEZE_FP32_2D_AXIS_1");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_2d_axis_neg1)
{
    PROGRAM("UNSQUEEZE_FP32_2D_AXIS_NEG1")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_2D_AXIS_NEG1") { result = Unsqueeze(operand, -1); }
=======
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_2d_axis_neg1)
{
    PROGRAM("UNSQUEEZE_FP32_2D_AXIS_NEG1")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP32, {4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_2D_AXIS_NEG1") { result = Unsqueeze(operand, -1); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "UNSQUEEZE_FP32_2D_AXIS_NEG1");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp16_2d_axis_0)
{
    PROGRAM("UNSQUEEZE_FP16_2D_AXIS_0")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP16, {4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP16_2D_AXIS_0") { result = Unsqueeze(operand, 0); }
=======
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp16_2d_axis_0)
{
    PROGRAM("UNSQUEEZE_FP16_2D_AXIS_0")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor operand(DT_FP16, {4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP16_2D_AXIS_0") { result = Unsqueeze(operand, 0); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "UNSQUEEZE_FP16_2D_AXIS_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_3d_axis_1)
{
    PROGRAM("UNSQUEEZE_FP32_3D_AXIS_1")
    {
        TileShape::Current().SetVecTile({2, 4, 4});
        Tensor operand(DT_FP32, {2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_3D_AXIS_1") { result = Unsqueeze(operand, 1); }
=======
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_3d_axis_1)
{
    PROGRAM("UNSQUEEZE_FP32_3D_AXIS_1")
    {
        TileShape::Current().SetVecTile({2, 4, 4});
        Tensor operand(DT_FP32, {2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_3D_AXIS_1") { result = Unsqueeze(operand, 1); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "UNSQUEEZE_FP32_3D_AXIS_1");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_4d_axis_2)
{
    PROGRAM("UNSQUEEZE_FP32_4D_AXIS_2")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 4});
        Tensor operand(DT_FP32, {2, 2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_4D_AXIS_2") { result = Unsqueeze(operand, 2); }
=======
TEST_F(LiteNPUCodeGenUnsqueeze, test_unsqueeze_fp32_4d_axis_2)
{
    PROGRAM("UNSQUEEZE_FP32_4D_AXIS_2")
    {
        TileShape::Current().SetVecTile({1, 1, 4, 4});
        Tensor operand(DT_FP32, {2, 2, 4, 4}, "operand");
        Tensor result;
        FUNCTION("UNSQUEEZE_FP32_4D_AXIS_2") { result = Unsqueeze(operand, 2); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "UNSQUEEZE_FP32_4D_AXIS_2");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
<<<<<<< HEAD
}
=======
}
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
