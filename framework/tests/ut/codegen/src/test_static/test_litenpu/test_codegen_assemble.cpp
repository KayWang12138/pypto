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
 * \file test_codegen_assemble.cpp
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

class LiteNPUCodeGenAssemble : public testing::Test {
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
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_2d_axis_0)
{
    PROGRAM("ASSEMBLE_FP32_2D_AXIS_0")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_2d_axis_0)
{
    PROGRAM("ASSEMBLE_FP32_2D_AXIS_0")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({4, 4});
        Tensor input1(DT_FP32, {4, 8}, "input1");
        Tensor input2(DT_FP32, {4, 8}, "input2");
        Tensor result;
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_FP32_2D_AXIS_0")
        {
            result = Assemble({{input1, { 0 }}, { input2, { 4 } }});
        }
=======
        FUNCTION("ASSEMBLE_FP32_2D_AXIS_0") { result = Assemble({{input1, {0}}, {input2, {4}}}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_2D_AXIS_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_2d_axis_1)
{
    PROGRAM("ASSEMBLE_FP32_2D_AXIS_1")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_2d_axis_1)
{
    PROGRAM("ASSEMBLE_FP32_2D_AXIS_1")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({4, 4});
        Tensor input1(DT_FP32, {8, 4}, "input1");
        Tensor input2(DT_FP32, {8, 4}, "input2");
        Tensor result;
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_FP32_2D_AXIS_1")
        {
            result = Assemble({{input1, { 0 }}, { input2, { 4 } }});
        }
=======
        FUNCTION("ASSEMBLE_FP32_2D_AXIS_1") { result = Assemble({{input1, {0}}, {input2, {4}}}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_2D_AXIS_1");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp16_2d_axis_0)
{
    PROGRAM("ASSEMBLE_FP16_2D_AXIS_0")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp16_2d_axis_0)
{
    PROGRAM("ASSEMBLE_FP16_2D_AXIS_0")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({4, 4});
        Tensor input1(DT_FP16, {4, 8}, "input1");
        Tensor input2(DT_FP16, {4, 8}, "input2");
        Tensor result;
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_FP16_2D_AXIS_0")
        {
            result = Assemble({{input1, { 0 }}, { input2, { 4 } }});
            ;
=======
        FUNCTION("ASSEMBLE_FP16_2D_AXIS_0")
        {
            result = Assemble({{input1, {0}}, {input2, {4}}});
            ;
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP16_2D_AXIS_0");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_3d_axis_1)
{
    PROGRAM("ASSEMBLE_FP32_3D_AXIS_1")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_3d_axis_1)
{
    PROGRAM("ASSEMBLE_FP32_3D_AXIS_1")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({2, 4, 4});
        Tensor input1(DT_FP32, {2, 4, 8}, "input1");
        Tensor input2(DT_FP32, {2, 4, 8}, "input2");
        Tensor result;
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_FP32_3D_AXIS_1")
        {
            result = Assemble({{input1, { 0, 0 }}, { input2, { 0, 4 } }});
        }
=======
        FUNCTION("ASSEMBLE_FP32_3D_AXIS_1") { result = Assemble({{input1, {0, 0}}, {input2, {0, 4}}}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_3D_AXIS_1");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_4d_axis_2)
{
    PROGRAM("ASSEMBLE_FP32_4D_AXIS_2")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_4d_axis_2)
{
    PROGRAM("ASSEMBLE_FP32_4D_AXIS_2")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({1, 1, 4, 4});
        Tensor input1(DT_FP32, {2, 2, 4, 8}, "input1");
        Tensor input2(DT_FP32, {2, 2, 4, 8}, "input2");
        Tensor result;
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_FP32_4D_AXIS_2")
        {
            result = Assemble({{input1, { 0, 0 }}, { input2, { 0, 4 } }});
        }
=======
        FUNCTION("ASSEMBLE_FP32_4D_AXIS_2") { result = Assemble({{input1, {0, 0}}, {input2, {0, 4}}}); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_4D_AXIS_2");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_multiple_tensors_axis_0)
{
    PROGRAM("ASSEMBLE_FP32_MULTIPLE_TENSORS_AXIS_0")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_fp32_multiple_tensors_axis_0)
{
    PROGRAM("ASSEMBLE_FP32_MULTIPLE_TENSORS_AXIS_0")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({4, 4});
        Tensor input1(DT_FP32, {2, 8}, "input1");
        Tensor input2(DT_FP32, {2, 8}, "input2");
        Tensor input3(DT_FP32, {2, 8}, "input3");
        Tensor result;
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_FP32_MULTIPLE_TENSORS_AXIS_0")
        {
            result = result = Assemble({{input1, { 0 }}, {input2, { 4 }}, { input3, { 8 } }});
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_MULTIPLE_TENSORS_AXIS_0");
=======
        FUNCTION("ASSEMBLE_FP32_MULTIPLE_TENSORS_AXIS_0")
        {
            result = result = Assemble({{input1, {0}}, {input2, {4}}, {input3, {8}}});
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_FP32_MULTIPLE_TENSORS_AXIS_0");
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp32_2d_parallel_true)
{
    PROGRAM("ASSEMBLE_CALL_FP32_2D_PARALLEL_TRUE")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp32_2d_parallel_true)
{
    PROGRAM("ASSEMBLE_CALL_FP32_2D_PARALLEL_TRUE")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({1, 2});
        Tensor input1(DT_FP32, {2, 2}, "input1");
        Tensor input2(DT_FP32, {2, 2}, "input2");
        Tensor out(DT_FP32, {4, 4}, "out");
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_CALL_FP32_2D_PARALLEL_TRUE")
        {
            Assemble({{input1, { 0, 0 }}, { input2, { 2, 2 } }}, out, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_CALL_FP32_2D_PARALLEL_TRUE");
=======
        FUNCTION("ASSEMBLE_CALL_FP32_2D_PARALLEL_TRUE") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, true); }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_CALL_FP32_2D_PARALLEL_TRUE");
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp32_2d_parallel_false)
{
    PROGRAM("ASSEMBLE_CALL_FP32_2D_PARALLEL_FALSE")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp32_2d_parallel_false)
{
    PROGRAM("ASSEMBLE_CALL_FP32_2D_PARALLEL_FALSE")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({1, 2});
        Tensor input1(DT_FP32, {2, 2}, "input1");
        Tensor input2(DT_FP32, {2, 2}, "input2");
        Tensor out(DT_FP32, {4, 4}, "out");
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_CALL_FP32_2D_PARALLEL_FALSE")
        {
            Assemble({{input1, { 0, 0 }}, { input2, { 2, 2 } }}, out, false);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_CALL_FP32_2D_PARALLEL_FALSE");
=======
        FUNCTION("ASSEMBLE_CALL_FP32_2D_PARALLEL_FALSE") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, false); }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_CALL_FP32_2D_PARALLEL_FALSE");
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp32_1d)
{
    PROGRAM("ASSEMBLE_CALL_FP32_1D")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp32_1d)
{
    PROGRAM("ASSEMBLE_CALL_FP32_1D")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({2});
        Tensor input1(DT_FP32, {4}, "input1");
        Tensor input2(DT_FP32, {4}, "input2");
        Tensor out(DT_FP32, {8}, "out");
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_CALL_FP32_1D")
        {
            Assemble({{input1, { 0 }}, { input2, { 4 } }}, out, false);
        }
=======
        FUNCTION("ASSEMBLE_CALL_FP32_1D") { Assemble({{input1, {0}}, {input2, {4}}}, out, false); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_CALL_FP32_1D");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp32_3d)
{
    PROGRAM("ASSEMBLE_CALL_FP32_3D")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp32_3d)
{
    PROGRAM("ASSEMBLE_CALL_FP32_3D")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({1, 1, 2});
        Tensor input1(DT_FP32, {2, 2, 2}, "input1");
        Tensor input2(DT_FP32, {2, 2, 2}, "input2");
        Tensor out(DT_FP32, {4, 4, 4}, "out");
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_CALL_FP32_3D")
        {
            Assemble({{input1, { 0, 0, 0 }}, { input2, { 2, 2, 2 } }}, out, false);
        }
=======
        FUNCTION("ASSEMBLE_CALL_FP32_3D") { Assemble({{input1, {0, 0, 0}}, {input2, {2, 2, 2}}}, out, false); }
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_CALL_FP32_3D");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

<<<<<<< HEAD
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp16_2d_parallel_true)
{
    PROGRAM("ASSEMBLE_CALL_FP16_2D_PARALLEL_TRUE")
    {
=======
TEST_F(LiteNPUCodeGenAssemble, test_assemble_call_fp16_2d_parallel_true)
{
    PROGRAM("ASSEMBLE_CALL_FP16_2D_PARALLEL_TRUE")
    {
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
        TileShape::Current().SetVecTile({1, 2});
        Tensor input1(DT_FP16, {2, 2}, "input1");
        Tensor input2(DT_FP16, {2, 2}, "input2");
        Tensor out(DT_FP16, {4, 4}, "out");
<<<<<<< HEAD
        FUNCTION("ASSEMBLE_CALL_FP16_2D_PARALLEL_TRUE")
        {
            Assemble({{input1, { 0, 0 }}, { input2, { 2, 2 } }}, out, true);
        }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_CALL_FP16_2D_PARALLEL_TRUE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
=======
        FUNCTION("ASSEMBLE_CALL_FP16_2D_PARALLEL_TRUE") { Assemble({{input1, {0, 0}}, {input2, {2, 2}}}, out, true); }
    }
    auto function =
        Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "ASSEMBLE_CALL_FP16_2D_PARALLEL_TRUE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
>>>>>>> 7eb5107b3132f71277db6ee374fffa12949f5481
