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

class LiteNPUCodeGenCompare : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override
    {
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetBuildStatic(true);
    }

    void TearDown() override {}
};

TEST_F(LiteNPUCodeGenCompare, test_compare_eq_001)
{
    PROGRAM("COMPARE_EQ")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_EQ") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_eq_002)
{
    PROGRAM("COMPARE_EQ")
    {
        TileShape::Current().SetVecTile({8, 8});
        Tensor input(DT_FP32, {8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        auto output = Tensor(DT_FP32, {8, 8}, "output");
        FUNCTION("COMPARE_EQ") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_eq_003)
{
    PROGRAM("COMPARE_EQ")
    {
        TileShape::Current().SetVecTile({8, 8, 8});
        Tensor input(DT_FP32, {8, 8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        // std::vector<int64_t> dstShape = {8, 8, 8};
        auto output = Tensor(DT_FP32, {8, 8, 8}, "output");
        FUNCTION("COMPARE_EQ") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_eq_004)
{
    PROGRAM("COMPARE_EQ")
    {
        TileShape::Current().SetVecTile({4, 4, 4});
        Tensor input(DT_FP32, {8, 8, 1}, "input");
        Tensor other(DT_FP32, {1, 8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8, 8};
        Tensor output;
        FUNCTION("COMPARE_EQ") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_eq_005)
{
    PROGRAM("COMPARE_EQ")
    {
        TileShape::Current().SetVecTile({4, 4, 4, 4});
        Tensor input(DT_FP32, {8, 8, 8, 1}, "input");
        Tensor other(DT_FP32, {1, 8, 8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8, 8, 8};
        Tensor output;
        FUNCTION("COMPARE_EQ") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_ge_001)
{
    PROGRAM("COMPARE_GE")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {1, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_GE") { output = Compare(input, other, OpType::GE, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_GE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_ge_002)
{
    PROGRAM("COMPARE_GE")
    {
        TileShape::Current().SetVecTile({8, 8, 8, 8});
        Tensor input(DT_FP32, {8, 8, 8, 8}, "input");
        Element other(DataType::DT_FP32, 1.0);
        std::vector<int64_t> dstShape = {8, 8, 8, 8};
        auto output = Tensor(DT_FP32, {8, 8, 8, 8}, "output");
        FUNCTION("COMPARE_GE") { output = Compare(input, other, OpType::GE, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_GE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_gt)
{
    PROGRAM("COMPARE_GT")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_GT") { output = Compare(input, other, OpType::GT, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_GT");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_le)
{
    PROGRAM("COMPARE_LE")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_LE") { output = Compare(input, other, OpType::LE, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_LE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_lt)
{
    PROGRAM("COMPARE_LT")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_LT") { output = Compare(input, other, OpType::LT, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_LT");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenCompare, test_compare_ne)
{
    PROGRAM("COMPARE_NE")
    {
        TileShape::Current().SetVecTile({4, 4});
        Tensor input(DT_FP32, {8, 8}, "input");
        Tensor other(DT_FP32, {8, 8}, "other");
        std::vector<int64_t> dstShape = {8, 8};
        Tensor output;
        FUNCTION("COMPARE_NE") { output = Compare(input, other, OpType::NE, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_NE");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// ==============================
// FP16 全部 34 个测试用例
// ==============================

// fp16_001 | 1D 尾轴对齐，w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_001)
{
    PROGRAM("COMPARE_EQ_FP16_001")
    {
        TileShape::Current().SetVecTile({50});
        Tensor input(DT_FP16, {112}, "input");
        Tensor other(DT_FP16, {112}, "other");
        auto output = Tensor(DT_BOOL, {112}, "output");
        FUNCTION("COMPARE_EQ_FP16_001") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_002 | 1D 张量 + 标量广播，w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_ge_fp16_002)
{
    PROGRAM("COMPARE_GE_FP16_002")
    {
        TileShape::Current().SetVecTile({32});
        Tensor input(DT_FP16, {64}, "input");
        Element other(DataType::DT_FP16, 1.0f);
        auto output = Tensor(DT_BOOL, {64}, "output");
        FUNCTION("COMPARE_GE_FP16_002") { output = Compare(input, other, OpType::GE, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_GE_FP16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_003 | 1D 无广播，纯 w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_003)
{
    PROGRAM("COMPARE_EQ_FP16_003")
    {
        TileShape::Current().SetVecTile({16});
        Tensor input(DT_FP16, {32}, "input");
        Tensor other(DT_FP16, {32}, "other");
        auto output = Tensor(DT_BOOL, {32}, "output");
        FUNCTION("COMPARE_EQ_FP16_003") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_004 | 1D 标量 + 张量广播，w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_004)
{
    PROGRAM("COMPARE_EQ_FP16_004")
    {
        TileShape::Current().SetVecTile({16, 8});
        Tensor input(DT_FP16, {16, 16}, "input");
        Element other(DataType::DT_FP16, 1.0f);
        auto output = Tensor(DT_BOOL, {16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP16_004") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_005 | 2D 大尺寸，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_005)
{
    PROGRAM("COMPARE_EQ_FP16_005")
    {
        TileShape::Current().SetVecTile({2, 40});
        Tensor input(DT_FP16, {4, 80}, "input");
        Tensor other(DT_FP16, {4, 80}, "other");
        auto output = Tensor(DT_BOOL, {4, 80}, "output");
        FUNCTION("COMPARE_EQ_FP16_005") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_006 | 2D 一维广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_006)
{
    PROGRAM("COMPARE_EQ_FP16_006")
    {
        TileShape::Current().SetVecTile({1, 48});
        Tensor input(DT_FP16, {2, 96}, "input");
        Tensor other(DT_FP16, {1, 96}, "other");
        auto output = Tensor(DT_BOOL, {2, 96}, "output");
        FUNCTION("COMPARE_EQ_FP16_006") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_007 | 2D 一维广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_007)
{
    PROGRAM("COMPARE_EQ_FP16_007")
    {
        TileShape::Current().SetVecTile({5, 16});
        Tensor input(DT_FP16, {5, 1}, "input");
        Tensor other(DT_FP16, {5, 32}, "other");
        auto output = Tensor(DT_BOOL, {5, 32}, "output");
        FUNCTION("COMPARE_EQ_FP16_007") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_008 | 2D 无广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_008)
{
    PROGRAM("COMPARE_EQ_FP16_008")
    {
        TileShape::Current().SetVecTile({2, 64});
        Tensor input(DT_FP16, {3, 128}, "input");
        Tensor other(DT_FP16, {3, 128}, "other");
        auto output = Tensor(DT_BOOL, {3, 128}, "output");
        FUNCTION("COMPARE_EQ_FP16_008") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_009 | 2D 一维广播，仅 h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_009)
{
    PROGRAM("COMPARE_EQ_FP16_009")
    {
        TileShape::Current().SetVecTile({32, 64});
        Tensor input(DT_FP16, {64, 1}, "input");
        Tensor other(DT_FP16, {64, 64}, "other");
        auto output = Tensor(DT_BOOL, {64, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_009") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_010 | 2D 一维广播，仅 w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_010)
{
    PROGRAM("COMPARE_EQ_FP16_010")
    {
        TileShape::Current().SetVecTile({64, 32});
        Tensor input(DT_FP16, {1, 64}, "input");
        Tensor other(DT_FP16, {64, 64}, "other");
        auto output = Tensor(DT_BOOL, {64, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_010") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_011 | 3D 无广播，c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_011)
{
    PROGRAM("COMPARE_EQ_FP16_011")
    {
        TileShape::Current().SetVecTile({1, 32, 32});
        Tensor input(DT_FP16, {2, 64, 64}, "input");
        Tensor other(DT_FP16, {2, 64, 64}, "other");
        auto output = Tensor(DT_BOOL, {2, 64, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_011") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_012 | 3D 一维广播，c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_012)
{
    PROGRAM("COMPARE_EQ_FP16_012")
    {
        TileShape::Current().SetVecTile({1, 1, 24});
        Tensor input(DT_FP16, {2, 1, 48}, "input");
        Tensor other(DT_FP16, {2, 3, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 3, 48}, "output");
        FUNCTION("COMPARE_EQ_FP16_012") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_013 | 3D 一维广播，c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_013)
{
    PROGRAM("COMPARE_EQ_FP16_013")
    {
        TileShape::Current().SetVecTile({1, 32, 24});
        Tensor input(DT_FP16, {3, 64, 1}, "input");
        Tensor other(DT_FP16, {3, 64, 48}, "other");
        auto output = Tensor(DT_BOOL, {3, 64, 48}, "output");
        FUNCTION("COMPARE_EQ_FP16_013") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_014 | 3D 广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_014)
{
    PROGRAM("COMPARE_EQ_FP16_014")
    {
        TileShape::Current().SetVecTile({1, 24, 24});
        Tensor input(DT_FP16, {1, 48, 48}, "input");
        Tensor other(DT_FP16, {1, 1, 48}, "other");
        auto output = Tensor(DT_BOOL, {1, 48, 48}, "output");
        FUNCTION("COMPARE_EQ_FP16_014") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_015 | 3D 无广播，c+h+w切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_015)
{
    PROGRAM("COMPARE_EQ_FP16_015")
    {
        TileShape::Current().SetVecTile({1, 32, 24});
        Tensor input(DT_FP16, {2, 64, 48}, "input");
        Tensor other(DT_FP16, {2, 64, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 64, 48}, "output");
        FUNCTION("COMPARE_EQ_FP16_015") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_016 | 3D 无广播，仅 w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_016)
{
    PROGRAM("COMPARE_EQ_FP16_016")
    {
        TileShape::Current().SetVecTile({3, 32, 32});
        Tensor input(DT_FP16, {3, 32, 64}, "input");
        Tensor other(DT_FP16, {3, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {3, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_016") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_017 | 3D 广播，c+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_017)
{
    PROGRAM("COMPARE_EQ_FP16_017")
    {
        TileShape::Current().SetVecTile({1, 32, 32});
        Tensor input(DT_FP16, {2, 32, 1}, "input");
        Tensor other(DT_FP16, {2, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {2, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_017") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_018 | 3D 广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_018)
{
    PROGRAM("COMPARE_EQ_FP16_018")
    {
        TileShape::Current().SetVecTile({48, 24, 32});
        Tensor input(DT_FP16, {1, 48, 64}, "input");
        Tensor other(DT_FP16, {48, 48, 64}, "other");
        auto output = Tensor(DT_BOOL, {48, 48, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_018") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_019 | 4D 无广播，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_019)
{
    PROGRAM("COMPARE_EQ_FP16_019")
    {
        TileShape::Current().SetVecTile({1, 1, 16, 16});
        Tensor input(DT_FP16, {2, 2, 32, 32}, "input");
        Tensor other(DT_FP16, {2, 2, 32, 32}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 32, 32}, "output");
        FUNCTION("COMPARE_EQ_FP16_019") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_020 | 4D 一维广播，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_020)
{
    PROGRAM("COMPARE_EQ_FP16_020")
    {
        TileShape::Current().SetVecTile({1, 2, 8, 8});
        Tensor input(DT_FP16, {2, 4, 16, 16}, "input");
        Tensor other(DT_FP16, {2, 1, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {2, 4, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP16_020") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_021 | 4D 一维广播，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_021)
{
    PROGRAM("COMPARE_EQ_FP16_021")
    {
        TileShape::Current().SetVecTile({1, 1, 12, 12});
        Tensor input(DT_FP16, {2, 1, 24, 24}, "input");
        Tensor other(DT_FP16, {2, 3, 24, 24}, "other");
        auto output = Tensor(DT_BOOL, {2, 3, 24, 24}, "output");
        FUNCTION("COMPARE_EQ_FP16_021") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_022 | 4D 一维广播，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_022)
{
    PROGRAM("COMPARE_EQ_FP16_022")
    {
        TileShape::Current().SetVecTile({1, 1, 20, 20});
        Tensor input(DT_FP16, {2, 2, 40, 40}, "input");
        Tensor other(DT_FP16, {2, 2, 1, 40}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 40, 40}, "output");
        FUNCTION("COMPARE_EQ_FP16_022") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_023 | 4D 标准尺寸，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_023)
{
    PROGRAM("COMPARE_EQ_FP16_023")
    {
        TileShape::Current().SetVecTile({1, 1, 8, 8});
        Tensor input(DT_FP16, {2, 3, 16, 16}, "input");
        Tensor other(DT_FP16, {2, 3, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {2, 3, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP16_023") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_024 | 跨维度广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_024)
{
    PROGRAM("COMPARE_EQ_FP16_024")
    {
        TileShape::Current().SetVecTile({4, 1});
        Tensor input(DT_FP16, {8, 4}, "input");
        Element other(DT_FP16, 1.0);
        auto output = Tensor(DT_BOOL, {8, 1}, "output");
        FUNCTION("COMPARE_EQ_FP16_024") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_025 | 4D 无广播，仅 n 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_025)
{
    PROGRAM("COMPARE_EQ_FP16_025")
    {
        TileShape::Current().SetVecTile({2, 1, 32, 32});
        Tensor input(DT_FP16, {4, 1, 32, 32}, "input");
        Tensor other(DT_FP16, {4, 1, 32, 32}, "other");
        auto output = Tensor(DT_BOOL, {4, 1, 32, 32}, "output");
        FUNCTION("COMPARE_EQ_FP16_025") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_026 | 4D 无广播，仅 c 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_026)
{
    PROGRAM("COMPARE_EQ_FP16_026")
    {
        TileShape::Current().SetVecTile({1, 4, 16, 16});
        Tensor input(DT_FP16, {1, 8, 16, 16}, "input");
        Tensor other(DT_FP16, {1, 8, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {1, 8, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP16_026") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_027 | 4D 无广播，仅 h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_027)
{
    PROGRAM("COMPARE_EQ_FP16_027")
    {
        TileShape::Current().SetVecTile({2, 2, 24, 48});
        Tensor input(DT_FP16, {2, 2, 48, 48}, "input");
        Tensor other(DT_FP16, {2, 2, 48, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 48, 48}, "output");
        FUNCTION("COMPARE_EQ_FP16_027") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_028 | 4D 无广播，仅 w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_028)
{
    PROGRAM("COMPARE_EQ_FP16_028")
    {
        TileShape::Current().SetVecTile({1, 4, 32, 32});
        Tensor input(DT_FP16, {1, 4, 32, 64}, "input");
        Tensor other(DT_FP16, {1, 4, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {1, 4, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_028") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_029 | 4D 广播，n+c 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_029)
{
    PROGRAM("COMPARE_EQ_FP16_029")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 16});
        Tensor input(DT_FP16, {2, 4, 16, 16}, "input");
        Tensor other(DT_FP16, {2, 1, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {2, 4, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP16_029") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_029");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_030 | 4D 广播，n+h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_030)
{
    PROGRAM("COMPARE_EQ_FP16_030")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 32});
        Tensor input(DT_FP16, {2, 2, 32, 32}, "input");
        Tensor other(DT_FP16, {2, 2, 1, 32}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 32, 32}, "output");
        FUNCTION("COMPARE_EQ_FP16_030") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_030");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_031 | 4D 广播，n+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_031)
{
    PROGRAM("COMPARE_EQ_FP16_031")
    {
        TileShape::Current().SetVecTile({1, 3, 24, 24});
        Tensor input(DT_FP16, {2, 3, 24, 1}, "input");
        Tensor other(DT_FP16, {2, 3, 24, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 3, 24, 48}, "output");
        FUNCTION("COMPARE_EQ_FP16_031") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_031");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_032 | 4D 广播，c+h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_032)
{
    PROGRAM("COMPARE_EQ_FP16_032")
    {
        TileShape::Current().SetVecTile({1, 2, 8, 16});
        Tensor input(DT_FP16, {1, 4, 16, 16}, "input");
        Tensor other(DT_FP16, {1, 1, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {1, 4, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP16_032") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_032");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_033 | 4D 广播，c+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_033)
{
    PROGRAM("COMPARE_EQ_FP16_033")
    {
        TileShape::Current().SetVecTile({2, 1, 32, 32});
        Tensor input(DT_FP16, {2, 2, 32, 1}, "input");
        Tensor other(DT_FP16, {2, 2, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_033") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_033");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_034 | 4D 无广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp16_034)
{
    PROGRAM("COMPARE_EQ_FP16_034")
    {
        TileShape::Current().SetVecTile({1, 1, 24, 32});
        Tensor input(DT_FP16, {1, 1, 48, 64}, "input");
        Tensor other(DT_FP16, {1, 1, 48, 64}, "other");
        auto output = Tensor(DT_BOOL, {1, 1, 48, 64}, "output");
        FUNCTION("COMPARE_EQ_FP16_034") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP16_034");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// ==============================
// FP32 全部 38 个测试用例
// ==============================

// fp32_001 | 1D 尾轴对齐，w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_001)
{
    PROGRAM("COMPARE_EQ_FP32_001")
    {
        TileShape::Current().SetVecTile({50});
        Tensor input(DT_FP32, {112}, "input");
        Tensor other(DT_FP32, {112}, "other");
        auto output = Tensor(DT_BOOL, {112}, "output");
        FUNCTION("COMPARE_EQ_FP32_001") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_002 | 1D 张量 + 标量广播，w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_002)
{
    PROGRAM("COMPARE_EQ_FP32_002")
    {
        TileShape::Current().SetVecTile({8, 8, 8, 4});
        Tensor input(DT_FP32, {8, 8, 8, 8}, "input");
        Element other(DataType::DT_FP32, 1.0);
        auto output = Tensor(DT_BOOL, {8, 8, 8, 8}, "output");
        FUNCTION("COMPARE_EQ_FP32_002") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_003 | 1D 无广播，纯 w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_003)
{
    PROGRAM("COMPARE_EQ_FP32_003")
    {
        TileShape::Current().SetVecTile({16});
        Tensor input(DT_FP32, {32}, "input");
        Tensor other(DT_FP32, {32}, "other");
        auto output = Tensor(DT_BOOL, {32}, "output");
        FUNCTION("COMPARE_EQ_FP32_003") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_004 | 1D 标量 + 张量广播，w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_004)
{
    PROGRAM("COMPARE_EQ_FP32_004")
    {
        TileShape::Current().SetVecTile({8, 8, 4});
        Tensor input(DT_FP32, {8, 8, 8}, "input");
        Element other(DataType::DT_FP32, 1.0);
        auto output = Tensor(DT_BOOL, {8, 8, 8}, "output");
        FUNCTION("COMPARE_EQ_FP32_004") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_005 | 2D 大尺寸，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_005)
{
    PROGRAM("COMPARE_EQ_FP32_005")
    {
        TileShape::Current().SetVecTile({2, 40});
        Tensor input(DT_FP32, {4, 80}, "input");
        Tensor other(DT_FP32, {4, 80}, "other");
        auto output = Tensor(DT_BOOL, {4, 80}, "output");
        FUNCTION("COMPARE_EQ_FP32_005") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_006 | 2D 一维广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_006)
{
    PROGRAM("COMPARE_EQ_FP32_006")
    {
        TileShape::Current().SetVecTile({1, 48});
        Tensor input(DT_FP32, {2, 96}, "input");
        Tensor other(DT_FP32, {1, 96}, "other");
        auto output = Tensor(DT_BOOL, {2, 96}, "output");
        FUNCTION("COMPARE_EQ_FP32_006") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_007 | 2D 一维广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_007)
{
    PROGRAM("COMPARE_EQ_FP32_007")
    {
        TileShape::Current().SetVecTile({2, 16});
        Tensor input(DT_FP32, {5, 1}, "input");
        Tensor other(DT_FP32, {5, 32}, "other");
        auto output = Tensor(DT_BOOL, {5, 32}, "output");
        FUNCTION("COMPARE_EQ_FP32_007") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_008 | 2D 无广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_008)
{
    PROGRAM("COMPARE_EQ_FP32_008")
    {
        TileShape::Current().SetVecTile({2, 64});
        Tensor input(DT_FP32, {3, 128}, "input");
        Tensor other(DT_FP32, {3, 128}, "other");
        auto output = Tensor(DT_BOOL, {3, 128}, "output");
        FUNCTION("COMPARE_EQ_FP32_008") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_009 | 2D 一维广播，仅 h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_009)
{
    PROGRAM("COMPARE_EQ_FP32_009")
    {
        TileShape::Current().SetVecTile({32, 64});
        Tensor input(DT_FP32, {64, 1}, "input");
        Tensor other(DT_FP32, {64, 64}, "other");
        auto output = Tensor(DT_BOOL, {64, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_009") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_010 | 2D 一维广播，仅 w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_010)
{
    PROGRAM("COMPARE_EQ_FP32_010")
    {
        TileShape::Current().SetVecTile({64, 32});
        Tensor input(DT_FP32, {1, 64}, "input");
        Tensor other(DT_FP32, {64, 64}, "other");
        auto output = Tensor(DT_BOOL, {64, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_010") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_011 | 3D 无广播，c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_011)
{
    PROGRAM("COMPARE_EQ_FP32_011")
    {
        TileShape::Current().SetVecTile({1, 32, 32});
        Tensor input(DT_FP32, {2, 64, 64}, "input");
        Tensor other(DT_FP32, {2, 64, 64}, "other");
        auto output = Tensor(DT_BOOL, {2, 64, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_011") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_012 | 3D 一维广播，c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_012)
{
    PROGRAM("COMPARE_EQ_FP32_012")
    {
        TileShape::Current().SetVecTile({1, 1, 24});
        Tensor input(DT_FP32, {2, 1, 48}, "input");
        Tensor other(DT_FP32, {2, 3, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 3, 48}, "output");
        FUNCTION("COMPARE_EQ_FP32_012") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_013 | 3D 一维广播，c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_013)
{
    PROGRAM("COMPARE_EQ_FP32_013")
    {
        TileShape::Current().SetVecTile({2, 32, 24});
        Tensor input(DT_FP32, {3, 64, 1}, "input");
        Tensor other(DT_FP32, {3, 64, 48}, "other");
        auto output = Tensor(DT_BOOL, {3, 64, 48}, "output");
        FUNCTION("COMPARE_EQ_FP32_013") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_014 | 3D 广播，仅 c 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_014)
{
    PROGRAM("COMPARE_EQ_FP32_014")
    {
        TileShape::Current().SetVecTile({8, 1, 48});
        Tensor input(DT_FP32, {16, 1, 48}, "input");
        Tensor other(DT_FP32, {1, 1, 48}, "other");
        auto output = Tensor(DT_BOOL, {16, 1, 48}, "output");
        FUNCTION("COMPARE_EQ_FP32_014") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_015 | 3D 无广播，c+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_015)
{
    PROGRAM("COMPARE_EQ_FP32_015")
    {
        TileShape::Current().SetVecTile({1, 64, 24});
        Tensor input(DT_FP32, {2, 64, 48}, "input");
        Tensor other(DT_FP32, {2, 64, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 64, 48}, "output");
        FUNCTION("COMPARE_EQ_FP32_015") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_016 | 3D 无广播，仅 w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_016)
{
    PROGRAM("COMPARE_EQ_FP32_016")
    {
        TileShape::Current().SetVecTile({3, 32, 32});
        Tensor input(DT_FP32, {3, 32, 64}, "input");
        Tensor other(DT_FP32, {3, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {3, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_016") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_017 | 3D 广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_017)
{
    PROGRAM("COMPARE_EQ_FP32_017")
    {
        TileShape::Current().SetVecTile({2, 16, 16});
        Tensor input(DT_FP32, {2, 32, 1}, "input");
        Tensor other(DT_FP32, {2, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {2, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_017") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_018 | 3D 广播，c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_018)
{
    PROGRAM("COMPARE_EQ_FP32_018")
    {
        TileShape::Current().SetVecTile({24, 24, 32});
        Tensor input(DT_FP32, {1, 48, 64}, "input");
        Tensor other(DT_FP32, {48, 48, 64}, "other");
        auto output = Tensor(DT_BOOL, {48, 48, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_018") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_019 | 4D 无广播，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_019)
{
    PROGRAM("COMPARE_EQ_FP32_019")
    {
        TileShape::Current().SetVecTile({1, 1, 16, 16});
        Tensor input(DT_FP32, {2, 2, 32, 32}, "input");
        Tensor other(DT_FP32, {2, 2, 32, 32}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 32, 32}, "output");
        FUNCTION("COMPARE_EQ_FP32_019") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_020 | 4D 一维广播，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_020)
{
    PROGRAM("COMPARE_EQ_FP32_020")
    {
        TileShape::Current().SetVecTile({1, 2, 8, 8});
        Tensor input(DT_FP32, {2, 4, 16, 16}, "input");
        Tensor other(DT_FP32, {2, 1, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {2, 4, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP32_020") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_021 | 4D 一维广播，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_021)
{
    PROGRAM("COMPARE_EQ_FP32_021")
    {
        TileShape::Current().SetVecTile({1, 1, 12, 12});
        Tensor input(DT_FP32, {2, 1, 24, 24}, "input");
        Tensor other(DT_FP32, {2, 3, 24, 24}, "other");
        auto output = Tensor(DT_BOOL, {2, 3, 24, 24}, "output");
        FUNCTION("COMPARE_EQ_FP32_021") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_022 | 4D 一维广播，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_022)
{
    PROGRAM("COMPARE_EQ_FP32_022")
    {
        TileShape::Current().SetVecTile({1, 1, 20, 20});
        Tensor input(DT_FP32, {2, 2, 40, 40}, "input");
        Tensor other(DT_FP32, {2, 2, 1, 40}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 40, 40}, "output");
        FUNCTION("COMPARE_EQ_FP32_022") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_023 | 4D 标准尺寸，n+c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_023)
{
    PROGRAM("COMPARE_EQ_FP32_023")
    {
        TileShape::Current().SetVecTile({1, 1, 8, 8});
        Tensor input(DT_FP32, {2, 3, 16, 16}, "input");
        Tensor other(DT_FP32, {2, 3, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {2, 3, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP32_023") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_024 | 标量+2D张量，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_024)
{
    PROGRAM("COMPARE_EQ_FP32_024")
    {
        TileShape::Current().SetVecTile({4, 1});
        Tensor input(DT_FP32, {8, 4}, "input");
        Element other(DataType::DT_FP32, 1.0);
        auto output = Tensor(DT_BOOL, {8, 4}, "output");
        FUNCTION("COMPARE_EQ_FP32_024") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_025 | 4D 无广播，仅 n 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_025)
{
    PROGRAM("COMPARE_EQ_FP32_025")
    {
        TileShape::Current().SetVecTile({2, 1, 32, 32});
        Tensor input(DT_FP32, {4, 1, 32, 32}, "input");
        Tensor other(DT_FP32, {4, 1, 32, 32}, "other");
        auto output = Tensor(DT_BOOL, {4, 1, 32, 32}, "output");
        FUNCTION("COMPARE_EQ_FP32_025") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_026 | 4D 无广播，仅 c 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_026)
{
    PROGRAM("COMPARE_EQ_FP32_026")
    {
        TileShape::Current().SetVecTile({1, 4, 16, 16});
        Tensor input(DT_FP32, {1, 8, 16, 16}, "input");
        Tensor other(DT_FP32, {1, 8, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {1, 8, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP32_026") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_027 | 4D 无广播，仅 h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_027)
{
    PROGRAM("COMPARE_EQ_FP32_027")
    {
        TileShape::Current().SetVecTile({2, 2, 24, 48});
        Tensor input(DT_FP32, {2, 2, 48, 48}, "input");
        Tensor other(DT_FP32, {2, 2, 48, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 48, 48}, "output");
        FUNCTION("COMPARE_EQ_FP32_027") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_028 | 4D 无广播，仅 w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_028)
{
    PROGRAM("COMPARE_EQ_FP32_028")
    {
        TileShape::Current().SetVecTile({1, 4, 32, 32});
        Tensor input(DT_FP32, {1, 4, 32, 64}, "input");
        Tensor other(DT_FP32, {1, 4, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {1, 4, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_028") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_029 | 4D 广播，n+c 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_029)
{
    PROGRAM("COMPARE_EQ_FP32_029")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 16});
        Tensor input(DT_FP32, {2, 4, 16, 16}, "input");
        Tensor other(DT_FP32, {2, 1, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {2, 4, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP32_029") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_029");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_030 | 4D 广播，n+h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_030)
{
    PROGRAM("COMPARE_EQ_FP32_030")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 32});
        Tensor input(DT_FP32, {2, 2, 32, 32}, "input");
        Tensor other(DT_FP32, {2, 2, 1, 32}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 32, 32}, "output");
        FUNCTION("COMPARE_EQ_FP32_030") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_030");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_031 | 4D 广播，n+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_031)
{
    PROGRAM("COMPARE_EQ_FP32_031")
    {
        TileShape::Current().SetVecTile({1, 3, 24, 24});
        Tensor input(DT_FP32, {2, 3, 24, 1}, "input");
        Tensor other(DT_FP32, {2, 3, 24, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 3, 24, 48}, "output");
        FUNCTION("COMPARE_EQ_FP32_031") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_031");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_032 | 4D 广播，c+h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_032)
{
    PROGRAM("COMPARE_EQ_FP32_032")
    {
        TileShape::Current().SetVecTile({1, 2, 8, 16});
        Tensor input(DT_FP32, {1, 4, 16, 16}, "input");
        Tensor other(DT_FP32, {1, 1, 16, 16}, "other");
        auto output = Tensor(DT_BOOL, {1, 4, 16, 16}, "output");
        FUNCTION("COMPARE_EQ_FP32_032") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_032");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_033 | 4D 广播，c+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_033)
{
    PROGRAM("COMPARE_EQ_FP32_033")
    {
        TileShape::Current().SetVecTile({2, 1, 32, 32});
        Tensor input(DT_FP32, {2, 2, 32, 1}, "input");
        Tensor other(DT_FP32, {2, 2, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {2, 2, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_033") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_033");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_034 | 4D 无广播，h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_034)
{
    PROGRAM("COMPARE_EQ_FP32_034")
    {
        TileShape::Current().SetVecTile({1, 1, 24, 32});
        Tensor input(DT_FP32, {1, 1, 48, 64}, "input");
        Tensor other(DT_FP32, {1, 1, 48, 64}, "other");
        auto output = Tensor(DT_BOOL, {1, 1, 48, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_034") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_034");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_035 | 4D 无广播，n+c+h 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_035)
{
    PROGRAM("COMPARE_EQ_FP32_035")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 48});
        Tensor input(DT_FP32, {2, 4, 32, 48}, "input");
        Tensor other(DT_FP32, {2, 4, 32, 48}, "other");
        auto output = Tensor(DT_BOOL, {2, 4, 32, 48}, "output");
        FUNCTION("COMPARE_EQ_FP32_035") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_035");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_036 | 4D 无广播，n+c+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_036)
{
    PROGRAM("COMPARE_EQ_FP32_036")
    {
        TileShape::Current().SetVecTile({1, 2, 32, 32});
        Tensor input(DT_FP32, {2, 4, 32, 64}, "input");
        Tensor other(DT_FP32, {2, 4, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {2, 4, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_036") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_036");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_037 | 4D 无广播，n+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_037)
{
    PROGRAM("COMPARE_EQ_FP32_037")
    {
        TileShape::Current().SetVecTile({2, 2, 16, 32});
        Tensor input(DT_FP32, {4, 2, 32, 64}, "input");
        Tensor other(DT_FP32, {4, 2, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {4, 2, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_037") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_037");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_038 | 4D 无广播，c+h+w 切分
TEST_F(LiteNPUCodeGenCompare, test_compare_eq_fp32_038)
{
    PROGRAM("COMPARE_EQ_FP32_038")
    {
        TileShape::Current().SetVecTile({1, 2, 16, 32});
        Tensor input(DT_FP32, {1, 4, 32, 64}, "input");
        Tensor other(DT_FP32, {1, 4, 32, 64}, "other");
        auto output = Tensor(DT_BOOL, {1, 4, 32, 64}, "output");
        FUNCTION("COMPARE_EQ_FP32_038") { output = Compare(input, other, OpType::EQ, OutType::BOOL); }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "COMPARE_EQ_FP32_038");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}
