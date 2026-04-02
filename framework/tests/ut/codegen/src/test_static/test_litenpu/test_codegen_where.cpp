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
 * \file test_codegen_where.cpp
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

class LiteNPUCodeGenWhere : public testing::Test {
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

TEST_F(LiteNPUCodeGenWhere, test_where_tt) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(8, 8);
        Tensor condition(DT_UINT8, {8, 2}, "condition");
        Tensor input(DT_FP32, {8, 16}, "input");
        Tensor other(DT_FP32, {8, 16}, "other");
        Tensor output;
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenWhere, test_where_ts) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(8, 8);
        Tensor condition(DT_UINT8, {8, 2}, "condition");
        Tensor input(DT_FP32, {8, 16}, "input");
        Element other(DT_FP32, 1.0);
        Tensor output;
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenWhere, test_where_ss) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(8, 8);
        Tensor condition(DT_UINT8, {8, 2}, "condition");
        Element input(DT_FP32, 8.0);
        Element other(DT_FP32, 1.0);
        Tensor output;
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenWhere, test_where_st) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(8, 8);
        Tensor condition(DT_UINT8, {8, 2}, "condition");
        Element input(DT_FP32, 1.0);
        Tensor other(DT_FP32, {8, 16}, "input");
        Tensor output;
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// 不能正确生成.cpp文件 说明不支持一维输入
TEST_F(LiteNPUCodeGenWhere, test_where_1d) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(8);
        Tensor condition(DT_UINT8, {8}, "condition");
        Tensor input(DT_FP32, {8}, "input");
        Tensor other(DT_FP32, {8}, "input");
        Tensor output;
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenWhere, test_where_tt_fp16_001) {
    PROGRAM("WHERE_fp16_001") {
        TileShape::Current().SetVecTile(1, 56);
        Tensor condition(DT_UINT8, {1, 14}, "condition");
        Tensor input(DT_FP16, {1, 112}, "input");
        Tensor other(DT_FP16, {1, 112}, "other");
        Tensor output;
        FUNCTION("WHERE_fp16_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_fp16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenWhere, test_where_tt_fp16_002) {
    PROGRAM("WHERE_fp16_002") {
        TileShape::Current().SetVecTile(8, 8);
        Tensor condition(DT_UINT8, {1, 8}, "condition");
        Tensor input(DT_FP16, {1, 64}, "input");
        Tensor other(DT_FP16, {1, 64}, "other");
        Tensor output;
        FUNCTION("WHERE_fp16_002") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_fp16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenWhere, test_where_tt_fp16_003) {
    PROGRAM("WHERE_fp16_003") {
        TileShape::Current().SetVecTile(32, 32);
        Tensor condition(DT_UINT8, {1, 16}, "condition");
        Tensor input(DT_FP16, {1, 128}, "input");
        Tensor other(DT_FP16, {1, 128}, "other");
        Tensor output;
        FUNCTION("WHERE_fp16_003") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_fp16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenWhere, test_where_tt_fp16_004) {
    PROGRAM("WHERE_fp16_004") {
        TileShape::Current().SetVecTile(8, 64);
        Tensor condition(DT_UINT8, {2, 12}, "condition");
        Tensor input(DT_FP16, {2, 96}, "input");
        Tensor other(DT_FP16, {1, 96}, "other");
        Tensor output;
        FUNCTION("WHERE_fp16_004") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_fp16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenWhere, test_where_tt_fp16_005) {
    PROGRAM("WHERE_fp16_005") {
        TileShape::Current().SetVecTile(8, 16);
        Tensor condition(DT_UINT8, {5, 4}, "condition");
        Tensor input(DT_FP16, {5, 32}, "input");
        Tensor other(DT_FP16, {5, 32}, "other");
        Tensor output;
        FUNCTION("WHERE_fp16_005") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_fp16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}