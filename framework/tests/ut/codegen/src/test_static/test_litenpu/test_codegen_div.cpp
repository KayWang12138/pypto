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
 * \file test_codegen_div.cpp
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

class LiteNPUCodeGenDiv : public testing::Test {
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

// ======================== 1D 用例 ========================

TEST_F(LiteNPUCodeGenDiv, test_div_001) {
    PROGRAM("DIV_001") {
        Tensor input0(DataType::DT_FP32, {160}, "input0");
        Tensor input1(DataType::DT_FP32, {160}, "input1");
        auto output = Tensor(DataType::DT_FP32, {160}, "output");
        FUNCTION("DIV_001") {
            TileShape::Current().SetVecTile({200});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_002) {
    PROGRAM("DIV_002") {
        Tensor input0(DataType::DT_FP16, {100}, "input0");
        Tensor input1(DataType::DT_FP16, {100}, "input1");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("DIV_002") {
            TileShape::Current().SetVecTile({100});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_003) {
    PROGRAM("DIV_003") {
        Tensor input0(DataType::DT_FP32, {112}, "input0");
        Tensor input1(DataType::DT_FP32, {112}, "input1");
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("DIV_003") {
            TileShape::Current().SetVecTile({100});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_004) {
    PROGRAM("DIV_004") {
        Tensor input0(DataType::DT_FP16, {101}, "input0");
        Tensor input1(DataType::DT_FP16, {101}, "input1");
        auto output = Tensor(DataType::DT_FP16, {101}, "output");
        FUNCTION("DIV_004") {
            TileShape::Current().SetVecTile({100});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_005) {
    PROGRAM("DIV_005") {
        Tensor input0(DataType::DT_FP32, {112}, "input0");
        Element input1(DataType::DT_FP32, 2.0f);
        auto output = Tensor(DataType::DT_FP32, {112}, "output");
        FUNCTION("DIV_005") {
            TileShape::Current().SetVecTile({100});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_006) {
    PROGRAM("DIV_006") {
        Tensor input0(DataType::DT_FP16, {101}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {101}, "output");
        FUNCTION("DIV_006") {
            TileShape::Current().SetVecTile({100});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_007) {
    PROGRAM("DIV_007") {
        Tensor input0(DataType::DT_FP32, {160}, "input0");
        Tensor input1(DataType::DT_FP32, {1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {160}, "output");
        FUNCTION("DIV_007") {
            TileShape::Current().SetVecTile({120});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_008) {
    PROGRAM("DIV_008") {
        Tensor input0(DataType::DT_FP16, {100}, "input0");
        Tensor input1(DataType::DT_FP16, {1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {100}, "output");
        FUNCTION("DIV_008") {
            TileShape::Current().SetVecTile({112});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// ======================== 2D 用例 ========================

TEST_F(LiteNPUCodeGenDiv, test_div_009) {
    PROGRAM("DIV_009") {
        Tensor input0(DataType::DT_FP32, {32, 20}, "input0");
        Element input1(DataType::DT_FP32, 2.0f);
        auto output = Tensor(DataType::DT_FP32, {32, 20}, "output");
        FUNCTION("DIV_009") {
            TileShape::Current().SetVecTile({64, 32});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_010) {
    PROGRAM("DIV_010") {
        Tensor input0(DataType::DT_FP16, {31, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("DIV_010") {
            TileShape::Current().SetVecTile({32, 10});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_011) {
    PROGRAM("DIV_011") {
        Tensor input0(DataType::DT_FP32, {31, 19}, "input0");
        Tensor input1(DataType::DT_FP32, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP32, {31, 19}, "output");
        FUNCTION("DIV_011") {
            TileShape::Current().SetVecTile({10, 30});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_012) {
    PROGRAM("DIV_012") {
        Tensor input0(DataType::DT_FP16, {31, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 19}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("DIV_012") {
            TileShape::Current().SetVecTile({10, 14});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_013) {
    PROGRAM("DIV_013") {
        Tensor input0(DataType::DT_FP32, {32, 20}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 20}, "input1");
        auto output = Tensor(DataType::DT_FP32, {32, 20}, "output");
        FUNCTION("DIV_013") {
            TileShape::Current().SetVecTile({64, 32});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_014) {
    PROGRAM("DIV_014") {
        Tensor input0(DataType::DT_FP16, {31, 21}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 21}, "output");
        FUNCTION("DIV_014") {
            TileShape::Current().SetVecTile({32, 10});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_015) {
    PROGRAM("DIV_015") {
        Tensor input0(DataType::DT_FP32, {31, 19}, "input0");
        Tensor input1(DataType::DT_FP32, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {31, 19}, "output");
        FUNCTION("DIV_015") {
            TileShape::Current().SetVecTile({10, 20});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_016) {
    PROGRAM("DIV_016") {
        Tensor input0(DataType::DT_FP16, {1, 19}, "input0");
        Tensor input1(DataType::DT_FP16, {31, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {31, 19}, "output");
        FUNCTION("DIV_016") {
            TileShape::Current().SetVecTile({10, 16});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// ======================== 3D 用例 ========================

TEST_F(LiteNPUCodeGenDiv, test_div_017) {
    PROGRAM("DIV_017") {
        Tensor input0(DataType::DT_FP32, {10, 32, 23}, "input0");
        Element input1(DataType::DT_FP32, 2.0f);
        auto output = Tensor(DataType::DT_FP32, {10, 32, 23}, "output");
        FUNCTION("DIV_017") {
            TileShape::Current().SetVecTile({5, 32, 30});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_018) {
    PROGRAM("DIV_018") {
        Tensor input0(DataType::DT_FP16, {10, 32, 23}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {10, 32, 23}, "output");
        FUNCTION("DIV_018") {
            TileShape::Current().SetVecTile({7, 32, 22});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_019) {
    PROGRAM("DIV_019") {
        Tensor input0(DataType::DT_FP32, {21, 19, 33}, "input0");
        Tensor input1(DataType::DT_FP32, {21, 19, 33}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 19, 33}, "output");
        FUNCTION("DIV_019") {
            TileShape::Current().SetVecTile({5, 12, 40});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_020) {
    PROGRAM("DIV_020") {
        Tensor input0(DataType::DT_FP16, {10, 32, 23}, "input0");
        Tensor input1(DataType::DT_FP16, {10, 32, 23}, "input1");
        auto output = Tensor(DataType::DT_FP16, {10, 32, 23}, "output");
        FUNCTION("DIV_020") {
            TileShape::Current().SetVecTile({5, 64, 22});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_021) {
    PROGRAM("DIV_021") {
        Tensor input0(DataType::DT_FP32, {63, 43, 27}, "input0");
        Tensor input1(DataType::DT_FP32, {63, 1, 27}, "input1");
        auto output = Tensor(DataType::DT_FP32, {63, 43, 27}, "output");
        FUNCTION("DIV_021") {
            TileShape::Current().SetVecTile({23, 20, 17});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_022) {
    PROGRAM("DIV_022") {
        Tensor input0(DataType::DT_FP16, {63, 1, 27}, "input0");
        Tensor input1(DataType::DT_FP16, {63, 43, 27}, "input1");
        auto output = Tensor(DataType::DT_FP16, {63, 43, 27}, "output");
        FUNCTION("DIV_022") {
            TileShape::Current().SetVecTile({23, 20, 20});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_023) {
    PROGRAM("DIV_023") {
        Tensor input0(DataType::DT_FP32, {63, 43, 27}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 1, 27}, "input1");
        auto output = Tensor(DataType::DT_FP32, {63, 43, 27}, "output");
        FUNCTION("DIV_023") {
            TileShape::Current().SetVecTile({23, 20, 17});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_024) {
    PROGRAM("DIV_024") {
        Tensor input0(DataType::DT_FP16, {63, 1, 27}, "input0");
        Tensor input1(DataType::DT_FP16, {1, 43, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {63, 43, 27}, "output");
        FUNCTION("DIV_024") {
            TileShape::Current().SetVecTile({23, 20, 20});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// ======================== 4D 用例 ========================

TEST_F(LiteNPUCodeGenDiv, test_div_025) {
    PROGRAM("DIV_025") {
        Tensor input0(DataType::DT_FP32, {5, 16, 11, 12}, "input0");
        Element input1(DataType::DT_FP32, 2.0f);
        auto output = Tensor(DataType::DT_FP32, {5, 16, 11, 12}, "output");
        FUNCTION("DIV_025") {
            TileShape::Current().SetVecTile({5, 20, 15, 12});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_026) {
    PROGRAM("DIV_026") {
        Tensor input0(DataType::DT_FP16, {10, 15, 16, 10}, "input0");
        Element input1(DataType::DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {10, 15, 16, 10}, "output");
        FUNCTION("DIV_026") {
            TileShape::Current().SetVecTile({7, 16, 20, 10});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_027) {
    PROGRAM("DIV_027") {
        Tensor input0(DataType::DT_FP32, {21, 12, 15, 16}, "input0");
        Tensor input1(DataType::DT_FP32, {21, 12, 15, 16}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 12, 15, 16}, "output");
        FUNCTION("DIV_027") {
            TileShape::Current().SetVecTile({11, 12, 16, 6});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_028) {
    PROGRAM("DIV_028") {
        Tensor input0(DataType::DT_FP16, {21, 19, 33, 12}, "input0");
        Tensor input1(DataType::DT_FP16, {21, 19, 33, 12}, "input1");
        auto output = Tensor(DataType::DT_FP16, {21, 19, 33, 12}, "output");
        FUNCTION("DIV_028") {
            TileShape::Current().SetVecTile({10, 20, 5, 12});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_029) {
    PROGRAM("DIV_029") {
        Tensor input0(DataType::DT_FP32, {21, 11, 13, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {1, 11, 13, 17}, "input1");
        auto output = Tensor(DataType::DT_FP32, {21, 11, 13, 17}, "output");
        FUNCTION("DIV_029") {
            TileShape::Current().SetVecTile({15, 12, 11, 6});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_029");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_030) {
    PROGRAM("DIV_030") {
        Tensor input0(DataType::DT_FP16, {25, 11, 15, 1}, "input0");
        Tensor input1(DataType::DT_FP16, {25, 11, 15, 17}, "input1");
        auto output = Tensor(DataType::DT_FP16, {25, 11, 15, 17}, "output");
        FUNCTION("DIV_030") {
            TileShape::Current().SetVecTile({23, 5, 17, 10});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_030");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_031) {
    PROGRAM("DIV_031") {
        Tensor input0(DataType::DT_FP32, {13, 16, 1, 17}, "input0");
        Tensor input1(DataType::DT_FP32, {13, 16, 17, 1}, "input1");
        auto output = Tensor(DataType::DT_FP32, {13, 16, 17, 17}, "output");
        FUNCTION("DIV_031") {
            TileShape::Current().SetVecTile({10, 15, 5, 17});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_031");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

TEST_F(LiteNPUCodeGenDiv, test_div_032) {
    PROGRAM("DIV_032") {
        Tensor input0(DataType::DT_FP16, {15, 1, 1, 13}, "input0");
        Tensor input1(DataType::DT_FP16, {1, 35, 27, 1}, "input1");
        auto output = Tensor(DataType::DT_FP16, {15, 35, 27, 13}, "output");
        FUNCTION("DIV_032") {
            TileShape::Current().SetVecTile({13, 12, 11, 10});
            output = Div(input0, input1);
        }
    }
    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "DIV_032");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}