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
#include "codegen/npu/litenpu/codegen_litenpu.h"

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

// input和other均为张量
TEST_F(LiteNPUCodeGenWhere, test_where_tt) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(8, 8);
        Tensor condition(DT_UINT8, {8, 2}, "condition");
        Tensor input(DT_FP32, {8, 16}, "input");
        Tensor other(DT_FP32, {8, 16}, "other");
        auto output = Tensor(DataType::DT_FP32, {8, 16}, "output");
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// input为张量 other为标量
TEST_F(LiteNPUCodeGenWhere, test_where_ts) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(1, 1, 40, 5);
        Tensor condition(DT_UINT8, {1, 2, 5, 5}, "condition");
        Tensor input(DT_FP32, {1, 2, 40, 5}, "input");
        Element other(DT_FP32, 1.0);
        auto output = Tensor(DataType::DT_FP32, {1, 2, 40, 5}, "output");
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// input、other均为张量
TEST_F(LiteNPUCodeGenWhere, test_where_ss) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(8, 2, 8, 8);
        Tensor condition(DT_UINT8, {8, 2, 8, 8}, "condition");
        Element input(DT_FP32, 8.0);
        Element other(DT_FP32, 1.0);
        auto output = Tensor(DataType::DT_FP32, {8, 2, 8, 8}, "output");
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// input为标量 other为张量 
TEST_F(LiteNPUCodeGenWhere, test_where_st) {
    PROGRAM("WHERE_001") {
        TileShape::Current().SetVecTile(8, 8);
        Tensor condition(DT_UINT8, {8, 2}, "condition");
        Element input(DT_FP32, 1.0);
        Tensor other(DT_FP32, {8, 16}, "input");
        auto output = Tensor(DataType::DT_FP32, {8, 16}, "output");
        FUNCTION("WHERE_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_001 | 2D 双张量，N单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_001) {
    PROGRAM("WHERE_FP16_001") {
        TileShape::Current().SetVecTile(1, 64);
        Tensor condition(DT_UINT8, {2, 8}, "condition");
        Tensor input(DT_FP16, {2, 64}, "input");
        Tensor other(DT_FP16, {2, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 64}, "output");
        FUNCTION("WHERE_FP16_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_002 | 2D 双张量，W单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_002) {
    PROGRAM("WHERE_FP16_002") {
        TileShape::Current().SetVecTile(4, 16);
        Tensor condition(DT_UINT8, {4, 4}, "condition");
        Tensor input(DT_FP16, {4, 32}, "input");
        Tensor other(DT_FP16, {4, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {4, 32}, "output");
        FUNCTION("WHERE_FP16_002") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_003 | 2D 广播，W单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_003) {
    PROGRAM("WHERE_FP16_003") {
        TileShape::Current().SetVecTile(2, 32);
        Tensor condition(DT_UINT8, {2, 8}, "condition");
        Tensor input(DT_FP16, {2, 64}, "input");
        Tensor other(DT_FP16, {1, 64}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 64}, "output");
        FUNCTION("WHERE_FP16_003") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_004 | 2D Tensor+标量，N单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_004) {
    PROGRAM("WHERE_FP16_004") {
        TileShape::Current().SetVecTile(2, 32);
        Tensor condition(DT_UINT8, {4, 4}, "condition");
        Tensor input(DT_FP16, {4, 32}, "input");
        Element other(DT_FP16, 1.0);
        auto output = Tensor(DataType::DT_FP16, {4, 32}, "output");
        FUNCTION("WHERE_FP16_004") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_005 | 2D 双张量，NC双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_005) {
    PROGRAM("WHERE_FP16_005") {
        TileShape::Current().SetVecTile(1, 40);
        Tensor condition(DT_UINT8, {2, 5}, "condition");
        Tensor input(DT_FP16, {2, 40}, "input");
        Tensor other(DT_FP16, {2, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 40}, "output");
        FUNCTION("WHERE_FP16_005") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_006 | 3D 双张量，H单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_006) {
    PROGRAM("WHERE_FP16_006") {
        TileShape::Current().SetVecTile(2, 16, 32);
        Tensor condition(DT_UINT8, {2, 1, 4}, "condition");
        Tensor input(DT_FP16, {2, 32, 32}, "input");
        Tensor other(DT_FP16, {2, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP16_006") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_007 | 3D 双张量，W单轴切分 
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_007) {
    PROGRAM("WHERE_FP16_007") {
        TileShape::Current().SetVecTile(2, 32, 32);
        /* TileShape::Current().SetVecTile(2, 32, 16);  出现了跟compare低维向高维广播相同的报错 */
        Tensor condition(DT_UINT8, {2, 1, 4}, "condition");
        Tensor input(DT_FP16, {2, 32, 32}, "input");
        Tensor other(DT_FP16, {2, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP16_007") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_008 | 3D 双张量，N单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_008) {
    PROGRAM("WHERE_FP16_008") {
        TileShape::Current().SetVecTile(1, 24, 24);
        Tensor condition(DT_UINT8, {2, 1, 3}, "condition");
        Tensor input(DT_FP16, {2, 24, 24}, "input");
        Tensor other(DT_FP16, {2, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 24, 24}, "output");
        FUNCTION("WHERE_FP16_008") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_009 | 3D Tensor+标量，H单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_009) {
    PROGRAM("WHERE_FP16_009") {
        TileShape::Current().SetVecTile(3, 8, 48);
        Tensor condition(DT_UINT8, {3, 1, 6}, "condition");
        Tensor input(DT_FP16, {3, 16, 48}, "input");
        Element other(DT_FP16, 1.0);
        auto output = Tensor(DataType::DT_FP16, {3, 16, 48}, "output");
        FUNCTION("WHERE_FP16_009") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_010 | 3D 双张量，HC双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_010) {
    PROGRAM("WHERE_FP16_010") {
        TileShape::Current().SetVecTile(2, 16, 32);
        Tensor condition(DT_UINT8, {2, 1, 5}, "condition");
        Tensor input(DT_FP16, {2, 32, 40}, "input");
        Tensor other(DT_FP16, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP16_010") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_011 | 3D 双张量，WC双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_011) {
    PROGRAM("WHERE_FP16_011") {
        /*
        TileShape::Current().SetVecTile(2, 32, 16); 
        Tensor condition(DT_UINT8, {2, 1, 5}, "condition");
        出现了跟compare低维向高维广播相同的报错
        */
        TileShape::Current().SetVecTile(2, 16, 32);
        Tensor condition(DT_UINT8, {2, 32, 5}, "condition");
        Tensor input(DT_FP16, {2, 32, 40}, "input");
        Tensor other(DT_FP16, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP16_011") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_012 | 4D 双张量，H单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_012) {
    PROGRAM("WHERE_FP16_012") {
        TileShape::Current().SetVecTile(1, 2, 20, 40);
        Tensor condition(DT_UINT8, {1, 2, 1, 5}, "condition");
        Tensor input(DT_FP16, {1, 2, 40, 40}, "input");
        Tensor other(DT_FP16, {1, 2, 40, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {1, 2, 40, 40}, "output");
        FUNCTION("WHERE_FP16_012") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_013 | 4D 双张量，W单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_013) {
    PROGRAM("WHERE_FP16_013") {
        TileShape::Current().SetVecTile(1, 2, 40, 16);
        // Tensor condition(DT_UINT8, {1, 2, 1, 5}, "condition"); 会出现跟compare同样的报错 只支持单轴广播？ 
        Tensor condition(DT_UINT8, {1, 2, 40, 5}, "condition");
        Tensor input(DT_FP16, {1, 2, 40, 40}, "input");
        Tensor other(DT_FP16, {1, 2, 40, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {1, 2, 40, 40}, "output");
        FUNCTION("WHERE_FP16_013") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_014 | 4D 双张量，N单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_014) {
    PROGRAM("WHERE_FP16_014") {
        TileShape::Current().SetVecTile(1, 3, 24, 24);
        Tensor condition(DT_UINT8, {2, 3, 1, 3}, "condition");
        Tensor input(DT_FP16, {2, 3, 24, 24}, "input");
        Tensor other(DT_FP16, {2, 3, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 3, 24, 24}, "output");
        FUNCTION("WHERE_FP16_014") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_015 | 4D 广播，C单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_015) {
    PROGRAM("WHERE_FP16_015") {
        TileShape::Current().SetVecTile(1, 2, 32, 32);
        Tensor condition(DT_UINT8, {1, 1, 32, 4}, "condition");
        Tensor input(DT_FP16, {1, 4, 32, 32}, "input");
        Tensor other(DT_FP16, {1, 1, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP16, {1, 4, 32, 32}, "output");
        FUNCTION("WHERE_FP16_015") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_016 | 4D 双张量，H+W双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_016) {
    PROGRAM("WHERE_FP16_016") {
        /* 
        TileShape::Current().SetVecTile(1, 2, 12, 16); 
        Tensor condition(DT_UINT8, {2, 2, 1, 5}, "condition");
        出现与compare相同报错
        */
        // TileShape::Current().SetVecTile(1, 2, 16, 16);
        TileShape::Current().SetVecTile(1, 2, 16, 16);
        Tensor condition(DT_UINT8, {2, 2, 24, 5}, "condition");
        Tensor input(DT_FP16, {2, 2, 24, 40}, "input");
        Tensor other(DT_FP16, {2, 2, 24, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 2, 24, 40}, "output");
        FUNCTION("WHERE_FP16_016") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_017 | 3D 双张量，N+H双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_017) {
    PROGRAM("WHERE_FP16_017") {
        TileShape::Current().SetVecTile(1, 16, 24);
        // Tensor condition(DT_UINT8, {2, 1, 5}, "condition"); 出现与compare相同报错
        Tensor condition(DT_UINT8, {2, 32, 5}, "condition");
        Tensor input(DT_FP16, {2, 32, 40}, "input");
        Tensor other(DT_FP16, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP16_017") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_018 | 3D 双张量，H+W双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_018) {
    PROGRAM("WHERE_FP16_018") {
        TileShape::Current().SetVecTile(2, 16, 24);
        Tensor condition(DT_UINT8, {2, 32, 5}, "condition");
        Tensor input(DT_FP16, {2, 32, 40}, "input");
        Tensor other(DT_FP16, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP16_018") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_019 | 4D 双张量，N+H+W三轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_019) {
    PROGRAM("WHERE_FP16_019") {
        TileShape::Current().SetVecTile(1, 4, 12, 8);
        Tensor condition(DT_UINT8, {2, 4, 24, 3}, "condition");
        Tensor input(DT_FP16, {2, 4, 24, 24}, "input");
        Tensor other(DT_FP16, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP16_019") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_020 | 4D 双张量，C+H+W三轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_020) {
    PROGRAM("WHERE_FP16_020") {
        TileShape::Current().SetVecTile(2, 2, 12, 8);
        Tensor condition(DT_UINT8, {2, 4, 24, 3}, "condition");
        Tensor input(DT_FP16, {2, 4, 24, 24}, "input");
        Tensor other(DT_FP16, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP16_020") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_021 | 4D 双张量，N+C+H三轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_021) {
    PROGRAM("WHERE_FP16_021") {
        TileShape::Current().SetVecTile(1, 2, 12, 24);
        Tensor condition(DT_UINT8, {2, 4, 24, 3}, "condition");
        Tensor input(DT_FP16, {2, 4, 24, 24}, "input");
        Tensor other(DT_FP16, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP16_021") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_022 | 4D 双张量，N+C+H+W四轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_022) {
    PROGRAM("WHERE_FP16_022") {
        TileShape::Current().SetVecTile(1, 2, 12, 8);
        Tensor condition(DT_UINT8, {2, 4, 24, 3}, "condition");
        Tensor input(DT_FP16, {2, 4, 24, 24}, "input");
        Tensor other(DT_FP16, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP16_022") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_023 | 4D input+标量，N+H+W切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_023) {
    PROGRAM("WHERE_FP16_023") {
        TileShape::Current().SetVecTile(1, 4, 16, 16);
        Tensor condition(DT_UINT8, {2, 4, 32, 5}, "condition");
        Tensor input(DT_FP16, {2, 4, 32, 40}, "input");
        Element other(DT_FP16, 1.0);
        auto output = Tensor(DataType::DT_FP16, {2, 4, 32, 40}, "output");
        FUNCTION("WHERE_FP16_023") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_024 | 4D input+标量，C+H+W切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_024) {
    PROGRAM("WHERE_FP16_024") {
        TileShape::Current().SetVecTile(2, 2, 20, 16);
        Tensor condition(DT_UINT8, {2, 4, 40, 5}, "condition");
        Tensor input(DT_FP16, {2, 4, 40, 40}, "input");
        Element other(DT_FP16, 1.0);
        auto output = Tensor(DataType::DT_FP16, {2, 4, 40, 40}, "output");
        FUNCTION("WHERE_FP16_024") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_025 | 4D 双张量，NCW三轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_025) {
    PROGRAM("WHERE_FP16_025") {
        TileShape::Current().SetVecTile(1, 1, 16, 16);
        Tensor condition(DT_UINT8, {2, 3, 32, 5}, "condition");
        Tensor input(DT_FP16, {2, 3, 32, 40}, "input");
        Tensor other(DT_FP16, {2, 3, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP16, {2, 3, 32, 40}, "output");
        FUNCTION("WHERE_FP16_025") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_026 | 2D 双标量，N+W占位切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_026) {
    PROGRAM("WHERE_FP16_026") {
        TileShape::Current().SetVecTile(1, 8);
        Tensor condition(DT_UINT8, {2, 8}, "condition");
        Element input(DT_FP16, 1.0);
        Element other(DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {2, 8}, "output");
        FUNCTION("WHERE_FP16_026") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_027 | 3D 双标量，N+H+W占位切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_027) {
    PROGRAM("WHERE_FP16_027") {
        TileShape::Current().SetVecTile(1, 4, 8);
        Tensor condition(DT_UINT8, {1, 1, 8}, "condition");
        Element input(DT_FP16, 1.0);
        Element other(DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {1, 8, 8}, "output");
        FUNCTION("WHERE_FP16_027") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp16_028 | 4D 双标量，N+C+H+W占位切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp16_028) {
    PROGRAM("WHERE_FP16_028") {
        TileShape::Current().SetVecTile(1, 1, 2, 8);
        Tensor condition(DT_UINT8, {1, 1, 1, 8}, "condition");
        Element input(DT_FP16, 1.0);
        Element other(DT_FP16, 2.0);
        auto output = Tensor(DataType::DT_FP16, {1, 1, 1, 8}, "output");
        FUNCTION("WHERE_FP16_028") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP16_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_001 | 2D 双张量，N单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_001) {
    PROGRAM("WHERE_FP32_001") {
        TileShape::Current().SetVecTile(1, 64);
        Tensor condition(DT_UINT8, {2, 8}, "condition");
        Tensor input(DT_FP32, {2, 64}, "input");
        Tensor other(DT_FP32, {2, 64}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 64}, "output");
        FUNCTION("WHERE_FP32_001") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_001");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_002 | 2D 双张量，W单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_002) {
    PROGRAM("WHERE_FP32_002") {
        TileShape::Current().SetVecTile(4, 16);
        Tensor condition(DT_UINT8, {4, 4}, "condition");
        Tensor input(DT_FP32, {4, 32}, "input");
        Tensor other(DT_FP32, {4, 32}, "other");
        auto output = Tensor(DataType::DT_FP32, {4, 32}, "output");
        FUNCTION("WHERE_FP32_002") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_002");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_003 | 2D 广播，W单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_003) {
    PROGRAM("WHERE_FP32_003") {
        TileShape::Current().SetVecTile(2, 32);
        Tensor condition(DT_UINT8, {2, 8}, "condition");
        Tensor input(DT_FP32, {2, 64}, "input");
        Tensor other(DT_FP32, {1, 64}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 64}, "output");
        FUNCTION("WHERE_FP32_003") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_003");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_004 | 2D Tensor+标量，N单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_004) {
    PROGRAM("WHERE_FP32_004") {
        TileShape::Current().SetVecTile(2, 32);
        Tensor condition(DT_UINT8, {4, 4}, "condition");
        Tensor input(DT_FP32, {4, 32}, "input");
        Element other(DT_FP32, 1.0);
        auto output = Tensor(DataType::DT_FP32, {4, 32}, "output");
        FUNCTION("WHERE_FP32_004") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_004");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_005 | 2D 双张量，NC双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_005) {
    PROGRAM("WHERE_FP32_005") {
        TileShape::Current().SetVecTile(1, 40);
        Tensor condition(DT_UINT8, {2, 5}, "condition");
        Tensor input(DT_FP32, {2, 40}, "input");
        Tensor other(DT_FP32, {2, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 40}, "output");
        FUNCTION("WHERE_FP32_005") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_005");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_006 | 3D 双张量，H单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_006) {
    PROGRAM("WHERE_FP32_006") {
        TileShape::Current().SetVecTile(2, 16, 32);
        Tensor condition(DT_UINT8, {2, 32, 4}, "condition");
        Tensor input(DT_FP32, {2, 32, 32}, "input");
        Tensor other(DT_FP32, {2, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP32_006") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_006");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_007 | 3D 双张量，W单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_007) {
    PROGRAM("WHERE_FP32_007") {
        TileShape::Current().SetVecTile(2, 32, 16);
        Tensor condition(DT_UINT8, {2, 32, 4}, "condition");
        Tensor input(DT_FP32, {2, 32, 32}, "input");
        Tensor other(DT_FP32, {2, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 32}, "output");
        FUNCTION("WHERE_FP32_007") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_007");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_008 | 3D 双张量，N单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_008) {
    PROGRAM("WHERE_FP32_008") {
        TileShape::Current().SetVecTile(1, 24, 24);
        Tensor condition(DT_UINT8, {2, 24, 3}, "condition");
        Tensor input(DT_FP32, {2, 24, 24}, "input");
        Tensor other(DT_FP32, {2, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 24, 24}, "output");
        FUNCTION("WHERE_FP32_008") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_008");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_009 | 3D Tensor+标量，H单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_009) {
    PROGRAM("WHERE_FP32_009") {
        TileShape::Current().SetVecTile(3, 8, 48);
        Tensor condition(DT_UINT8, {3, 16, 6}, "condition");
        Tensor input(DT_FP32, {3, 16, 48}, "input");
        Element other(DT_FP32, 1.0);
        auto output = Tensor(DataType::DT_FP32, {3, 16, 48}, "output");
        FUNCTION("WHERE_FP32_009") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_009");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_010 | 3D 双张量，HC双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_010) {
    PROGRAM("WHERE_FP32_010") {
        TileShape::Current().SetVecTile(2, 16, 40);
        Tensor condition(DT_UINT8, {2, 32, 5}, "condition");
        Tensor input(DT_FP32, {2, 32, 40}, "input");
        Tensor other(DT_FP32, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP32_010") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_010");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_011 | 3D 双张量，WC双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_011) {
    PROGRAM("WHERE_FP32_011") {
        TileShape::Current().SetVecTile(2, 32, 16);
        Tensor condition(DT_UINT8, {2, 32, 5}, "condition");
        Tensor input(DT_FP32, {2, 32, 40}, "input");
        Tensor other(DT_FP32, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP32_011") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_011");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_012 | 4D 双张量，H单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_012) {
    PROGRAM("WHERE_FP32_012") {
        TileShape::Current().SetVecTile(1, 2, 20, 40);
        Tensor condition(DT_UINT8, {1, 2, 40, 5}, "condition");
        Tensor input(DT_FP32, {1, 2, 40, 40}, "input");
        Tensor other(DT_FP32, {1, 2, 40, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {1, 2, 40, 40}, "output");
        FUNCTION("WHERE_FP32_012") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_012");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_013 | 4D 双张量，W单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_013) {
    PROGRAM("WHERE_FP32_013") {
        TileShape::Current().SetVecTile(1, 2, 40, 16);
        Tensor condition(DT_UINT8, {1, 2, 40, 5}, "condition");
        Tensor input(DT_FP32, {1, 2, 40, 40}, "input");
        Tensor other(DT_FP32, {1, 2, 40, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {1, 2, 40, 40}, "output");
        FUNCTION("WHERE_FP32_013") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_013");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_014 | 4D 双张量，N单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_014) {
    PROGRAM("WHERE_FP32_014") {
        TileShape::Current().SetVecTile(1, 3, 24, 24);
        Tensor condition(DT_UINT8, {2, 3, 24, 3}, "condition");
        Tensor input(DT_FP32, {2, 3, 24, 24}, "input");
        Tensor other(DT_FP32, {2, 3, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 3, 24, 24}, "output");
        FUNCTION("WHERE_FP32_014") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_014");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_015 | 4D 广播，C单轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_015) {
    PROGRAM("WHERE_FP32_015") {
        TileShape::Current().SetVecTile(1, 2, 32, 32);
        Tensor condition(DT_UINT8, {1, 4, 32, 4}, "condition");
        Tensor input(DT_FP32, {1, 4, 32, 32}, "input");
        Tensor other(DT_FP32, {1, 1, 32, 32}, "other");
        auto output = Tensor(DataType::DT_FP32, {1, 4, 32, 32}, "output");
        FUNCTION("WHERE_FP32_015") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_015");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_016 | 4D 双张量，H+W双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_016) {
    PROGRAM("WHERE_FP32_016") {
        TileShape::Current().SetVecTile(1, 2, 12, 16);
        Tensor condition(DT_UINT8, {2, 2, 24, 5}, "condition");
        Tensor input(DT_FP32, {2, 2, 24, 40}, "input");
        Tensor other(DT_FP32, {2, 2, 24, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 2, 24, 40}, "output");
        FUNCTION("WHERE_FP32_016") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_016");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_017 | 3D 双张量，N+H双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_017) {
    PROGRAM("WHERE_FP32_017") {
        TileShape::Current().SetVecTile(1, 16, 16);
        Tensor condition(DT_UINT8, {2, 32, 5}, "condition");
        Tensor input(DT_FP32, {2, 32, 40}, "input");
        Tensor other(DT_FP32, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP32_017") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_017");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_018 | 3D 双张量，H+W双轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_018) {
    PROGRAM("WHERE_FP32_018") {
        TileShape::Current().SetVecTile(2, 16, 16);
        Tensor condition(DT_UINT8, {2, 32, 5}, "condition");
        Tensor input(DT_FP32, {2, 32, 40}, "input");
        Tensor other(DT_FP32, {2, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 32, 40}, "output");
        FUNCTION("WHERE_FP32_018") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_018");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_019 | 4D 双张量，N+H+W三轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_019) {
    PROGRAM("WHERE_FP32_019") {
        TileShape::Current().SetVecTile(1, 4, 12, 8);
        Tensor condition(DT_UINT8, {2, 4, 24, 3}, "condition");
        Tensor input(DT_FP32, {2, 4, 24, 24}, "input");
        Tensor other(DT_FP32, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP32_019") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_019");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_020 | 4D 双张量，C+H+W三轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_020) {
    PROGRAM("WHERE_FP32_020") {
        TileShape::Current().SetVecTile(2, 2, 12, 8);
        Tensor condition(DT_UINT8, {2, 4, 24, 3}, "condition");
        Tensor input(DT_FP32, {2, 4, 24, 24}, "input");
        Tensor other(DT_FP32, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP32_020") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_020");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_021 | 4D 双张量，N+C+H三轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_021) {
    PROGRAM("WHERE_FP32_021") {
        TileShape::Current().SetVecTile(1, 2, 12, 24);
        Tensor condition(DT_UINT8, {2, 4, 24, 3}, "condition");
        Tensor input(DT_FP32, {2, 4, 24, 24}, "input");
        Tensor other(DT_FP32, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP32_021") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_021");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_022 | 4D 双张量，N+C+H+W四轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_022) {
    PROGRAM("WHERE_FP32_022") {
        TileShape::Current().SetVecTile(1, 2, 12, 16);
        Tensor condition(DT_UINT8, {2, 4, 24, 3}, "condition");
        Tensor input(DT_FP32, {2, 4, 24, 24}, "input");
        Tensor other(DT_FP32, {2, 4, 24, 24}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 4, 24, 24}, "output");
        FUNCTION("WHERE_FP32_022") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_022");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_023 | 4D input+标量，N+H+W切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_023) {
    PROGRAM("WHERE_FP32_023") {
        TileShape::Current().SetVecTile(1, 4, 16, 16);
        Tensor condition(DT_UINT8, {2, 4, 32, 5}, "condition");
        Tensor input(DT_FP32, {2, 4, 32, 40}, "input");
        Element other(DT_FP32, 1.0);
        auto output = Tensor(DataType::DT_FP32, {2, 4, 32, 40}, "output");
        FUNCTION("WHERE_FP32_023") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_023");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_024 | 4D input+标量，C+H+W切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_024) {
    PROGRAM("WHERE_FP32_024") {
        TileShape::Current().SetVecTile(2, 2, 16, 16);
        Tensor condition(DT_UINT8, {2, 4, 40, 5}, "condition");
        Tensor input(DT_FP32, {2, 4, 40, 40}, "input");
        Element other(DT_FP32, 1.0);
        auto output = Tensor(DataType::DT_FP32, {2, 4, 40, 40}, "output");
        FUNCTION("WHERE_FP32_024") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_024");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_025 | 4D 双张量，NCW三轴切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_025) {
    PROGRAM("WHERE_FP32_025") {
        TileShape::Current().SetVecTile(1, 1, 16, 16);
        Tensor condition(DT_UINT8, {2, 3, 32, 5}, "condition");
        Tensor input(DT_FP32, {2, 3, 32, 40}, "input");
        Tensor other(DT_FP32, {2, 3, 32, 40}, "other");
        auto output = Tensor(DataType::DT_FP32, {2, 3, 32, 40}, "output");
        FUNCTION("WHERE_FP32_025") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_025");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_026 | 2D 双标量，N+W占位切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_026) {
    PROGRAM("WHERE_FP32_026") {
        TileShape::Current().SetVecTile(1, 8);
        Tensor condition(DT_UINT8, {2, 8}, "condition");
        Element input(DT_FP32, 1.0);
        Element other(DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {2, 64}, "output");
        FUNCTION("WHERE_FP32_026") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_026");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_027 | 3D 双标量，N+H+W占位切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_027) {
    PROGRAM("WHERE_FP32_027") {
        TileShape::Current().SetVecTile(1, 4, 8);
        Tensor condition(DT_UINT8, {1, 1, 8}, "condition");
        Element input(DT_FP32, 1.0);
        Element other(DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {1, 8, 64}, "output");
        FUNCTION("WHERE_FP32_027") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_027");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}

// fp32_028 | 4D 双标量，N+C+H+W占位切分
TEST_F(LiteNPUCodeGenWhere, test_where_fp32_028) {
    PROGRAM("WHERE_FP32_028") {
        TileShape::Current().SetVecTile(1, 1, 2, 8);
        Tensor condition(DT_UINT8, {1, 1, 1, 8}, "condition");
        Element input(DT_FP32, 1.0);
        Element other(DT_FP32, 2.0);
        auto output = Tensor(DataType::DT_FP32, {1, 1, 16, 64}, "output");
        FUNCTION("WHERE_FP32_028") {
            output = Where(condition, input, other);
        }
    }

    auto function = Program::GetInstance().GetFunctionByRawName(FUNCTION_PREFIX + "WHERE_FP32_028");
    npu::tile_fwk::CodeGenCtx ctx;
    npu::tile_fwk::CodeGenLiteNPU codeGen(ctx);
    codeGen.GenCode(*function, {});
}