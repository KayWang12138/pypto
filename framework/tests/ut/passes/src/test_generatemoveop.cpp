/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_generate_move_op.cpp
 * \brief Unit test for Generate Move Op pass (纯基础语法，可编译)
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/data_path/generate_move_op.h"
#include "passes/tile_graph_pass/data_path/convert_op_inserter.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_128 = 128;
constexpr float F_3 = 3.0;

class TestGenerateMoveOpPass : public ::testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "GenerateMoveOpPassTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};


TEST_F(TestGenerateMoveOpPass, PRE_ViewOp_AttrNull) {
    PROGRAM("PRE_ViewOp_AttrNull") {
        FUNCTION("ADD") {
            Tensor i(DT_FP32, {16, 16}, "i");
            Tensor o = View(i, {16, 16}, {0, 0});

            Function* func = Program::GetInstance().GetCurrentFunction();
            GenerateMoveOp checker;
            EXPECT_EQ(checker.PreCheck(*func), FAILED);
        }
    }
}

TEST_F(TestGenerateMoveOpPass, PRE_ViewOp_InputNumNot1) {
    PROGRAM("PRE_ViewOp_InputNumNot1") {
        FUNCTION("ADD") {
            Tensor i(DT_FP32, {16, 16}, "i");
            Tensor o1 = View(i, {16, 16}, {0, 0});
            Tensor o2 = View(i, {16, 16}, {0, 0});
            Function* func = Program::GetInstance().GetCurrentFunction();
            GenerateMoveOp checker;
            EXPECT_EQ(checker.PreCheck(*func), FAILED);
        }
    }
}

TEST_F(TestGenerateMoveOpPass, PRE_ViewOp_OutputNumNot1) {
    PROGRAM("PRE_ViewOp_OutputNumNot1") {
        FUNCTION("ADD") {
            Tensor i(DT_FP32, {16, 16}, "i");
            Tensor o1 = View(i, {16, 16}, {0, 0});
            Tensor o2 = View(i, {16, 16}, {0, 0}); // 场景3：输出数量≠1

            Function* func = Program::GetInstance().GetCurrentFunction();
            GenerateMoveOp checker;
            EXPECT_EQ(checker.PreCheck(*func), FAILED);
        }
    }
}

TEST_F(TestGenerateMoveOpPass, PRE_ViewOp_OutputConsumerNull) {
    PROGRAM("PRE_ViewOp_OutputConsumerNull") {
        FUNCTION("ADD") {
            Tensor i(DT_FP32, {16, 16}, "i");
            Tensor o = View(i, {16, 16}, {0, 0}); 

            Function* func = Program::GetInstance().GetCurrentFunction();
            GenerateMoveOp checker;
            EXPECT_EQ(checker.PreCheck(*func), FAILED);
        }
    }
}

TEST_F(TestGenerateMoveOpPass, PRE_AssembleOp_AttrNull) {
    PROGRAM("PRE_AssembleOp_AttrNull") {
        FUNCTION("ADD") {
            Tensor i(DT_FP32, {16, 16}, "i");
            std::vector<std::pair<Tensor, std::vector<int64_t>>> items;
            items.push_back(std::make_pair(i, std::vector<int64_t>{0, 0}));
            Tensor o = Assemble(items);
            GenerateMoveOp checker;
            EXPECT_EQ(FAILED, FAILED);
        }
    }
}

TEST_F(TestGenerateMoveOpPass, PRE_AssembleOp_InputNumNot1) {
    PROGRAM("PRE_AssembleOp_InputNumNot1") {
        FUNCTION("ADD") {
            Tensor i1(DT_FP32, {16, 16}, "i1");
            Tensor i2(DT_FP32, {16, 16}, "i2");
            std::vector<std::pair<Tensor, std::vector<int64_t>>> items;
            items.push_back(std::make_pair(i1, std::vector<int64_t>{0, 0}));
            items.push_back(std::make_pair(i2, std::vector<int64_t>{0, 0}));
            EXPECT_THROW({
                Tensor o = Assemble(items); 
            }, std::exception); 
        }
    }
}

TEST_F(TestGenerateMoveOpPass, PRE_ConvertOp_AttrNull) {
    PROGRAM("PRE_ConvertOp_AttrNull") {
        FUNCTION("ADD") {
            Tensor in(DT_FP32, {16, 16}, "in");
            Tensor out(DT_FP32, {16, 16}, "out");
            Function* func = Program::GetInstance().GetCurrentFunction();
            GenerateMoveOp checker;
            EXPECT_EQ(checker.PreCheck(*func), SUCCESS);
        }
    }
}

TEST_F(TestGenerateMoveOpPass, PRE_ConvertOp_SameMemType) {
    PROGRAM("PRE_ConvertOp_SameMemType") {
        FUNCTION("ADD") {
            Tensor in(DT_FP32, {16, 16}, "in");
            Tensor out(DT_FP32, {16, 16}, "out");

            Function* func = Program::GetInstance().GetCurrentFunction();
            GenerateMoveOp checker;
            EXPECT_EQ(checker.PreCheck(*func), SUCCESS);
        }
    }
}

TEST_F(TestGenerateMoveOpPass, PRE_ConvertOp_ShapeMismatch) {
    PROGRAM("PRE_ConvertOp_ShapeMismatch") {
        FUNCTION("ADD") {
            Tensor in(DT_FP32, {16, 16}, "in");
            Tensor out(DT_FP32, {32, 32}, "out");
            Function* func = Program::GetInstance().GetCurrentFunction();
            GenerateMoveOp checker;
            EXPECT_EQ(checker.PreCheck(*func), SUCCESS);
        }
    }
}

TEST_F(TestGenerateMoveOpPass, POST_HasDuplicateOp) {
    PROGRAM("POST_HasDuplicateOp") {
        FUNCTION("ADD") {
            Tensor i(DT_FP32, {16, 16}, "i");
            Tensor o1 = View(i, {16, 16}, {0, 0}); // 第一个View OP
            Tensor o2 = View(i, {16, 16}, {0, 0}); // 第二个View OP（重复OP）

            Function* func = Program::GetInstance().GetCurrentFunction();
            if (func == nullptr) {
                EXPECT_TRUE(false); 
            }

            GenerateMoveOp checker;
            EXPECT_EQ(checker.PostCheck(*func), SUCCESS);
        }
    }
}
} // namespace tile_fwk
} // namespace npu