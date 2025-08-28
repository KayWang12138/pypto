/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_deadoperationeliminate.cpp
 * \brief Unit test for DeadOperationEliminate pass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"

#define private public
#include "passes/tile_graph_pass/dead_operation_eliminate.h"

namespace npu {
namespace tile_fwk{
static const uint16_t kNumZero = 0u;
static const uint16_t kNumOne = 1u;
static const uint16_t kNumEight = 8u;
static const size_t kSizeZero = 0UL;

class TestDeadOperationEliminatePass : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "DeadOpEliminateTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

/*
DeadOperationEliminate
inCast{8,8}->view->outCast{8,8}
            ->view->ddrTensor{8,8}
inCast{8,8}->view->outCast{8,8}
*/
TEST_F(TestDeadOperationEliminatePass, DeadOperationEliminateUTest1) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestDeadOperationEliminate", "TestDeadOperationEliminate", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    std::vector<int64_t> shape = {kNumEight, kNumEight};
    auto inCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ddrTensor = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &view1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {ddrTensor});
    auto &view2 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {inCast}, {outCast});

    currFunctionPtr->inCasts_.push_back(inCast);
    currFunctionPtr->outCasts_.push_back(outCast);

    DeadOperationEliminator deadopeliminator;
    auto status = deadopeliminator.EliminateDeadOperation(*currFunctionPtr);
    EXPECT_EQ(status, SUCCESS);

    uint32_t view_num = kNumZero;
    for (auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            EXPECT_NE(view1.GetOpMagic(), op.GetOpMagic());
            EXPECT_EQ(view2.GetOpMagic(), op.GetOpMagic());
            EXPECT_EQ(view2.GetInputOperand(kSizeZero), inCast);
            ++view_num;
        }
    }
    EXPECT_EQ(view_num, kNumOne);
}
}
}