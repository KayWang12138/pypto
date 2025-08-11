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
 * \file test_n_buffer_merge.cpp
 * \brief Unit test for n_buffer_merge pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "passes/pass_registry.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/n_buffer_merge.h"
#include <fstream>
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk{

class NBufferMergeTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

        constexpr int DbType = 1;
        constexpr int NBuffer = 2;
        Program::GetInstance().GetConfig().Set<int>(DB_TYPE, DbType);
        Program::GetInstance().GetConfig().Set<int>(NBUFFER_NUM, NBuffer);
    }

    void TearDown() override {}
};

TEST_F(NBufferMergeTest, TestNBufferMerge) {
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestNBufferMerge", "TestNBufferMerge", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int> shape = {8, 16};
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->subGraphID = subGraphID0;
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast2->subGraphID = subGraphID1;
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->subGraphID = subGraphID0;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor2->subGraphID = subGraphID0;
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->subGraphID = subGraphID1;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor4->subGraphID = subGraphID1;

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    auto &copy_out1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);
    auto &copy_out2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    NBufferMerge nPass;
    Pass &nbufferPass = nPass;
    nbufferPass.PreCheck(*currFunctionPtr);
    nbufferPass.Run(*currFunctionPtr, "", "");
    nbufferPass.PostCheck(*currFunctionPtr);
}
}
}