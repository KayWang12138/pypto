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
 * \file test_l1_copy_reuse.cpp
 * \brief Unit test for L1CopyInReuse pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "passes/pass_registry.h"
#include "interface/configs/config_manager.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class L1CopyInReuseTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

        Program::GetInstance().GetConfig().Set<int>(L1_REUSE, 4);
        Program::GetInstance().GetConfig().Set<std::map<int, int>>(L1_REUSE_MAP, {{0,2}});
        Program::GetInstance().GetConfig().Set<int>(CUBE_NBUFFER, 1);
        Program::GetInstance().GetConfig().Set<std::map<int, int>>(CUBE_NBUFFER_MAP, {{0,1}});
        Program::GetInstance().GetConfig().Set<bool>(LOAD_BALANCE, true);
    }

    void TearDown() override {}
};

TEST_F(L1CopyInReuseTest, TwoCopyIn) {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("L1ReusePassStrategy", {
        {        "L1CopyInReusePass",        "L1CopyInReusePass",    PassType::TYPE_TILE_GRAPH},
    });
    config::SetHostConfig(KEY_STRATEGY, "L1ReusePassStrategy");
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestL1CopyInReuse", "TestL1CopyInReuse", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);

    // Prepare the graph
    constexpr int subGraphID0 = 0;
    constexpr int subGraphID1 = 1;
    std::vector<int> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast1->tensor->rawmagic = 1;
    incast1->memoryTypeToBe_ = MEM_DEVICE_DDR;
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    incast2->tensor->rawmagic = 1;
    incast2->memoryTypeOriginal_ = MEM_DEVICE_DDR;
    auto tensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor1->memoryTypeOriginal_ = MEM_L1;
    tensor1->tensor->rawmagic = 2;
    auto tensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto tensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    tensor3->memoryTypeOriginal_ = MEM_L1;
    tensor3->tensor->rawmagic = 3;
    auto tensor4 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {tensor1});
    copy_op1.UpdateSubgraphID(subGraphID0);
    copy_op1.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L1, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>()));
    auto &copy_out1 = currFunctionPtr->AddOperation(Opcode::OP_L1_TO_L0A, {tensor1}, {tensor2});
    copy_out1.UpdateSubgraphID(subGraphID0);

    auto &view_op1 = currFunctionPtr->AddOperation(Opcode::OP_VIEW, {incast1}, {incast2});
    view_op1.SetOpAttribute(std::make_shared<ViewOpAttribute>(std::vector<int>{0, 0}));
    view_op1.UpdateSubgraphID(subGraphID1);
    auto &alloc_op1 = currFunctionPtr->AddOperation(Opcode::OP_L1_ALLOC, {}, {tensor3});
    alloc_op1.UpdateSubgraphID(subGraphID1);
    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {tensor3});
    copy_op2.UpdateSubgraphID(subGraphID1);
    incast2->AddConsumer(copy_op2);
    copy_op2.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified({0, 0}), MEM_L1, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>()));
    auto &copy_out2 = currFunctionPtr->AddOperation(Opcode::OP_L1_TO_L0A, {tensor3}, {tensor4});
    copy_out2.UpdateSubgraphID(subGraphID1);

    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->outCasts_.push_back(tensor2);
    currFunctionPtr->outCasts_.push_back(tensor4);

    // Call the pass
    const std::string passName = "L1CopyInReusePass";
    const std::string identifier = "L1CopyInReusePass";
    auto pass = PassRegistry::GetInstance().CreatePass(passName);
    pass->PreCheck(*currFunctionPtr);
    pass->Run(*currFunctionPtr, config::GetPassStrategy(), identifier);
    pass->PostCheck(*currFunctionPtr);
}
