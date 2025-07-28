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
 * \file test_subgraph_datapath.cpp
 * \brief Unit test for Subgraph Datapath passes.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "ut_json/ut_json_tool.h"

namespace npu::tile_fwk {
class TestSubgraphDatapath : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "SubgraphDatapathStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

std::shared_ptr<Function> PrepareStrategyAndFunction() {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("SubgraphDatapathStrategy", {
        {        "UpdateMemoryMap",        "UpdateMemoryMap", PassType::TYPE_TILE_GRAPH},
        {         "GenerateMoveOp",         "GenerateMoveOp", PassType::TYPE_TILE_GRAPH},
        { "SplitLargeLocalRawPass", "SplitLargeLocalRawPass", PassType::TYPE_TILE_GRAPH},
        {       "InsertCopyOpPass",       "InsertCopyOpPass", PassType::TYPE_TILE_GRAPH},
    });
    auto function = std::make_shared<Function>(Program::GetInstance(),
        "TestSubgraphDatapath", "TestSubgraphDatapath", nullptr);
    return function;
}

std::unordered_map<int, std::unordered_set<int>> CollectOperations(std::shared_ptr<Function> function) {
    std::unordered_map<int, std::unordered_set<int>> ret;
    for (auto& op : function->Operations()) {
        ret[op.GetSubgraphID()].insert(op.GetOpMagic());
    }
    return ret;
}

std::unordered_set<int> SetSubstract(std::unordered_set<int>& a, std::unordered_set<int>& b) {
    std::unordered_set<int> c;
    for (int i : a) {
        if (b.find(i) == b.end()) {
            c.insert(i);
        }
    }
    return c;
}

enum TestTensorType {
    TT_FUNC_IN = 0,
    TT_FUNC_OUT = 1,
    TT_FUNC_NONE,
};

LogicalTensorPtr CreateTestTensor(std::shared_ptr<Function> function, int& rawMagic,
    TestTensorType tensorType=TT_FUNC_NONE) {
    std::vector<int> shape = {16, 16};
    LogicalTensorPtr tensor = std::make_shared<LogicalTensor>(*function, DT_FP32, shape);
    tensor->memoryTypeOriginal_ = MEM_UB;
    tensor->memoryTypeToBe_ = MEM_UB;
    tensor->tensor->rawmagic = rawMagic++;
    if (tensorType == TT_FUNC_IN) {
        function->inCasts_.push_back(tensor);
    } else if (tensorType == TT_FUNC_OUT) {
        function->outCasts_.push_back(tensor);
    }
    return tensor;
}

LogicalTensors CollectBoundaryTensors(std::shared_ptr<Function> function) {
    LogicalTensors ret;
    for (auto &[magic, tensor] : function->GetTensorMap().inverseMap_) {
        (void)magic;
        std::set<int> colorSet;
        for (auto& producer : tensor->GetProducers()) {
            colorSet.insert(producer->GetSubgraphID());
        }
        for (auto& consumer : tensor->GetConsumers()) {
            colorSet.insert(consumer->GetSubgraphID());
        }
        if (colorSet.size() > 1) {
            ret.push_back(tensor);
        }
    }
    return ret;
}

Operation &AddTestOperation(std::shared_ptr<Function> function, const Opcode opcode,
    const LogicalTensors &iOperands, const LogicalTensors &oOperands, int subgraphId) {
    Operation &op = function->AddRawOperation(opcode, {iOperands}, {oOperands});
    op.UpdateSubgraphID(subgraphId);
    if (opcode == Opcode::OP_ASSEMBLE) {
        std::vector<int> offset = {0, 0};
        op.SetOpAttribute(std::make_shared<AssembleOpAttribute>(MEM_UB, offset));
    } else if (opcode == Opcode::OP_VIEW) {
        std::vector<int> offset = {0, 0};
        op.SetOpAttribute(std::make_shared<ViewOpAttribute>(offset, MEM_UB));
    }
    return op;
}

void ValidateSubgraphBoundary(std::shared_ptr<Function> function, size_t expectedNumTensors) {
    auto boundaryTensors = CollectBoundaryTensors(function);
    EXPECT_TRUE(boundaryTensors.size() == expectedNumTensors);
    for (auto& tensor : boundaryTensors) {
        EXPECT_TRUE(tensor->GetMemoryTypeOriginal() == MEM_DEVICE_DDR);
        EXPECT_TRUE(tensor->GetMemoryTypeToBe() == MEM_DEVICE_DDR);
        for (auto& producer : tensor->GetProducers()) {
            EXPECT_TRUE(producer->GetOpcode() == Opcode::OP_COPY_OUT);
        }
        for (auto& consumer : tensor->GetConsumers()) {
            EXPECT_TRUE(consumer->GetOpcode() == Opcode::OP_COPY_IN);
        }
    }
}

TEST_F(TestSubgraphDatapath, IntraSubgraphAdapter_SP_SC_UB) {
    // Single producer (not ASSEMBLE), single consumer (not VIEW)
    auto function = PrepareStrategyAndFunction();

    // Prepare the graph
    int rawMagic = 1;
    constexpr int expectedNumNewBoundaryTensors = 1;
    auto tensor1 = CreateTestTensor(function, rawMagic, TT_FUNC_IN);
    auto tensor2 = CreateTestTensor(function, rawMagic);
    auto tensor3 = CreateTestTensor(function, rawMagic, TT_FUNC_OUT);

    // subgraph 1
    constexpr int subgraphID0 = 0;
    constexpr int expectedNumNewOperations0 = 1;
    AddTestOperation(function, Opcode::OP_ADD, {tensor1}, {tensor2}, subgraphID0);
    
    // subgraph 2
    constexpr int subgraphID1 = 1;
    constexpr int expectedNumNewOperations1 = 1;
    AddTestOperation(function, Opcode::OP_SUB, {tensor2}, {tensor3}, subgraphID1);

    auto opsBefore = CollectOperations(function);
    PassManager::Instance().RunPass(Program::GetInstance(), *function, "SubgraphDatapathStrategy");
    auto opsAfter = CollectOperations(function);

    EXPECT_TRUE(opsAfter[subgraphID0].size() == (opsBefore[subgraphID0].size() + expectedNumNewOperations0));
    EXPECT_TRUE(opsAfter[subgraphID1].size() == (opsBefore[subgraphID1].size() + expectedNumNewOperations1));

    for (int opMagic : SetSubstract(opsAfter[subgraphID0], opsBefore[subgraphID0])) {
        EXPECT_TRUE(function->GetOpByOpMagic(opMagic)->GetOpcode() == Opcode::OP_COPY_OUT);
    }
    for (int opMagic : SetSubstract(opsAfter[subgraphID1], opsBefore[subgraphID1])) {
        EXPECT_TRUE(function->GetOpByOpMagic(opMagic)->GetOpcode() == Opcode::OP_COPY_IN);
    }

    ValidateSubgraphBoundary(function, expectedNumNewBoundaryTensors);
}
} // namespace npu::tile_fwk
