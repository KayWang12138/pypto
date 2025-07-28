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
 * \file test_partitionvc.cpp
 * \brief Unit test for PartitionVC pass.
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/tile_graph_pass/partitioner.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

void PrintGraphInfoPartitionVC(Function* func) {
    std::cout << "func->Operations().size() = "  << func->Operations().size() << std::endl;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.GetOpMagic() << " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
        }
        std::cout << std::endl;
    }
}

class PartitionVCTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "PartitionVCTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(PartitionVCTest, TestOnlyReshape) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int> shape1{2, 1};
    std::vector<int> shape2{1, 2};

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PartitionVCTestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor(DT_FP32, shape2, "out_tensor");
    Tensor in_tensor1(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor1(DT_FP32, shape2, "out_tensor");

    FUNCTION("PartitionVCFunction") {
        out_tensor = Reshape(in_tensor, shape2);
        out_tensor1 = Reshape(in_tensor1, shape2);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/partitionvc_only_reshape.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PartitionVCFunction");
    PartitionVCPass partitionVCPass;
    partitionVCPass.PreCheck(*func);
    partitionVCPass.RunOnFunction(*func);
    partitionVCPass.PostCheck(*func);

    PrintGraphInfoPartitionVC(func);

    // ================== Verify Pass Effect ==================
    auto loop_set = func->LoopCheck();
    std::cout << "loop_set.size() = " << loop_set.size() << std::endl;
    for (auto i : loop_set) {
        std::cout << "in loop_set, loop_set[i] number = " << i << std::endl;
    }
    bool isGraphSplit = false;
    auto updated_operations = func->Operations();
    std::cout << "updated_operations.size() = " << updated_operations.size() << std::endl;
    for (size_t i = 0; i < updated_operations.size(); i++) {
        std::cout << "updated_operations[i].GetOpcode() = " << updated_operations[i].GetOpcodeStr() << std::endl;
        std::cout << "updated_operations[i].opmagic = " << updated_operations[i].opmagic << std::endl;
        std::cout << "updated_operations[i].GetSubgraphID() = " << updated_operations[i].GetSubgraphID() << std::endl;
        if (updated_operations[i].GetSubgraphID() != 0) {
            isGraphSplit = true;
        }
    }
    int opSize = 6;
    EXPECT_EQ(loop_set.size(), 0) << "SubGraphInCycle number is 0";
    EXPECT_EQ(updated_operations.size(), opSize) << "6 operations should remain(View + Reshape + Assemble)";
    EXPECT_EQ(isGraphSplit, true) << "Graph split success";
    EXPECT_NE(updated_operations[0].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[1].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[2].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
}
