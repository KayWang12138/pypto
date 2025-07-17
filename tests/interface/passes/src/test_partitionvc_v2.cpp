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
 * \file test_partitionvc_v2.cpp
 * \brief Unit test for PartitionVC2 pass.
 */

#include "common/data_type.h"
#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "passes/tile_graph_pass/iso_partitioner.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class PartitionVC2Test : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "PartitionVC2TestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(PartitionVC2Test, TestHealthReport) {
    config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    config::SetPassConfig("PVC2_OOO", "GraphInitPass", "HEALTH_CHECK", true);
    // Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};

    // Initialize PassManager
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_4_A(DT_FP16, shape1, "out_tensor_4_A");
    Tensor out_tensor_4_B(DT_FP16, shape4, "out_tensor_4_B");

    FUNCTION("PartitionVC2Function") {
        auto out_tensor_1 = Reshape(in_tensor, shape2);
        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 1, 64);
        auto out_tensor_2_A = Transpose(out_tensor_1, {0, 1});
        auto out_tensor_2_B = Transpose(out_tensor_1, {0, 1});
        auto out_tensor_3_A = Reshape(out_tensor_2_A, shape1);
        auto out_tensor_3_B = Reshape(out_tensor_2_B, shape4);
        out_tensor_4_A = Cast(out_tensor_3_A, DT_FP16);
        out_tensor_4_B = Cast(out_tensor_3_B, DT_FP16);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::string jsonFilePath = "./config/pass/json/partitionvc_v2_health_report.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    Function* func = Program::GetInstance().GetCurrentFunction();
    (void)func;
}

TEST_F(PartitionVC2Test, TestExistIsomorphismGraph) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    // Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PartitionVC2TestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_4_A(DT_FP16, shape1, "out_tensor_4_A");
    Tensor out_tensor_4_B(DT_FP16, shape4, "out_tensor_4_B");

    FUNCTION("PartitionVC2Function") {
        auto out_tensor_1 = Reshape(in_tensor, shape2);
        Program::GetInstance().GetTileShape().SetVecTileShapes(1, 1, 1, 64);
        auto out_tensor_2_A = Transpose(out_tensor_1, {0, 1});
        auto out_tensor_2_B = Transpose(out_tensor_1, {0, 1});
        auto out_tensor_3_A = Reshape(out_tensor_2_A, shape1);
        auto out_tensor_3_B = Reshape(out_tensor_2_B, shape4);
        out_tensor_4_A = Cast(out_tensor_3_A, DT_FP16);
        out_tensor_4_B = Cast(out_tensor_3_B, DT_FP16);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    // Init Program and Function
    std::string jsonFilePath = "./config/pass/json/partitionvc_v2_exist_isomorphism_graph.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    // Print and draw the graph
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PartitionVC2Function");
    // Call the pass
    GraphPartitionPass graphPartitionPass;
    graphPartitionPass.PreCheck(*func);
    graphPartitionPass.RunOnFunction(*func);
    graphPartitionPass.PostCheck(*func);

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

    // ================== Verify Pass Effect ==================
    auto loop_set = func->LoopCheck();
    std::cout << "loop_set.size() = " << loop_set.size() << std::endl;
    for (auto i : loop_set) {
        std::cout << "in loop_set, loop_set[i] number = " << i << std::endl;
    }
    bool isExistIsomorphismGraph = false;
    auto updated_operations = func->Operations();
    std::cout << "updated_operations.size() = " << updated_operations.size() << std::endl;
    for (auto &updated_operation : updated_operations) {
        std::cout << "updated_operations[i].GetOpcode() = " << updated_operation.GetOpcodeStr() << std::endl;
        std::cout << "updated_operations[i].opmagic = " << updated_operation.opmagic << std::endl;
        std::cout << "updated_operations[i].GetSubgraphID() = " << updated_operation.GetSubgraphID() << std::endl;
        if (updated_operation.GetSubgraphID() != 0) {
            isExistIsomorphismGraph = true;
        }
    }
    int opSize = 18;
    int INDEX_16 = 16;
    EXPECT_EQ(loop_set.size(), 0) << "SubGraphInCycle number is 0";
    EXPECT_EQ(updated_operations.size(), opSize) << "18 operations should remain(CopyIn + Reshape + Transpose + Cast + CopyOut)";
    EXPECT_EQ(isExistIsomorphismGraph, true) << "Exist IsomorphismGraph";
    EXPECT_NE(updated_operations[0].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[2].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[3].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_EQ(updated_operations[INDEX_16].GetOpcode(), Opcode::OP_COPY_OUT) << "The last operation of graph is COPY_OUT";
    EXPECT_EQ(updated_operations[17].GetOpcode(), Opcode::OP_COPY_OUT) << "The last operation of graph is COPY_OUT";
}

TEST_F(PartitionVC2Test, TestCheckCycle) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    // Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PartitionVC2TestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_1(DT_FP32, shape2, "out_tensor_1");
    Tensor out_tensor_2_A(DT_FP32, shape3, "out_tensor_2_A");
    Tensor out_tensor_2_B(DT_FP32, shape3, "out_tensor_2_B");
    Tensor out_tensor_0_A(DT_FP16, shape3, "out_tensor_0_A");
    Tensor out_tensor_0_B(DT_FP16, shape3, "out_tensor_0_B");
    Tensor out_tensor_3_A(DT_FP16, shape1, "out_tensor_3_A");
    Tensor out_tensor_3_B(DT_FP16, shape1, "out_tensor_3_B");
    Tensor out_tensor_4_A(DT_FP8, shape2, "out_tensor_4_A");
    Tensor out_tensor_4_B(DT_FP8, shape2, "out_tensor_4_B");
    Tensor out_tensor_5(DT_FP16, shape1, "out_tensor_5");
    Tensor out_tensor_6_A(DT_FP16, shape2, "out_tensor_4_A");
    Tensor out_tensor_6_B(DT_FP16, shape2, "out_tensor_4_B");

    FUNCTION("PartitionVC2Function") {
        auto f_out_tensor_1 = Reshape(in_tensor, shape2);
        auto f_out_tensor_2_A = Transpose(f_out_tensor_1, {0, 1});
        auto f_out_tensor_2_B = Transpose(f_out_tensor_1, {0, 1});
        auto f_out_tensor_0_A = Cast(f_out_tensor_2_A, DT_FP16);
        auto f_out_tensor_0_B = Cast(f_out_tensor_2_B, DT_FP16);
        auto f_out_tensor_3_A = Reshape(f_out_tensor_0_A, shape1);
        auto f_out_tensor_3_B = Reshape(f_out_tensor_0_B, shape1);
        auto f_out_tensor_5 = Add(f_out_tensor_3_A, f_out_tensor_3_B);
        auto f_out_tensor_6_A = Reshape(f_out_tensor_5, shape2);
        auto f_out_tensor_6_B = Reshape(f_out_tensor_5, shape2);
        out_tensor_4_A = Cast(f_out_tensor_6_A, DT_FP8);
        out_tensor_4_B = Cast(f_out_tensor_6_B, DT_FP8);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/partitionvc_v2_check_cycle.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    // Print and draw the graph
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PartitionVC2Function");
    // Call the pass
    GraphPartitionPass graphPartitionPass;
    graphPartitionPass.PreCheck(*func);
    graphPartitionPass.RunOnFunction(*func);
    graphPartitionPass.PostCheck(*func);

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

    // ================== Verify Pass Effect ==================
    auto loop_set = func->LoopCheck();
    std::cout << "loop_set.size() = " << loop_set.size() << std::endl;
    for (auto i : loop_set) {
        std::cout << "in loop_set, loop_set[i] number = " << i << std::endl;
    }
    bool isExistIsomorphismGraph = false;
    auto updated_operations = func->Operations();
    std::cout << "updated_operations.size() = " << updated_operations.size() << std::endl;
    for (auto &updated_operation : updated_operations) {
        std::cout << "updated_operations[i].GetOpcode() = " << updated_operation.GetOpcodeStr() << std::endl;
        std::cout << "updated_operations[i].opmagic = " << updated_operation.opmagic << std::endl;
        std::cout << "updated_operations[i].GetSubgraphID() = " << updated_operation.GetSubgraphID() << std::endl;
        if (updated_operation.GetSubgraphID() != 0) {
            isExistIsomorphismGraph = true;
        }
    }
    (void)isExistIsomorphismGraph;

    int opSize = 23;
    int INDEX_22 = 22;
    EXPECT_EQ(loop_set.size(), 0) << "SubGraphInCycle number is 0";
    EXPECT_EQ(updated_operations.size(), opSize) << "23 operations should remain(CopyIn + Reshape + Transpose + Cast + Add + CopyOut)";
    EXPECT_NE(updated_operations[0].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[11].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[12].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_EQ(updated_operations[21].GetOpcode(), Opcode::OP_COPY_OUT) << "The last operation of graph is COPY_OUT";
    EXPECT_EQ(updated_operations[INDEX_22].GetOpcode(), Opcode::OP_COPY_OUT) << "The last operation of graph is COPY_OUT";
}

TEST_F(PartitionVC2Test, TestExistIsomorphismGraphWithSingleCopyOut) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    // Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PartitionVC2TestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_1(DT_FP32, shape2, "out_tensor_1");
    Tensor out_tensor_2_A(DT_FP32, shape3, "out_tensor_2_A");
    Tensor out_tensor_2_B(DT_FP32, shape3, "out_tensor_2_B");
    Tensor out_tensor_3_A(DT_FP32, shape1, "out_tensor_3_A");
    Tensor out_tensor_3_B(DT_FP32, shape4, "out_tensor_3_B");
    Tensor out_tensor_4_A(DT_FP16, shape1, "out_tensor_4_A");
    Tensor out_tensor_4_B(DT_FP16, shape4, "out_tensor_4_B");

    FUNCTION("PartitionVC2Function") {
        out_tensor_1 = Reshape(in_tensor, shape2);
        out_tensor_2_A = Transpose(out_tensor_1, {0, 1});
        out_tensor_2_B = Transpose(out_tensor_1, {0, 1});
        out_tensor_3_A = Reshape(out_tensor_2_A, shape1);
        out_tensor_3_B = Reshape(out_tensor_2_B, shape4);
        out_tensor_4_A = Cast(out_tensor_3_A, DT_FP16);
        out_tensor_4_B = Cast(out_tensor_3_B, DT_FP16);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/partitionvc_v2_exist_isomorphism_graph_with_single_copyout.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    // Print and draw the graph
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PartitionVC2Function");
    // Call the pass
    GraphPartitionPass graphPartitionPass;
    graphPartitionPass.PreCheck(*func);
    graphPartitionPass.RunOnFunction(*func);
    graphPartitionPass.PostCheck(*func);

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

    // ================== Verify Pass Effect ==================
    auto loop_set = func->LoopCheck();
    std::cout << "loop_set.size() = " << loop_set.size() << std::endl;
    for (auto i : loop_set) {
        std::cout << "in loop_set, loop_set[i] number = " << i << std::endl;
    }
    bool isExistIsomorphismGraph = false;
    if (!func->sorted_) {
        std::cout << "================== Sorting Operations ... ==================" << std::endl;
        func->SortOperations();
    }
    auto updated_operations = func->Operations();
    std::cout << "updated_operations.size() = " << updated_operations.size() << std::endl;
    for (size_t i = 0 ; i < updated_operations.size(); i++) {
        std::cout << "updated_operations[" << i << "].GetOpcode() = " << updated_operations[i].GetOpcodeStr() << std::endl;
        std::cout << "updated_operations[" << i << "].opmagic = " << updated_operations[i].opmagic << std::endl;
        std::cout << "updated_operations[" << i << "].GetSubgraphID() = " << updated_operations[i].GetSubgraphID() << std::endl;
        if (updated_operations[i].GetSubgraphID() != 0) {
            isExistIsomorphismGraph = true;
        }
    }

    int opSize = 23;
    EXPECT_EQ(loop_set.size(), 0) << "SubGraphInCycle number is 0";
    EXPECT_EQ(updated_operations.size(), opSize) << "23 operations should remain(CopyIn + Reshape + Transpose + Cast + CopyOut)";
    EXPECT_EQ(isExistIsomorphismGraph, true) << "Exist IsomorphismGraph";
    EXPECT_NE(updated_operations[0].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[2].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[3].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    for (int i = 12; i <= 18; i++) {
        auto &op = updated_operations[17];
        ASSERT(op.GetOpcode() == Opcode::OP_COPY_OUT || op.GetOpcode() == Opcode::OP_ASSEMBLE);
        if (op.GetInputOperand(0)->GetMemoryTypeOriginal() == npu::tile_fwk::MEM_UB) {
            ASSERT(op.GetOpcode() == Opcode::OP_COPY_OUT);
        } else {
            ASSERT(op.GetOpcode() == Opcode::OP_ASSEMBLE);
        }
    }
}

TEST_F(PartitionVC2Test, TestNonexistIsomorphismGraphWithSingleCopyOut) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    // Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PartitionVC2TestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},

    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_1(DT_FP32, shape2, "out_tensor_1");
    Tensor out_tensor_2_A(DT_FP16, shape2, "out_tensor_2_A");
    Tensor out_tensor_2_B(DT_FP32, shape3, "out_tensor_2_B");
    Tensor out_tensor_3_A(DT_FP16, shape1, "out_tensor_3_A");
    Tensor out_tensor_3_B(DT_FP32, shape4, "out_tensor_3_B");
    Tensor out_tensor_4_A(DT_FP32, shape2, "out_tensor_4_A");
    Tensor out_tensor_4_B(DT_FP32, shape3, "out_tensor_4_B");

    FUNCTION("PartitionVC2Function") {
        out_tensor_1 = Reshape(in_tensor, shape2);
        out_tensor_2_A = Cast(out_tensor_1, DT_FP16);
        out_tensor_2_B = Transpose(out_tensor_1, {0, 1});
        out_tensor_3_A = Reshape(out_tensor_2_A, shape1);
        out_tensor_3_B = Reshape(out_tensor_2_B, shape4);
        out_tensor_4_A = Cast(out_tensor_3_A, DT_FP32);
        out_tensor_4_B = Cast(out_tensor_3_B, DT_FP16);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/partitionvc_v2_nonexist_isomorphism_graph_with_single_copyout.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    // Print and draw the graph
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PartitionVC2Function");

    // Call the pass
    GraphPartitionPass graphPartitionPass;
    graphPartitionPass.PreCheck(*func);
    graphPartitionPass.RunOnFunction(*func);
    graphPartitionPass.PostCheck(*func);

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

    // ================== Verify Pass Effect ==================
    auto loop_set = func->LoopCheck();
    std::cout << "loop_set.size() = " << loop_set.size() << std::endl;
    for (auto i : loop_set) {
        std::cout << "in loop_set, loop_set[i] number = " << i << std::endl;
    }
    bool isExistIsomorphismGraph = false;
    auto updated_operations = func->Operations();
    std::cout << "updated_operations.size() = " << updated_operations.size() << std::endl;
    size_t copyOutNum = 0;
    size_t assembleNum = 0;
    for (size_t i = 0; i < updated_operations.size(); i++) {
        std::cout << "updated_operations[" << i << "].GetOpcode() = " << updated_operations[i].GetOpcodeStr() << std::endl;
        std::cout << "updated_operations[" << i << "].opmagic = " << updated_operations[i].opmagic << std::endl;
        std::cout << "updated_operations[" << i << "].GetSubgraphID() = " << updated_operations[i].GetSubgraphID() << std::endl;
        if (updated_operations[i].GetSubgraphID() != 0) {
            isExistIsomorphismGraph = false;
        }
        if (updated_operations[i].GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutNum++;
        } else if (updated_operations[i].GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleNum++;
        }
    }

    int opSize = 19;
    EXPECT_EQ(loop_set.size(), 0) << "SubGraphInCycle number is 0";
    EXPECT_EQ(updated_operations.size(), opSize) << "19 operations should remain(CopyIn + Reshape + Transpose + Cast + CopyOut)";
    EXPECT_NE(isExistIsomorphismGraph, true) << "Do not exist IsomorphismGraph";
    EXPECT_NE(updated_operations[0].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[2].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[3].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_EQ(copyOutNum, 6) << "The number of OP_COPY_OUT is 6";
    EXPECT_EQ(assembleNum, 1) << "The number of OP_ASSEMBLE is 1";
}

TEST_F(PartitionVC2Test, TestCheckCycleWithSingleCopyOut) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    // Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PartitionVC2TestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_1(DT_FP32, shape2, "out_tensor_1");
    Tensor out_tensor_2_A(DT_FP32, shape3, "out_tensor_2_A");
    Tensor out_tensor_2_B(DT_FP32, shape3, "out_tensor_2_B");
    Tensor out_tensor_0_A(DT_FP16, shape3, "out_tensor_0_A");
    Tensor out_tensor_0_B(DT_FP16, shape3, "out_tensor_0_B");
    Tensor out_tensor_3_A(DT_FP16, shape1, "out_tensor_3_A");
    Tensor out_tensor_3_B(DT_FP16, shape1, "out_tensor_3_B");
    Tensor out_tensor_4_A(DT_FP8, shape2, "out_tensor_4_A");
    Tensor out_tensor_4_B(DT_FP8, shape2, "out_tensor_4_B");
    Tensor out_tensor_5(DT_FP16, shape1, "out_tensor_5");
    Tensor out_tensor_6_A(DT_FP16, shape2, "out_tensor_4_A");
    Tensor out_tensor_6_B(DT_FP16, shape2, "out_tensor_4_B");

    FUNCTION("PartitionVC2Function") {
        out_tensor_1 = Reshape(in_tensor, shape2);
        auto f_out_tensor_2_A = Transpose(out_tensor_1, {0, 1});
        auto f_out_tensor_2_B = Transpose(out_tensor_1, {0, 1});
        out_tensor_0_A = Cast(f_out_tensor_2_A, DT_FP16);
        out_tensor_0_B = Cast(f_out_tensor_2_B, DT_FP16);
        out_tensor_3_A = Reshape(out_tensor_0_A, shape1);
        out_tensor_3_B = Reshape(out_tensor_0_B, shape1);
        auto f_out_tensor_5 = Add(out_tensor_3_A, out_tensor_3_B);
        out_tensor_6_A = Reshape(f_out_tensor_5, shape2);
        out_tensor_6_B = Reshape(f_out_tensor_5, shape2);
        out_tensor_4_A = Cast(out_tensor_6_A, DT_FP8);
        out_tensor_4_B = Cast(out_tensor_6_B, DT_FP8);

        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/partitionvc_v2_check_cycle_with_single_copyout.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    // Print and draw the graph
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PartitionVC2Function");
    // Call the pass
    GraphPartitionPass graphPartitionPass;
    graphPartitionPass.PreCheck(*func);
    graphPartitionPass.RunOnFunction(*func);
    graphPartitionPass.PostCheck(*func);

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

    // ================== Verify Pass Effect ==================
    auto loop_set = func->LoopCheck();
    std::cout << "loop_set.size() = " << loop_set.size() << std::endl;
    for (auto i : loop_set) {
        std::cout << "in loop_set, loop_set[i] number = " << i << std::endl;
    }
    bool isExistIsomorphismGraph = false;
    auto updated_operations = func->Operations();
    std::cout << "updated_operations.size() = " << updated_operations.size() << std::endl;
    for (auto &updated_operation : updated_operations) {
        std::cout << "updated_operations[i].GetOpcode() = " << updated_operation.GetOpcodeStr() << std::endl;
        std::cout << "updated_operations[i].opmagic = " << updated_operation.opmagic << std::endl;
        std::cout << "updated_operations[i].GetSubgraphID() = " << updated_operation.GetSubgraphID() << std::endl;
        if (updated_operation.GetSubgraphID() != 0) {
            isExistIsomorphismGraph = true;
        }
    }
    (void)isExistIsomorphismGraph;

    int opSize = 30;
    int INDEX_28 = 28;
    int INDEX_29 = 29;
    EXPECT_EQ(loop_set.size(), 0) << "SubGraphInCycle number is 0";
    EXPECT_EQ(updated_operations.size(), opSize) << "30 operations should remain(CopyIn + Reshape + Transpose + Cast + Add + CopyOut)";
    EXPECT_NE(updated_operations[0].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[10].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_NE(updated_operations[11].GetOpcode(), Opcode::OP_COPY_OUT) << "The first operation of graph is not COPY_OUT";
    EXPECT_EQ(updated_operations[INDEX_28].GetOpcode(), Opcode::OP_COPY_OUT) << "The last operation of graph is COPY_OUT";
    EXPECT_EQ(updated_operations[INDEX_29].GetOpcode(), Opcode::OP_COPY_OUT) << "The last operation of graph is COPY_OUT";
}

TEST_F(PartitionVC2Test, TestOnlyReshape) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    // Define the shape of the Tensors
    std::vector<int> shape1{2, 1};
    std::vector<int> shape2{1, 2};

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PartitionVC2TestStrategy", {
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor(DT_FP32, shape2, "out_tensor");
    Tensor in_tensor1(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor1(DT_FP32, shape2, "out_tensor");

    FUNCTION("PartitionVC2Function") {
        out_tensor = Reshape(in_tensor, shape2);
        out_tensor1 = Reshape(in_tensor1, shape2);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/partitionvc_v2_only_reshape.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
        Json readData = LoadJsonFile(jsonFilePath);
        Program::GetInstance().LoadJson(readData);
    }

    // Print and draw the graph
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PartitionVC2Function");
    // Call the pass
    GraphPartitionPass graphPartitionPass;
    graphPartitionPass.PreCheck(*func);
    graphPartitionPass.RunOnFunction(*func);
    graphPartitionPass.PostCheck(*func);

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

    // ================== Verify Pass Effect ==================
    auto loop_set = func->LoopCheck();
    std::cout << "loop_set.size() = " << loop_set.size() << std::endl;
    for (auto i : loop_set) {
        std::cout << "in loop_set, loop_set[i] number = " << i << std::endl;
    }
    bool isExistIsomorphismGraph = false;
    auto updated_operations = func->Operations();
    std::cout << "updated_operations.size() = " << updated_operations.size() << std::endl;
    for (auto &updated_operation : updated_operations) {
        std::cout << "updated_operations[i].GetOpcode() = " << updated_operation.GetOpcodeStr() << std::endl;
        std::cout << "updated_operations[i].opmagic = " << updated_operation.opmagic << std::endl;
        std::cout << "updated_operations[i].GetSubgraphID() = " << updated_operation.GetSubgraphID() << std::endl;
        if (updated_operation.GetSubgraphID() != 0) {
            isExistIsomorphismGraph = true;
        }
    }
    (void)isExistIsomorphismGraph;

    EXPECT_EQ(loop_set.size(), 0) << "SubGraphInCycle number is 0";
    EXPECT_EQ(updated_operations.size(), 6) << "6 operations should remain(View + Reshape + ASSEMBLE)";
    EXPECT_NE(updated_operations[0].GetOpcode(), Opcode::OP_ASSEMBLE) << "The first operation of graph is VIEW";
    EXPECT_NE(updated_operations[1].GetOpcode(), Opcode::OP_ASSEMBLE) << "The first operation of graph is VIEW";
    EXPECT_EQ(updated_operations[4].GetOpcode(), Opcode::OP_ASSEMBLE) << "The last operation of graph is ASSEMBLE";
    EXPECT_EQ(updated_operations[5].GetOpcode(), Opcode::OP_ASSEMBLE) << "The last operation of graph is ASSEMBLE";
}
