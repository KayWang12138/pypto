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
 * \file test_pre_graph.cpp
 * \brief Unit test for PreGraph pass.
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/tile_graph_pass/pre_graph.h"
#include "ut_json/ut_json_tool.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

void PrintGraphInfoPreGraph(Function* func, int& memoryMapSize, std::set<int>& tensorMagicWithColorSet) {
    std::cout << "func->Operations().size() = "  << func->Operations().size() << std::endl;
    for (auto &op : func->Operations()) {
        std::cout << "Op:" << op.GetOpMagic() << " " <<  op.GetOpcodeStr() << std::endl;
        std::cout << "input operation:";
        for (const std::shared_ptr<LogicalTensor> &input_tensor : op.GetIOperands()) {
            for (const auto &item_op : input_tensor->GetProducers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
            if (input_tensor->GetMemoryTypeOriginal() == npu::tile_fwk::MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memorySize = input_tensor->memorymap.size();
            memoryMapSize = memorySize == 0 ? 0 : memoryMapSize;
            std::cout << "input tensor, cur memory size is " << memorySize << std::endl;
            int curColor = input_tensor->subGraphID;
            std::cout << "input tensor, cur color is " << curColor << std::endl;
            if (curColor > 0) {
                tensorMagicWithColorSet.insert(input_tensor->magic);
                std::cout << "cur input tensor magic is " << input_tensor->magic << std::endl;
            }
        }
        std::cout << std::endl << "output operation:";
        for (const std::shared_ptr<LogicalTensor> &output_tensor : op.GetOOperands()) {
            for (const auto &item_op : output_tensor->GetConsumers()) {
                std::cout << "(" << item_op->opmagic << ", " << item_op->GetOpcodeStr() << ") ";
            }
            if (output_tensor->GetMemoryTypeOriginal() == npu::tile_fwk::MemoryType::MEM_DEVICE_DDR) {
                continue;
            }
            int memorySize = output_tensor->memorymap.size();
            memoryMapSize = memorySize == 0 ? 0 : memoryMapSize;
            std::cout << "output tensor, cur memory size is " << memorySize << std::endl;
            int curColor = output_tensor->subGraphID;
            std::cout << "output tensor, cur color is " << curColor << std::endl;
            if (curColor > 0) {
                tensorMagicWithColorSet.insert(output_tensor->magic);
                std::cout << "cur output tensor magic is " << output_tensor->magic << std::endl;
            }
        }
        std::cout << std::endl;
    }
}

class PreGraphTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "PreGraphTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(PreGraphTest, TestVCPartition) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    //Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};
    std::vector<int> shape5{2, 64};
    std::vector<int> shape6{64, 2};
    std::vector<int> shape7{2, 2};

    //Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PreGraphTestStrategy", {
    {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
    {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
    {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
    {        "GenerateMoveOp_01",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {              "CubeProcess",              "CubeProcess",    PassType::TYPE_TILE_GRAPH},
    {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
    {         "NBufferMergePass",         "NBufferMergePass",    PassType::TYPE_TILE_GRAPH},
    {          "UpdateMemoryMap",          "UpdateMemoryMap",    PassType::TYPE_TILE_GRAPH},
    {        "GenerateMoveOp_02",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
    {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
    { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
    {        "L1CopyInReusePass",        "L1CopyInReusePass",    PassType::TYPE_TILE_GRAPH},

    });
    ConfigManager::Instance();

    //Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    //Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor(DT_FP16, shape7, "in_tensor");

    FUNCTION("PreGraphFunction") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({2, 1, 1, 64});
        auto out_tensor_1_A = Reshape(in_tensor, shape2);
        auto out_tensor_1_B = Reshape(in_tensor, shape2);
        auto out_tensor_2_A = Transpose(out_tensor_1_A, {0, 1});
        auto out_tensor_2_B = Transpose(out_tensor_1_B, {0, 1});
        auto out_tensor_3_A = Reshape(out_tensor_2_A, shape1);
        auto out_tensor_3_B = Reshape(out_tensor_2_B, shape4);
        Program::GetInstance().GetTileShape().SetVecTileShapes({2, 1, 64});
        auto out_tensor_4_A = Cast(out_tensor_3_A, DT_FP16);
        auto out_tensor_4_B = Cast(out_tensor_3_B, DT_FP16);
        auto out_tensor_5_A = Reshape(out_tensor_4_A, shape5);
        auto out_tensor_5_B = Reshape(out_tensor_4_A, shape6);
        Program::GetInstance().GetTileShape().SetCubeTileShapes({2, 2}, {64, 64}, {2, 2});
        out_tensor = npu::tile_fwk::Matrix::Matmul<false, false>(DataType::DT_FP32, out_tensor_5_A, out_tensor_5_B);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::string jsonFilePath = "./config/pass/json/pre_graph_vc_partition.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PreGraphFunction");
    npu::tile_fwk::PreGraphPass preGraphPass;
    preGraphPass.PreCheck(*func);
    preGraphPass.RunOnFunction(*func);
    preGraphPass.PostCheck(*func);

    int memoryMapSize = 1;
    std::set<int> tensorMagicWithColorSet;
    PrintGraphInfoPreGraph(func, memoryMapSize, tensorMagicWithColorSet);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 14;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 12 operations";
    const int lastOpIdx = opSize - 1;
    EXPECT_EQ(updated_operations[lastOpIdx].GetOpcode(), Opcode::OP_ASSEMBLE) << "The last operation of graph should be Assemble";
    auto inputTensor = updated_operations[lastOpIdx].GetIOperands().front();
    auto outputTensor = updated_operations[lastOpIdx].GetOOperands().front();
    EXPECT_EQ(inputTensor->GetMemoryTypeOriginal(), MemoryType::MEM_L0C) << "The last operation's input is on L0C";
    EXPECT_EQ(outputTensor->GetMemoryTypeOriginal(), MemoryType::MEM_DEVICE_DDR) << "The last operation's output is on DDR";
}

TEST_F(PreGraphTest, TestAssemble) {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PreGraphTestStrategy", {
    {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
    {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
    {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
    {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
    {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
    {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
    { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},

    });
    int dim1 = 8;
    int dim2 = 2;
    int dim3 = 64;
    Program::GetInstance().GetTileShape().SetVecTileShapes(dim1, dim1, dim1, dim1);
    Tensor input(DT_FP32, {1, 384}, "a");
    Tensor res1;
    FUNCTION("TestAssign") {
        Program::GetInstance().GetTileShape().SetVecTileShapes(1, dim3);
        Tensor res = Exp(input);
        Tensor test = Reshape(res, {2, 1, 1, 192});
        Program::GetInstance().GetTileShape().SetVecTileShapes(dim2, 1, dim2, dim3);
        res1 = Exp(test);
    }

    std::string jsonFilePath = "./config/pass/json/pre_graph_assemble.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_TestAssign");
    npu::tile_fwk::PreGraphPass preGraphPass;
    preGraphPass.PreCheck(*func);
    preGraphPass.RunOnFunction(*func);
    preGraphPass.PostCheck(*func);

    int memoryMapSize = 1;
    std::set<int> tensorMagicWithColorSet;
    PrintGraphInfoPreGraph(func, memoryMapSize, tensorMagicWithColorSet);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 30;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 30 operations";
    EXPECT_NE(memoryMapSize, 0) << "All op memory size should not be 0";
    EXPECT_EQ(tensorMagicWithColorSet.size() > 0, true) << "There should be many tensor magic with color";
}

TEST_F(PreGraphTest, TestView) {
config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    std::vector<int> shape1{128, 128};
    std::vector<int> shape2{64, 64};
    std::vector<int> shape3{16, 256};

    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PreGraphTestStrategy", {
    {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
    {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
    {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
    {        "GenerateMoveOp_01",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {              "CubeProcess",              "CubeProcess",    PassType::TYPE_TILE_GRAPH},
    {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
    {         "NBufferMergePass",         "NBufferMergePass",    PassType::TYPE_TILE_GRAPH},
    {          "UpdateMemoryMap",          "UpdateMemoryMap",    PassType::TYPE_TILE_GRAPH},
    {        "GenerateMoveOp_02",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
    {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
    { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
    {        "L1CopyInReusePass",        "L1CopyInReusePass",    PassType::TYPE_TILE_GRAPH},
    });
    ConfigManager::Instance();

    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor in_tensor1(DT_FP32, shape1, "in_tensor1");
    Tensor out_tensor(DT_FP32, shape3, "out_tensor");

    FUNCTION("PreGraphFunction") {
        Program::GetInstance().GetTileShape().SetVecTileShapes({64, 64});
        auto a = View(in_tensor, shape2, {0,0});
        auto b = View(in_tensor1, shape2, {32,32});
        auto a0 = AddS(a, Element(DataType::DT_FP32, 0.0f));
        auto a1 = Reshape(a0, shape3);
        auto b0 = MulS(b, Element(DataType::DT_FP32, 0.1f));
        auto b1 = Reshape(b0, shape3);
        out_tensor = Add(a1, b1);
        originFunction = Program::GetInstance().GetCurrentFunction();
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }

    std::string jsonFilePath = "./config/pass/json/pre_graph_view.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_PreGraphFunction");
    npu::tile_fwk::PreGraphPass preGraphPass;
    preGraphPass.PreCheck(*func);
    preGraphPass.RunOnFunction(*func);
    preGraphPass.PostCheck(*func);

    int memoryMapSize = 1;
    std::set<int> tensorMagicWithColorSet;
    PrintGraphInfoPreGraph(func, memoryMapSize, tensorMagicWithColorSet);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 32;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 32 operations";
    EXPECT_NE(memoryMapSize, 0) << "All op memory size should not be 0";
    EXPECT_EQ(tensorMagicWithColorSet.size() > 0, true) << "There should be many tensor magic with color";
}

TEST_F(PreGraphTest, TestROWMAX_SINGLE) {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("PreGraphTestStrategy", {
    {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
    {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
    {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
    {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
    {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
    {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
    {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
    {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
    { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},

    });

    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    std::vector<int> vecTileShape = {128, 128};
    int b = 2; // 32
    int s = 1; // 1, optimize set_tile
    int h = 256;
    std::cout << "Test_deepseekAttention  b,s,h: " << b << ", " << s << ", " << h << std::endl;

    Tensor input = Tensor(DT_FP16, {b * s, h}, "input");
    Tensor res;

    Program::GetInstance().GetTileShape().SetCubeTileShapes({std::min(128, s), std::min(128, s)}, {256, 256}, {64, 64});
    Program::GetInstance().GetTileShape().SetVecTileShapes(vecTileShape[0], vecTileShape[1]); // for Assemble

    FUNCTION("A") {
        res = std::get<0>(Quant(input));
    }

    std::string jsonFilePath = "./config/pass/json/pre_graph_rowmax_single.json";
    bool dumpJsonFlag = false;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }

    // Call the pass
    Function* func = Program::GetInstance().GetFunctionByRawName("TENSOR_A");
    npu::tile_fwk::PreGraphPass preGraphPass;
    preGraphPass.PreCheck(*func);
    preGraphPass.RunOnFunction(*func);
    preGraphPass.PostCheck(*func);

    int memoryMapSize = 1;
    std::set<int> tensorMagicWithColorSet;
    PrintGraphInfoPreGraph(func, memoryMapSize, tensorMagicWithColorSet);

    // ================== Verify the effect of the Pass ==================
    auto updated_operations = func->Operations();
    int opSize = 21;

    EXPECT_EQ(updated_operations.size(), opSize) << "After the Pass, there should be 29 operations";
    EXPECT_EQ(tensorMagicWithColorSet.size() > 0, true) << "There should be many tensor magic with color";
}
