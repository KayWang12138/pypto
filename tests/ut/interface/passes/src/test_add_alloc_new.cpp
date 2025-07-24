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
 * \file test_add_alloc_new.cpp
 * \brief Unit test for AddAlloc pass.
 */

#include "gtest/gtest.h"
#include "operation/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "passes/execute_graph_pass/add_alloc.h"
#include "ut_json/ut_json_tool.h"
#include "interface/configs/config_manager.h"
#include <vector>

namespace npu {
namespace tile_fwk{
class AddAllocTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "AddAllocTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(AddAllocTest, TestAllocNode) {
    config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);

    // Define the shape of the Tensors
    std::vector<int> shape1{2, 1, 64};
    std::vector<int> shape2{2, 1, 1, 64};
    std::vector<int> shape3{1, 2, 1, 64};
    std::vector<int> shape4{1, 2, 64};

    // Initialize PassManager
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("AddAllocTestStrategy", {
    {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
    {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
    {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
    {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
    {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
    {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
    {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
    {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
    {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
    {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
    {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
    { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
    {             "PreGraphPass",             "PreGraphPass",    PassType::TYPE_TILE_GRAPH},
    {           "PadLocalBuffer",           "PadLocalBuffer",    PassType::TYPE_TILE_GRAPH},
    {       "SubgraphToFunction",       "SubgraphToFunction", PassType::TYPE_EXECUTE_GRAPH},
    {"SrcDstBufferMergePass","SrcDstBufferMergePass", PassType::TYPE_EXECUTE_GRAPH},

    });
    ConfigManager::Instance();

    // Create and configure the function
    Function* originFunction = nullptr;
    std::vector<int> originOpmagic;

    // Create Tensor
    Tensor in_tensor(DT_FP32, shape1, "in_tensor");
    Tensor out_tensor_4_A(DT_FP16, shape1, "out_tensor_4_A");
    Tensor out_tensor_4_B(DT_FP16, shape4, "out_tensor_4_B");

    FUNCTION("AddAllocFunction") {
        auto out_tensor_1 = Reshape(in_tensor, shape2);
        auto out_tensor_2_A = Transpose(out_tensor_1, {0, 1});
        auto out_tensor_2_B = Transpose(out_tensor_1, {0, 1});
        auto out_tensor_3_A = Reshape(out_tensor_2_A, shape1);
        auto out_tensor_3_B = Reshape(out_tensor_2_B, shape4);
        out_tensor_4_A = Cast(out_tensor_3_A, DT_FP16);
        out_tensor_4_B = Cast(out_tensor_3_B, DT_FP16);

        originFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_AddAllocFunction");
        ASSERT_NE(originFunction, nullptr) << "当前函数指针为空";
        auto operations = originFunction->Operations();
        for (const auto &op : operations) {
            originOpmagic.emplace_back(op.opmagic);
        }
    }
    std::string jsonFilePath = "./config/pass/json/add_alloc_new_alloc_node.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    // Call the pass
    Function* func = Program::GetInstance().GetCurrentFunction();
    (void)func;
}

TEST_F(AddAllocTest, TestFindTensorAllocMsg) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int> shape = {128, 128};
    auto shapeImme = OpImmediate::Specified(shape);
    std::vector<int> offset = {0, 0};

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 1;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);
    alloc1.UpdateSubgraphID(1);

    AddAllocPass addAlloc;
    addAlloc.AddAndCheckAlloc(function);
}
} // namespace tile_fwk
} // namespace npu