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
 * \file test_merge_src_dst_buffer.cpp
 * \brief Unit test for SrcDstBufferMergePass pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "passes/pass_registry.h"
#include "interface/configs/config_manager.h"
#include "ut_json/ut_json_tool.h"
#include "passes/execute_graph_pass/merge_src_dst_buffer.h"
#include "passes/pass_utils/pass_utils.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class MergeSrcDstBufferTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
    }
    void TearDown() override {}
};

TEST_F(MergeSrcDstBufferTest, Replaced) {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("SrcDstBufferMergeIncludePrePassStrategy", {
         {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
                                 {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
                                 {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
                                 {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
                                 {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
                                 {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
                                 {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
                                 {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
                                 {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
                                 {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
                                 {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
                                 {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
                                 {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
                                 {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
                                 {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
                                 {      "L1CopyInReusePass",       "L1CopyInReusePass",    PassType::TYPE_TILE_GRAPH},
                                 { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
                                 {             "PreGraphPass",             "PreGraphPass",    PassType::TYPE_TILE_GRAPH},
                                 {           "PadLocalBuffer",           "PadLocalBuffer",    PassType::TYPE_TILE_GRAPH},
                                 {       "SubgraphToFunction",       "SubgraphToFunction", PassType::TYPE_EXECUTE_GRAPH},

    });
    config::SetHostConfig(KEY_STRATEGY, "SrcDstBufferMergeIncludePrePassStrategy");
    config::SetPlatformConfig("TEST_IS_TIG", true);
    config::SetPassConfig("SrcDstBufferMergeIncludePrePassStrategy", "SrcDstBufferMergePass", "PRINT_FUNCTION", true);
    config::SetPassConfig("SrcDstBufferMergeIncludePrePassStrategy", "SrcDstBufferMergePass", "DUMP_FUNCTION_GRAPH_BEFORE_PASS", true);
    config::SetPassConfig("SrcDstBufferMergeIncludePrePassStrategy", "SrcDstBufferMergePass", "DUMP_FUNCTION_GRAPH_AFTER_PASS", true);
    constexpr int32_t tilex = 8;
    constexpr int32_t tiley = 8;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tilex, tiley);

    std::vector<int> shape = {8, 16};
    Tensor input(DT_FP32, shape, "input");
    Tensor output(DT_FP32, shape, "output");
    FUNCTION("AddFunction") {
        output = Add(input, input);
    }
    std::string jsonFilePath = "./config/pass/json/merge_src_dst_buffer_replaced.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);

    Function* func = Program::GetInstance().GetCurrentFunction();

    Program testProgram(HostMachineMode::SERVER);
    SrcDstBufferMergePass srcDstBufferMergePass;
    srcDstBufferMergePass.PreCheck(*func);
    srcDstBufferMergePass.RunOnFunction(*func);
    srcDstBufferMergePass.PostCheck(*func);

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_AddFunction");
    for (const auto &op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_ADD) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            ASSERT_EQ(outputTensor->GetDataSize(), inputTensor->GetDataSize());
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, NoReplaced) {
    PassManager &passManager = PassManager::Instance();
    passManager.RegisterStrategy("SrcDstBufferMergeIncludePrePassStrategy", {
         {   "RemoveRedundentReshape",   "RemoveRedundentReshape",  PassType::TYPE_TENSOR_GRAPH},
                                 {           "ExpandFunction",           "ExpandFunction",  PassType::TYPE_TENSOR_GRAPH},
                                 {            "DuplicateView",            "DuplicateView",    PassType::TYPE_TILE_GRAPH},
                                 {        "MergeViewAssemble",        "MergeViewAssemble",    PassType::TYPE_TILE_GRAPH},
                                 {         "AssignMemoryType",         "AssignMemoryType",    PassType::TYPE_TILE_GRAPH},
                                 {          "InsertConvertOp",          "InsertConvertOp",    PassType::TYPE_TILE_GRAPH},
                                 {   "SplitLargeFanoutTensor",   "SplitLargeFanoutTensor",    PassType::TYPE_TILE_GRAPH},
                                 {       "SplitReshapeOpPVC2",       "SplitReshapeOpPVC2",    PassType::TYPE_TILE_GRAPH},
                                 {        "RemoveRedundentOp",        "RemoveRedundentOp",    PassType::TYPE_TILE_GRAPH},
                                 {           "GenerateMoveOp",           "GenerateMoveOp",    PassType::TYPE_TILE_GRAPH},
                                 {             "GraphInitPass",           "GraphInitPass",    PassType::TYPE_TILE_GRAPH},
                                 {           "PartitionVCPass",         "PartitionVCPass",    PassType::TYPE_TILE_GRAPH},
                                 {        "GraphPartitionPass",        "GraphPartitionPass",    PassType::TYPE_TILE_GRAPH},
                                 {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
                                 {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
                                 {      "L1CopyInReusePass",       "L1CopyInReusePass",    PassType::TYPE_TILE_GRAPH},
                                 { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
                                 {             "PreGraphPass",             "PreGraphPass",    PassType::TYPE_TILE_GRAPH},
                                 {           "PadLocalBuffer",           "PadLocalBuffer",    PassType::TYPE_TILE_GRAPH},
                                 {       "SubgraphToFunction",       "SubgraphToFunction", PassType::TYPE_EXECUTE_GRAPH},
                                 {"SrcDstBufferMergePass","SrcDstBufferMergePass", PassType::TYPE_EXECUTE_GRAPH},
    });
    config::SetHostConfig(KEY_STRATEGY, "SrcDstBufferMergeIncludePrePassStrategy");
    config::SetPlatformConfig("TEST_IS_TIG", true);
    constexpr int32_t tilex = 1;
    constexpr int32_t tiley = 8;
    Program::GetInstance().GetTileShape().SetVecTileShapes(tilex, tiley);

    std::vector<int> shape = {1, 8};
    std::vector<int> shape2 = {8, 1};
    Tensor input1(DT_FP32, shape, "input1");
    Tensor input2(DT_FP32, shape2, "input2");
    Tensor output(DT_FP32, shape2, "output");
    FUNCTION("ReshapeFunction") {
        Tensor afterReshape = Reshape(input1, shape2);
        output = Add(afterReshape, input2);
    }

    Function* currentFunction = Program::GetInstance().GetFunctionByRawName("TENSOR_ReshapeFunction");
    for (const auto &op : currentFunction->Operations()) {
        if (op.GetOpcode() == Opcode::OP_RESHAPE) {
            auto outputTensor = op.GetOOperands()[0];
            auto inputTensor = op.GetIOperands()[0];
            ASSERT_NE(outputTensor->memorymap[0].memId, inputTensor->memorymap[0].memId);
            break;
        }
    }
}

TEST_F(MergeSrcDstBufferTest, AppointInplace) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int> shape = {128, 128};
    auto shapeImme = OpImmediate::Specified(shape);
    std::vector<int> offset = {0, 0};

    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor1->subGraphID = 0;
    tensor1->memorymap[0].memId = 1;

    std::shared_ptr<LogicalTensor> tensor2 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor2->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor2->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor2->subGraphID = 0;
    tensor2->memorymap[0].memId = 2;

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 3;

    std::shared_ptr<LogicalTensor> tensor4 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor4->SetMemoryTypeOriginal(MEM_UB);
    tensor4->SetMemoryTypeToBe(MEM_UB);
    tensor4->subGraphID = 0;
    tensor4->memorymap[0].memId = 4;

    std::shared_ptr<LogicalTensor> tensor5 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor5->SetMemoryTypeOriginal(MEM_UB);
    tensor5->SetMemoryTypeToBe(MEM_UB);
    tensor5->subGraphID = 0;
    tensor5->memorymap[0].memId = 5;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);
    alloc1.UpdateSubgraphID(0);
    auto &copyin1 =
        function.AddOperation(Opcode::OP_COPY_IN, std::vector<std::shared_ptr<LogicalTensor>>({tensor1}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    copyin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyin1.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc1, copyin1);

    auto &alloc2 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor4}));
    alloc2.UpdateLatency(1);
    alloc2.UpdateSubgraphID(0);
    auto &copyin2 =
        function.AddOperation(Opcode::OP_COPY_IN, std::vector<std::shared_ptr<LogicalTensor>>({tensor2}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor4}));
    copyin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyin2.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc2, copyin2);

    auto &alloc3 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor5}));
    alloc3.UpdateLatency(1);
    alloc3.UpdateSubgraphID(0);
    auto &add1 =
        function.AddOperation(Opcode::OP_ADD, std::vector<std::shared_ptr<LogicalTensor>>({tensor3, tensor4}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor5}));
    add1.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc3, add1);
    add1.SetAttr(OpAttributeKey::inplaceIdx, 0);

    SrcDstBufferMergePVC2 srcDstMerge;
    Function func(Program::GetInstance(), "", "", nullptr);
    Function func1(Program::GetInstance(), "", "", nullptr);
    Function *rootFunc = &func1;
    rootFunc->programs_.insert(std::pair<uint64_t, Function*>(1, &function));
    func.rootFunc_ = rootFunc;
    srcDstMerge.Run(func);
}
