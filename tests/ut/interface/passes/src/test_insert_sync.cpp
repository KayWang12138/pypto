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
 * \file test_insert_sync.cpp
 * \brief Unit test for InsertSyncPass.
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/execute_graph_pass/insert_sync.h"
#include "ut_json/ut_json_tool.h"

namespace npu::tile_fwk {
class TestInsertSync : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        Program::GetInstance().GetConfig().Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetHostConfig(KEY_STRATEGY, "InsertSyncTestStrategy");
        config::SetPlatformConfig("ENABLE_COST_MODEL", false);
        PassManager &passManager = PassManager::Instance();
        passManager.RegisterStrategy("InsertSyncTestStrategy", {
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
            {         "NBufferMergePass",         "NBufferMergePass",    PassType::TYPE_TILE_GRAPH},
            {          "UpdateMemoryMap",          "UpdateMemoryMap",    PassType::TYPE_TILE_GRAPH},
            {   "SplitLargeLocalRawPass",   "SplitLargeLocalRawPass",    PassType::TYPE_TILE_GRAPH},
            {         "InsertCopyOpPass",         "InsertCopyOpPass",    PassType::TYPE_TILE_GRAPH},
            {      "L1CopyInReusePass",       "L1CopyInReusePass",       PassType::TYPE_TILE_GRAPH},
            { "CommonOperationEliminate", "CommonOperationEliminate",    PassType::TYPE_TILE_GRAPH},
            {             "PreGraphPass",             "PreGraphPass",    PassType::TYPE_TILE_GRAPH},
            {           "PadLocalBuffer",           "PadLocalBuffer",    PassType::TYPE_TILE_GRAPH},
            {       "SubgraphToFunction",       "SubgraphToFunction", PassType::TYPE_EXECUTE_GRAPH},
            {    "SrcDstBufferMergePass",    "SrcDstBufferMergePass", PassType::TYPE_EXECUTE_GRAPH},
            {             "AddAllocPass",             "AddAllocPass", PassType::TYPE_EXECUTE_GRAPH},
            {          "OoOSchedulePass",          "OoOSchedulePass", PassType::TYPE_EXECUTE_GRAPH},
            {          "RemoveAllocPass",          "RemoveAllocPass", PassType::TYPE_EXECUTE_GRAPH}
        });
    }
    void TearDown() override {}
};

TEST_F(TestInsertSync, TestMainSchedule) {
    std::vector<int> shape{64, 64};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");
    constexpr int TILE_SHAPE = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(TILE_SHAPE,TILE_SHAPE);
    FUNCTION("A") {
        c = Div(a, b);
    }

    std::string jsonFilePath = "./config/pass/json/insert_sync_main_schedule.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);
    Function* currentFunction = Program::GetInstance().GetCurrentFunction();
    // load json error
    Program testProgram(HostMachineMode::SERVER);
    InsertSyncPass insertSync;
    insertSync.RunOnFunction(*currentFunction);
    // PostCheck Test
    EXPECT_EQ(insertSync.PostCheck(*currentFunction), SUCCESS);

    // SrcDst Match Test
    for (auto &program : currentFunction->rootFunc_->programs_) {
        auto opLogPtr = program.second->GetProgramOp();
        auto syncSrcCnt = 0;
        auto syncDstCnt = 0;
        for (auto &op : opLogPtr) {
            if (op->GetOpcode() == Opcode::OP_SYNC_SRC) {
                syncSrcCnt++;
            } else if (op->GetOpcode() == Opcode::OP_SYNC_DST) {
                syncDstCnt++;
            }
        }
        EXPECT_EQ(syncSrcCnt == syncDstCnt, true);
    }
}

TEST_F(TestInsertSync, TestInnerFunction) {
    std::vector<int> shape{64, 64};
    Tensor a(DT_FP32, shape, "a");
    Tensor b(DT_FP32, shape, "b");
    Tensor c(DT_FP32, shape, "c");
    constexpr int TILE_SHAPE = 32;
    Program::GetInstance().GetTileShape().SetVecTileShapes(TILE_SHAPE,TILE_SHAPE);
    FUNCTION("A") {
        c = Div(a, b);
    }

    std::string jsonFilePath = "./config/pass/json/insert_sync_inner_function.json";
    bool dumpJsonFlag = true;
    if (dumpJsonFlag) {
        auto programJson = Program::GetInstance().DumpJson();
        DumpJsonFile(programJson, jsonFilePath);
    }
    Json readData = LoadJsonFile(jsonFilePath);
    Program::GetInstance().LoadJson(readData);
    Function* currentFunction = Program::GetInstance().GetCurrentFunction();
    for (auto &program : currentFunction->rootFunc_->programs_) {
        PipeSync ps;
        std::vector<Operation *> syncedOpLogPtr;
        std::vector<Operation *> operationLogWithSync;
        ps.InsertSync(*program.second, syncedOpLogPtr);
        ps.PhaseKernelProcess(*program.second, syncedOpLogPtr, operationLogWithSync);

        // OpDep Dump Test
        std::string dumpString = ps.depOps_.back().Dump(syncedOpLogPtr);
        ALOG_DEBUG(dumpString);

        // AddOpDep Test
        auto  &setOp = ps.depOps_[0];
        auto  &waitOp = ps.depOps_[1];
        setOp.setPipe = std::vector<size_t> {2, 3, 4, 5};
        ps.depOps_[2].selfPipeCore = ps.depOps_[waitOp.idx].selfPipeCore;
        ps.AddOpDep(setOp, waitOp);
        EXPECT_EQ(setOp.setPipe.back(), waitOp.idx);
        EXPECT_EQ(waitOp.waitPipe.back(), setOp.idx);
    }
}

TEST_F(TestInsertSync, TestEnableDebug) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestAddParams", "TestAddParams", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());

    // Prepare the graph
    std::vector<int> shape = {8, 16};
    auto shapeImme = OpImmediate::Specified(shape);
    auto incast1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto incast2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto ubTensor3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    auto outCast = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);

    auto &copy_op1 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast1}, {ubTensor1});
    (void) copy_op1;
    auto &copy_op2 = currFunctionPtr->AddOperation(Opcode::OP_COPY_IN, {incast2}, {ubTensor2});
    (void) copy_op2;
    auto& add_op = currFunctionPtr->AddOperation(Opcode::OP_ADD, {ubTensor1, ubTensor2}, {ubTensor3});
    (void) add_op;
    auto& copy_out_op = currFunctionPtr->AddOperation(Opcode::OP_COPY_OUT, {ubTensor3}, {outCast});
    (void) copy_out_op;
    currFunctionPtr->inCasts_.push_back(incast1);
    currFunctionPtr->inCasts_.push_back(incast2);
    currFunctionPtr->outCasts_.push_back(outCast);
    InsertSyncPass syncPass;
    syncPass.SetEnableDebug(true);
    syncPass.RunOnFunction(*rootFuncPtr);
    EXPECT_TRUE(true);
}
} // namespace acend
