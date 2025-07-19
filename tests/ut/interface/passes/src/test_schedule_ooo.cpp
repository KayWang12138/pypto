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
 * \file test_schedule_ooo.cpp
 * \brief Unit test for OoOSchedulePass.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/execute_graph_pass/schedule_ooo.h"
#include "models/deepseek/deepseek_mla.h"

namespace npu::tile_fwk {
class ScheduleOoOTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ScheduleOoOTest, TestSpillCopyIn) {
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

    std::shared_ptr<LogicalTensor> tensor6 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor6->SetMemoryTypeOriginal(MEM_UB);
    tensor6->SetMemoryTypeToBe(MEM_UB);
    tensor6->subGraphID = 0;
    tensor6->memorymap[0].memId = 6;

    std::shared_ptr<LogicalTensor> tensor7 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor7->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor7->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor7->subGraphID = 0;
    tensor7->memorymap[0].memId = 7;

    std::shared_ptr<LogicalTensor> tensor8 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor8->SetMemoryTypeOriginal(MEM_UB);
    tensor8->SetMemoryTypeToBe(MEM_UB);
    tensor8->subGraphID = 0;
    tensor8->memorymap[0].memId = 8;

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

    auto &alloc4 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor6}));
    alloc4.UpdateLatency(1);
    alloc4.UpdateSubgraphID(0);
    auto &add2 =
        function.AddOperation(Opcode::OP_ADD, std::vector<std::shared_ptr<LogicalTensor>>({tensor3, tensor5}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor6}));
    add2.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc4, add2);

    auto &alloc5 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor8}));
    alloc5.UpdateLatency(1);
    alloc5.UpdateSubgraphID(0);
    auto &add3 =
        function.AddOperation(Opcode::OP_ADD, std::vector<std::shared_ptr<LogicalTensor>>({tensor6, tensor4}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor8}));
    add3.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc5, add3);

    auto &copyout =
        function.AddOperation(Opcode::OP_COPY_OUT, std::vector<std::shared_ptr<LogicalTensor>>({tensor8}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor7}));
    copyout.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB,
        OpImmediate::Specified(offset), shapeImme, shapeImme, std::vector<npu::tile_fwk::OpImmediate>()));
    copyout.UpdateSubgraphID(0);

    std::vector<Operation *> newOperations;
    for (auto& op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
        } else {
            newOperations.push_back(op);
        }
    }
    function.ScheduleBy(newOperations);

    std::vector<Operation *> scheduleOpList;
    OoOScheduler oooSchedule;
    oooSchedule.Schedule(function, function.Operations().DuplicatedOpList(), scheduleOpList);
}

TEST_F(ScheduleOoOTest, TestSpill) {
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

    std::shared_ptr<LogicalTensor> tensor6 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor6->SetMemoryTypeOriginal(MEM_UB);
    tensor6->SetMemoryTypeToBe(MEM_UB);
    tensor6->subGraphID = 0;
    tensor6->memorymap[0].memId = 6;

    std::shared_ptr<LogicalTensor> tensor7 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor7->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor7->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor7->subGraphID = 0;
    tensor7->memorymap[0].memId = 7;

    std::shared_ptr<LogicalTensor> tensor8 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor8->SetMemoryTypeOriginal(MEM_UB);
    tensor8->SetMemoryTypeToBe(MEM_UB);
    tensor8->subGraphID = 0;
    tensor8->memorymap[0].memId = 8;

    std::shared_ptr<LogicalTensor> tensor9 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor9->SetMemoryTypeOriginal(MEM_UB);
    tensor9->SetMemoryTypeToBe(MEM_UB);
    tensor9->subGraphID = 0;
    tensor9->memorymap[0].memId = 9;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);
    alloc1.UpdateSubgraphID(0);
    auto &copyin1 =
        function.AddOperation(Opcode::OP_COPY_IN, std::vector<std::shared_ptr<LogicalTensor>>({tensor1}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    copyin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme));
    copyin1.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc1, copyin1);

    auto &alloc2 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor4}));
    alloc2.UpdateLatency(1);
    alloc2.UpdateSubgraphID(0);
    auto &copyin2 =
        function.AddOperation(Opcode::OP_COPY_IN, std::vector<std::shared_ptr<LogicalTensor>>({tensor2}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor4}));
    copyin2.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme));
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

    auto &alloc4 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor6}));
    alloc4.UpdateLatency(1);
    alloc4.UpdateSubgraphID(0);
    auto &add2 =
        function.AddOperation(Opcode::OP_ADD, std::vector<std::shared_ptr<LogicalTensor>>({tensor3, tensor4}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor6}));
    add2.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc4, add2);

    auto &alloc5 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor8}));
    alloc5.UpdateLatency(1);
    alloc5.UpdateSubgraphID(0);
    auto &add3 =
        function.AddOperation(Opcode::OP_ADD, std::vector<std::shared_ptr<LogicalTensor>>({tensor6, tensor4}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor8}));
    add3.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc5, add3);

    auto &alloc6 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor9}));
    alloc6.UpdateLatency(1);
    alloc6.UpdateSubgraphID(0);
    auto &add4 =
        function.AddOperation(Opcode::OP_ADD, std::vector<std::shared_ptr<LogicalTensor>>({tensor8, tensor5}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor9}));
    add4.UpdateSubgraphID(0);
    FunctionUtils::AddControlEdge(alloc6, add4);

    auto &copyout =
        function.AddOperation(Opcode::OP_COPY_OUT, std::vector<std::shared_ptr<LogicalTensor>>({tensor9}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor7}));
    copyout.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB,
        OpImmediate::Specified(offset), shapeImme, shapeImme));
    copyout.UpdateSubgraphID(0);
    
    std::vector<Operation *> newOperations;
    for (auto& op : function.Operations().DuplicatedOpList()) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
        } else {
            newOperations.push_back(op);
        }
    }
    function.ScheduleBy(newOperations);

    std::vector<Operation *> scheduleOpList;
    OoOScheduler oooSchedule;
    oooSchedule.Schedule(function, function.Operations().DuplicatedOpList(), scheduleOpList);
}

TEST_F(ScheduleOoOTest, TestHealthReport) {
    config::SetHostConfig(KEY_STRATEGY, "PVC2_OOO");
    int m = 128;
    int k = 8192;
    int n = 512;

    Tensor inputA = Tensor(DT_INT8, {m, k}, "inputA");
    Tensor inputW = Tensor(DT_INT8, {k, n}, "inputW");
    Tensor res;

    Program::GetInstance().GetTileShape().SetCubeTileShapes({16, 16}, {128, 128}, {128, 128});
    Program::GetInstance().GetConfig().Set<int>(L1_REUSE, 4);
    Program::GetInstance().GetConfig().Set<int>(COPYIN_THRESHOLD, 32*1024*1024);
    config::SetPassDefaultConfig(KEY_HEALTH_CHECK, true);
    FUNCTION("A") {
        res = npu::tile_fwk::Matrix::Matmul<false, false>(DT_INT32, inputA, inputW);
    }
}

TEST_F(ScheduleOoOTest, TestOpNullptr) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;
    std::vector<Operation *> newScheduleOpList;
    OoOScheduler oooSchedule;
    Operation *op = nullptr;
    scheduleOpList.push_back(op);
    oooSchedule.Schedule(function, scheduleOpList, newScheduleOpList);
}

TEST_F(ScheduleOoOTest, TestDelBufCount) {
    OoOScheduler oooSchedule;
    oooSchedule.DelBufRefCount(-1);
}

TEST_F(ScheduleOoOTest, TestDelBufCount_1) {
    OoOScheduler oooSchedule;
    oooSchedule.bufRefCount[1] = -1;
    oooSchedule.DelBufRefCount(1);
}

TEST_F(ScheduleOoOTest, TestGenBufferSpill) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;

    std::vector<int> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 3;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);
    alloc1.UpdateSubgraphID(0);

    auto allocIssue = std::make_shared<IssueEntry>(&alloc1, 1);

    OoOScheduler oooSchedule;
    oooSchedule.GenBufferSpill(function, allocIssue, MemoryType::MEM_UB, scheduleOpList);
}

TEST_F(ScheduleOoOTest, TestUpdateReloadIssueInfo) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;

    std::vector<int> shape = {128, 128};
    auto shapeImme = OpImmediate::Specified(shape);
    std::vector<int> offset = {0, 0};
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor1->subGraphID = 0;
    tensor1->memorymap[0].memId = 1;

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 3;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);
    alloc1.UpdateSubgraphID(0);
    auto &copyin1 =
        function.AddOperation(Opcode::OP_COPY_IN, std::vector<std::shared_ptr<LogicalTensor>>({tensor1}),
                              std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    copyin1.SetOpAttribute(std::make_shared<CopyOpAttribute>(
            OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme));
    copyin1.UpdateSubgraphID(0);

    auto allocIssue = std::make_shared<IssueEntry>(&alloc1, 1);
    auto copyinIssue = std::make_shared<IssueEntry>(&copyin1, 2);

    OoOScheduler oooSchedule;
    oooSchedule.UpdateReloadIssueInfo(allocIssue, copyinIssue, copyinIssue, -1, -1);
}

TEST_F(ScheduleOoOTest, TestUpdateTensorAttr_DDR) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor1->subGraphID = 0;
    tensor1->memorymap[0].memId = 1;

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 3;

    OoOScheduler oooSchedule;
    oooSchedule.UpdateTensorAttr(tensor1, MemoryType::MEM_DEVICE_DDR, tensor3, -1);
}

TEST_F(ScheduleOoOTest, TestUpdateTensorAttr_UB) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor1->subGraphID = 0;
    tensor1->memorymap[0].memId = 1;

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 3;

    OoOScheduler oooSchedule;
    oooSchedule.UpdateTensorAttr(tensor1, MemoryType::MEM_UB, tensor3, -1);
}

TEST_F(ScheduleOoOTest, TestGetSpillTensor) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;

    std::vector<int> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 3;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);
    alloc1.UpdateSubgraphID(0);

    LogicalTensorPtr tensor = nullptr;
    auto allocIssue = std::make_shared<IssueEntry>(&alloc1, 1);

    OoOScheduler oooSchedule;
    oooSchedule.GetSpillTensor(allocIssue, 1, tensor);
}

TEST_F(ScheduleOoOTest, TestCheckAllocIssue) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;

    std::vector<int> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 3;

    std::shared_ptr<LogicalTensor> tensor2 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->subGraphID = 0;
    tensor3->memorymap[0].memId = 1;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3, tensor2}));
    alloc1.UpdateLatency(1);
    alloc1.UpdateSubgraphID(0);

    OoOScheduler oooSchedule;
    oooSchedule.Init(function.Operations().DuplicatedOpList());
}

TEST_F(ScheduleOoOTest, TestGetBufTimes) {
    Function function(Program::GetInstance(), "", "", nullptr);

    size_t bufNextUseTime;
    size_t bufLastUseTime;
    size_t bufLastWriteTime;

    OoOScheduler oooSchedule;
    oooSchedule.GetBufTimes(-1, bufNextUseTime, bufLastUseTime, bufLastWriteTime);
}
} // namespace npu::tile_fwk
