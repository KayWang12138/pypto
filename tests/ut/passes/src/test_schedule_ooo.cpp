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
 * \brief Unit test for OoOSchedule.
 */

#include <gtest/gtest.h>
#include <vector>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/block_graph_pass/schedule_ooo/schedule_ooo.h"
#include "operator/models/deepseek/deepseek_mla.h"
#include "computational_graph_builder.h"
#include "passes/pass_config/pass_config_manager.h"

namespace npu::tile_fwk {
class ScheduleOoOTest : public ::testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        PassConfigManager::Instance().Initialize(DPlatform::ASCEND_910B2);
    }
    void TearDown() override {}
};

IssueEntryPtr GetIssueEntry(const std::string& name, ComputationalGraphBuilder subGraph, OoOScheduler ooOScheduler) {
    EXPECT_NE(subGraph.GetOp(name), nullptr);
    Operation *op = subGraph.GetOp(name);
    for (auto &issue : ooOScheduler.issueEntries) {
        if (&(issue->tileOp) == op) {
            return issue;
        }
    }
    return nullptr;
}

void SetTensorAttr(LogicalTensorPtr tensor, MemoryType memType, int memId) {
    tensor->SetMemoryTypeOriginal(memType);
    tensor->SetMemoryTypeToBe(memType);
    tensor->memoryrange.memId = memId;
    tensor->UpdateDynValidShape({SymbolicScalar("S0"), SymbolicScalar("S1")});
}

void SetAllocAttr(Operation &alloc, int latency) {
    alloc.UpdateLatency(latency);
}

LogicalTensorPtr CreateTensor(Function &currFunction, DataType dateType, std::vector<int64_t> shape, MemoryType memType, int memId) {
    LogicalTensorPtr tensor = std::make_shared<LogicalTensor>(currFunction, dateType, shape);
    SetTensorAttr(tensor, memType, memId);
    return tensor;
}

Operation &CreateAllocOp(Function &currFunction, LogicalTensorPtr tensor, int latency) {
    Operation &alloc = currFunction.AddOperation(Opcode::OP_UB_ALLOC, {}, LogicalTensors({tensor}));
    SetAllocAttr(alloc, latency);
    return alloc;
}

Operation &CreateCopyOp(Function &currFunction, Opcode opcode, LogicalTensorPtr inTensor, LogicalTensorPtr outTensor, std::vector<int64_t> shape) {
    std::vector<int64_t> offset = {0, 0};
    auto &copy = currFunction.AddOperation(opcode, LogicalTensors({inTensor}), LogicalTensors({outTensor}));
    auto shapeImme = OpImmediate::Specified(shape);
    if (opcode == Opcode::OP_COPY_IN) {
        copy.SetOpAttribute(std::make_shared<CopyOpAttribute>(OpImmediate::Specified(offset), MEM_UB, shapeImme, shapeImme));
    }
    if (opcode == Opcode::OP_COPY_OUT) {
        copy.SetOpAttribute(std::make_shared<CopyOpAttribute>(MEM_UB, OpImmediate::Specified(offset), shapeImme, shapeImme));
    }
    return copy;
}

Operation &CreateAddOp(Function &currFunction, LogicalTensorPtr inTensor1, LogicalTensorPtr inTensor2, LogicalTensorPtr outTensor) {
    auto &add = currFunction.AddOperation(Opcode::OP_ADD, LogicalTensors({inTensor1, inTensor2}), LogicalTensors({outTensor}));
    return add;
}

void ReorderOperations(Function &function) {
    auto opList = function.Operations().DuplicatedOpList();
    std::vector<Operation *> newOperations;
    for (auto &op : opList) {
        if (op->GetOpcodeStr().find("ALLOC") != std::string::npos) {
            newOperations.insert(newOperations.begin(), op);
        } else {
            newOperations.push_back(op);
        }
    }
    function.ScheduleBy(newOperations);
}

TEST_F(ScheduleOoOTest, TestMainScheduleOoO) {
    auto rootFuncPtr = std::make_shared<Function>(Program::GetInstance(), "TestParams", "TestParams", nullptr);
    rootFuncPtr->rootFunc_ = rootFuncPtr.get();
    auto currFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "TestOOO", "TestOOO", rootFuncPtr.get());
    currFunctionPtr->paramConfigs_.OoOPreScheduleMethod = "LayerBasedDFS";
    auto emptyOpFunctionPtr = std::make_shared<Function>(Program::GetInstance(), "", "", rootFuncPtr.get());
    EXPECT_TRUE(currFunctionPtr != nullptr);
    EXPECT_TRUE(emptyOpFunctionPtr != nullptr);
    rootFuncPtr->rootFunc_->programs_.emplace(currFunctionPtr->GetFuncMagic(), currFunctionPtr.get());
    rootFuncPtr->rootFunc_->programs_.emplace(emptyOpFunctionPtr->GetFuncMagic(), emptyOpFunctionPtr.get());
    std::vector<int64_t> shape = {128, 128};
    auto shapeImme = OpImmediate::Specified(shape);

    auto tensor1 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_DEVICE_DDR, 1);
    auto tensor2 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_DEVICE_DDR, 2);
    auto tensor3 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 3);
    auto tensor4 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 4);
    auto tensor5 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 5);
    auto tensor6 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 6);
    auto tensor7 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_DEVICE_DDR, 7);
    auto tensor8 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 8);
    auto tensor9 = CreateTensor(*currFunctionPtr, DataType::DT_FP32, shape, MEM_UB, 9);
    auto &alloc1 = CreateAllocOp(*currFunctionPtr, tensor3, 1);
    auto &alloc2 = CreateAllocOp(*currFunctionPtr, tensor4, 1);
    auto &alloc3 = CreateAllocOp(*currFunctionPtr, tensor5, 1);
    auto &alloc4 = CreateAllocOp(*currFunctionPtr, tensor6, 1);
    auto &alloc5 = CreateAllocOp(*currFunctionPtr, tensor8, 1);
    auto &alloc6 = CreateAllocOp(*currFunctionPtr, tensor9, 1);
    auto &copyin1 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor1, tensor3, shape);
    auto &copyin2 = CreateCopyOp(*currFunctionPtr, Opcode::OP_COPY_IN, tensor2, tensor4, shape);
    auto &add1 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor5);
    auto &add2 = CreateAddOp(*currFunctionPtr, tensor3, tensor4, tensor6);
    auto &add3 = CreateAddOp(*currFunctionPtr, tensor6, tensor4, tensor8);
    auto &add4 = CreateAddOp(*currFunctionPtr, tensor8, tensor5, tensor9);
    (void)alloc1, (void)alloc2, (void)alloc3, (void)alloc4, (void)alloc5, (void)alloc6,
    (void)copyin1, (void)copyin2, (void)add1, (void)add2, (void)add3, (void)add4;
    for (auto &program : rootFuncPtr->rootFunc_->programs_) {
        ReorderOperations(*(program.second));
    }
    OoOSchedule oooSchedule;
    EXPECT_EQ(oooSchedule.PreCheck(*rootFuncPtr), SUCCESS);
    oooSchedule.RunOnFunction(*rootFuncPtr);
    EXPECT_EQ(oooSchedule.PostCheck(*rootFuncPtr), SUCCESS);
}

static bool CheckExists(std::unordered_set<int> &issueList, Operation *op,
    std::unordered_map<int, IssueEntryPtr> issueEntryMap) {
    for (auto &issueId : issueList) {
        auto issue = issueEntryMap[issueId];
        if (&(issue->tileOp) == op) {
            return true;
        }
    }
    return false;
}

TEST_F(ScheduleOoOTest, TestDependencies) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t3"}, {"t2"}, {"t4", "t5"}, {"t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t2"}, {"t4"}, {"t5", "t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "RowMax1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_NE(GetIssueEntry("RowMax1", subGraph, ooOScheduler), nullptr);
    IssueEntryPtr issue = GetIssueEntry("RowMax1", subGraph, ooOScheduler);
    EXPECT_EQ(issue->predecessors.size(), 3);
    EXPECT_TRUE(CheckExists(issue->predecessors, subGraph.GetOp("Alloc4"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(issue->successors.size(), 2);
    EXPECT_TRUE(CheckExists(issue->successors, subGraph.GetOp("Add1"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesView) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t2"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "View1", "View2", "View3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);
    for (size_t i = 1; i < tensorNames.size(); i++) {
        EXPECT_NE(subGraph.GetTensor(tensorNames[i]), nullptr);
        std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor(tensorNames[i]);
        tensor->memoryrange.memId =
            subGraph.GetTensor("t2")->memoryrange.memId;
    }

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    IssueEntryPtr copyin = GetIssueEntry("Copyin1", subGraph, ooOScheduler);
    EXPECT_NE(copyin, nullptr);
    IssueEntryPtr view3 = GetIssueEntry("View3", subGraph, ooOScheduler);
    EXPECT_NE(view3, nullptr);
    IssueEntryPtr view2 = GetIssueEntry("View2", subGraph, ooOScheduler);
    EXPECT_NE(view2, nullptr);
    IssueEntryPtr view1 = GetIssueEntry("View1", subGraph, ooOScheduler);
    EXPECT_NE(copyin, nullptr);
    EXPECT_TRUE(CheckExists(copyin->predecessors, subGraph.GetOp("Alloc1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(copyin->successors, subGraph.GetOp("View3"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(view3->predecessors, subGraph.GetOp("Copyin1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(view3->successors, subGraph.GetOp("View2"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(view2->predecessors, subGraph.GetOp("View3"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(view2->successors, subGraph.GetOp("View1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(view1->predecessors, subGraph.GetOp("View2"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(res, SUCCESS);
}


TEST_F(ScheduleOoOTest, TestDependenciesAssemble) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_SUB, Opcode::OP_SUB, Opcode::OP_SUB, Opcode::OP_ASSEMBLE,
        Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t4"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t7"}, {"t7"}, {"t8"}};
    std::vector<std::string> opNames{"Alloc1", "Sub1", "Sub2", "Sub3", "Assemble1", "Assemble2", "Assemble3", "Mul1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    for (size_t i = 3; i < tensorNames.size() - 1; i++) {
        EXPECT_NE(subGraph.GetTensor(tensorNames[i]), nullptr);
        std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor(tensorNames[i]);
        tensor->memoryrange.memId =
            subGraph.GetTensor("t4")->memoryrange.memId;
    }

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    IssueEntryPtr alloc = GetIssueEntry("Alloc1", subGraph, ooOScheduler);
    EXPECT_NE(alloc, nullptr);
    IssueEntryPtr sub = GetIssueEntry("Sub1", subGraph, ooOScheduler);
    EXPECT_NE(sub, nullptr);
    EXPECT_TRUE(CheckExists(alloc->successors, subGraph.GetOp("Sub3"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(sub->predecessors, subGraph.GetOp("Sub2"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(sub->successors, subGraph.GetOp("Assemble3"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesInplace) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId =
        subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    IssueEntryPtr add = GetIssueEntry("Add1", subGraph, ooOScheduler);
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(CheckExists(add->successors, subGraph.GetOp("Copyout1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(add->predecessors, subGraph.GetOp("Copyin1"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestDependenciesFailed) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN, Opcode::OP_UB_ALLOC, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {}, {"t2"}, {"t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"Copyin1", "Alloc1", "Add1", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {16, 16}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->memoryrange.memId =
        subGraph.GetTensor("t2")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    std::rotate(ooOScheduler.issueEntries.begin(), ooOScheduler.issueEntries.begin() + 1, ooOScheduler.issueEntries.end());
    ooOScheduler.InitDependencies();
    IssueEntryPtr add = GetIssueEntry("Add1", subGraph, ooOScheduler);
    EXPECT_NE(add, nullptr);
    EXPECT_TRUE(CheckExists(add->successors, subGraph.GetOp("Alloc1"), ooOScheduler.issueEntryMap));
    EXPECT_TRUE(CheckExists(add->predecessors, subGraph.GetOp("Copyin1"), ooOScheduler.issueEntryMap));
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillCopyIn) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.issueEntries[9]->tileOp.GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.issueEntries[10]->tileOp.GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpill) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Add4", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.issueEntries[6]->tileOp.GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.issueEntries[12]->tileOp.GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.issueEntries[13]->tileOp.GetOpcodeStr(), "COPY_IN");
}

TEST_F(ScheduleOoOTest, TestSpillInplace) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.issueEntries.begin(), ooOScheduler.issueEntries.begin() + 6, ooOScheduler.issueEntries.begin() + 11);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr add1 = GetIssueEntry("Add1", subGraph, ooOScheduler);
    EXPECT_NE(add1, nullptr);
    IssueEntryPtr add3 = GetIssueEntry("Add3", subGraph, ooOScheduler);
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ(ooOScheduler.issueEntryMap[(*add1->successors.begin())]->tileOp.GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(add3->predecessors.size(), 2);
}

TEST_F(ScheduleOoOTest, TestSpillMultiTensor) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN,
        Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t7", "t8"},
        {"t5", "t6", "t9"}, {"t7", "t8", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t5"}, {"t6"}, {"t7"}, {"t8"},
        {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Alloc7", "Copyin1", "Copyin2", "Copyin3", "Copyin4",
        "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {50, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->shape = {128, 128};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.issueEntries.size(), 21);
}

TEST_F(ScheduleOoOTest, TestSpillView) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestSpillAssemble) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1", "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestSpillFragFailed) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();

    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {32, 32};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    std::swap(ooOScheduler.issueEntries[0], ooOScheduler.issueEntries[1]);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSpillL0AFailed) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C,
        MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_L0C_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B,
        Opcode::OP_A_MULACC_B, Opcode::OP_A_MULACC_B, Opcode::OP_COPY_OUT, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t4"}, {"t2"}, {"t5"}, {"t3"}, {"t3"}, {"t6", "t7"}, {"t6", "t8"}, {"t9"}, {"t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t5"}, {"t3"}, {"t6"}, {"t7"}, {"t8"}, {"t2"}, {"t5"}, {"t3"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2",
        "L1toL0A1", "L1toL0A2", "AMulB1", "AMulB2", "AMulaccB1", "AMulaccB2", "Copyout1", "Copyout2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t10"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t7")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t10");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t8")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.GenSpillSchedule();
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestSchedule) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Add4", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr add = GetIssueEntry("Add2", subGraph, ooOScheduler);
    EXPECT_NE(add, nullptr);
    EXPECT_EQ(add->tileOp.oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add->tileOp.oOperand[0]->memoryrange.end, 65536);
}

TEST_F(ScheduleOoOTest, TestScheduleInplace) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr copyin = GetIssueEntry("Copyin1", subGraph, ooOScheduler);
    EXPECT_NE(copyin, nullptr);
    IssueEntryPtr add1 = GetIssueEntry("Add1", subGraph, ooOScheduler);
    EXPECT_NE(add1, nullptr);
    IssueEntryPtr add3 = GetIssueEntry("Add3", subGraph, ooOScheduler);
    EXPECT_NE(add3, nullptr);
    EXPECT_EQ(copyin->tileOp.oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(copyin->tileOp.oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(add1->tileOp.oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add1->tileOp.oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(add3->tileOp.oOperand[0]->memoryrange.start, 49152);
    EXPECT_EQ(add3->tileOp.oOperand[0]->memoryrange.end, 65536);
}

TEST_F(ScheduleOoOTest, TestScheduleView) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;
    tensor1->shape = {32, 32};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;
    tensor2->shape = {32, 32};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr copyin = GetIssueEntry("Copyin1", subGraph, ooOScheduler);
    EXPECT_NE(copyin, nullptr);
    IssueEntryPtr view1 = GetIssueEntry("View1", subGraph, ooOScheduler);
    EXPECT_NE(view1, nullptr);
    IssueEntryPtr view2 = GetIssueEntry("View2", subGraph, ooOScheduler);
    EXPECT_NE(view2, nullptr);
    EXPECT_EQ(copyin->tileOp.oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin->tileOp.oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(view1->tileOp.oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(view1->tileOp.oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(view2->tileOp.oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(view2->tileOp.oOperand[0]->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleAssemble) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1", "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {64, 64}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    tensor2->shape = {32, 32};
    std::shared_ptr<LogicalTensor> tensor3 = subGraph.GetTensor("t5");
    tensor3->shape = {32, 32};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    IssueEntryPtr copyin1 = GetIssueEntry("Copyin1", subGraph, ooOScheduler);
    EXPECT_NE(copyin1, nullptr);
    IssueEntryPtr copyin2 = GetIssueEntry("Copyin2", subGraph, ooOScheduler);
    EXPECT_NE(copyin2, nullptr);
    IssueEntryPtr assemble1 = GetIssueEntry("Assemble1", subGraph, ooOScheduler);
    EXPECT_NE(assemble1, nullptr);
    IssueEntryPtr assemble2 = GetIssueEntry("Assemble2", subGraph, ooOScheduler);
    EXPECT_NE(assemble2, nullptr);
    EXPECT_EQ(copyin1->tileOp.oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin1->tileOp.oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(copyin2->tileOp.oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(copyin2->tileOp.oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(assemble1->tileOp.oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(assemble1->tileOp.oOperand[0]->memoryrange.end, 16384);
    EXPECT_EQ(assemble2->tileOp.oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(assemble2->tileOp.oOperand[0]->memoryrange.end, 16384);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillCopyIn) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t5"}, {"t4", "t6"}, {"t8"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.newOperations_[9]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.newOperations_[9]->oOperand[0]->memoryrange.start, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[9]->oOperand[0]->memoryrange.end, 196608);
    EXPECT_EQ(ooOScheduler.newOperations_[10]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.newOperations_[10]->oOperand[0]->memoryrange.start, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[10]->oOperand[0]->memoryrange.end, 196608);
}

TEST_F(ScheduleOoOTest, TestScheduleSpill) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3", "t4"}, {"t3", "t4"}, {"t4", "t6"}, {"t5", "t8"}, {"t9"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t8"}, {"t9"}, {"t7"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2", "Add1", "Add2", "Add3", "Add4", "Copyout1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.newOperations_[6]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.newOperations_[6]->oOperand[0]->memoryrange.start, 0);
    EXPECT_EQ(ooOScheduler.newOperations_[6]->oOperand[0]->memoryrange.end, 65536);
    EXPECT_EQ(ooOScheduler.newOperations_[12]->GetOpcodeStr(), "UB_ALLOC");
    EXPECT_EQ(ooOScheduler.newOperations_[12]->oOperand[0]->memoryrange.start, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[12]->oOperand[0]->memoryrange.end, 196608);
    EXPECT_EQ(ooOScheduler.newOperations_[13]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.newOperations_[13]->oOperand[0]->memoryrange.start, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[13]->oOperand[0]->memoryrange.end, 196608);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillInplace) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5", "t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t10"}, {"t9"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t11"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t10");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t11");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    std::rotate(ooOScheduler.issueEntries.begin(), ooOScheduler.issueEntries.begin() + 6, ooOScheduler.issueEntries.begin() + 11);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
    EXPECT_EQ(ooOScheduler.newOperations_[9]->GetOpcodeStr(), "COPY_OUT");
    EXPECT_EQ(ooOScheduler.newOperations_[13]->GetOpcodeStr(), "COPY_IN");
    EXPECT_EQ(ooOScheduler.newOperations_[13]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.newOperations_[13]->oOperand[0]->memoryrange.end, 131072);
    EXPECT_EQ(ooOScheduler.newOperations_[14]->GetOpcodeStr(), "ADD");
    EXPECT_EQ(ooOScheduler.newOperations_[14]->oOperand[0]->memoryrange.start, 65536);
    EXPECT_EQ(ooOScheduler.newOperations_[14]->oOperand[0]->memoryrange.end, 131072);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillView) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_ADD, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t3"}, {"t5"}, {"t6"}, {"t4", "t7"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t7"}, {"t8"}, {"t9"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "View1", "View2", "Add1", "Add2", "Add3"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t6"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t5");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t3")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillAssemble) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_ASSEMBLE, Opcode::OP_ASSEMBLE, Opcode::OP_ADD, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {"t1"}, {"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7", "t8"}, {"t9", "t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t7"}, {"t8"}, {"t10"}, {"t11"}, {"t5"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t9"}, {"t10"}, {"t11"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Copyin1", "Copyin2", "Copyin3", "Copyin4", "Assemble1", "Assemble2", "Add1", "Add2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t9"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t6");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t5")->memoryrange.memId;

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillFragFailed) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB,
        MemoryType::MEM_UB, MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN,
        Opcode::OP_COPY_IN, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {"t1"}, {"t2"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t3", "t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Copyin1", "Copyin2", "Add1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t3"), nullptr);
    std::shared_ptr<LogicalTensor> tensor = subGraph.GetTensor("t3");
    tensor->shape = {32, 32};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleSpillL0AFailed) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A,
        MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1, MemoryType::MEM_L0A, MemoryType::MEM_L0C, MemoryType::MEM_L0C, MemoryType::MEM_L0C,
        MemoryType::MEM_L0C, MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_L1_ALLOC, Opcode::OP_L1_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0A_ALLOC, Opcode::OP_L0C_ALLOC,
        Opcode::OP_L0C_ALLOC, Opcode::OP_COPY_IN, Opcode::OP_COPY_IN, Opcode::OP_L1_TO_L0A, Opcode::OP_L1_TO_L0A, Opcode::OP_A_MUL_B, Opcode::OP_A_MUL_B,
        Opcode::OP_A_MULACC_B, Opcode::OP_A_MULACC_B, Opcode::OP_COPY_OUT, Opcode::OP_COPY_OUT};
    std::vector<std::vector<std::string>> ioperands{{}, {}, {}, {}, {}, {}, {"t1"}, {"t4"}, {"t2"}, {"t5"}, {"t3"}, {"t3"}, {"t6", "t7"}, {"t6", "t8"}, {"t9"}, {"t10"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t5"}, {"t3"}, {"t6"}, {"t7"}, {"t8"}, {"t2"}, {"t5"}, {"t3"}, {"t6"}, {"t7"}, {"t8"}, {"t9"}, {"t10"}, {"t11"}, {"t12"}};
    std::vector<std::string> opNames{"Alloc1", "Alloc2", "Alloc3", "Alloc4", "Alloc5", "Alloc6", "Copyin1", "Copyin2",
        "L1toL0A1", "L1toL0A2", "AMulB1", "AMulB2", "AMulaccB1", "AMulaccB2", "Copyout1", "Copyout2"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    EXPECT_NE(subGraph.GetTensor("t10"), nullptr);
    std::shared_ptr<LogicalTensor> tensor1 = subGraph.GetTensor("t9");
    tensor1->memoryrange.memId =
        subGraph.GetTensor("t7")->memoryrange.memId;
    tensor1->shape = {256, 128};
    subGraph.GetTensor("t7")->shape = {256, 128};
    std::shared_ptr<LogicalTensor> tensor2 = subGraph.GetTensor("t10");
    tensor2->memoryrange.memId =
        subGraph.GetTensor("t8")->memoryrange.memId;
    tensor2->shape = {256, 128};
    subGraph.GetTensor("t8")->shape = {256, 128};

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Init(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.SortOps();
    EXPECT_EQ(res, SUCCESS);
    res = ooOScheduler.ScheduleMainLoop();
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestEmptyOplist) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;
    OoOScheduler ooOScheduler(function);
    Status res = ooOScheduler.Schedule(scheduleOpList);
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestScheduleReshape) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_DEVICE_DDR};
    std::vector<Opcode> opCodes{Opcode::OP_RESHAPE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Reshape1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, SUCCESS);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin1) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestSingleCopyin2) {
    ComputationalGraphBuilder subGraph;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<MemoryType> tensorMemTypes{MemoryType::MEM_UB, MemoryType::MEM_UB};
    std::vector<Opcode> opCodes{Opcode::OP_UB_ALLOC, Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{}, {"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t2"}};
    std::vector<std::string> opNames{"Alloc1", "Copyin1"};
    EXPECT_EQ(subGraph.AddTensors(DataType::DT_FP32, {128, 128}, tensorMemTypes, tensorNames, 0), true);
    EXPECT_EQ(subGraph.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = subGraph.GetFunction();
    EXPECT_NE(function, nullptr);

    OoOScheduler ooOScheduler(*function);
    Status res = ooOScheduler.Schedule(function->Operations().DuplicatedOpList());
    EXPECT_EQ(res, FAILED);
}

TEST_F(ScheduleOoOTest, TestDelBufCount) {
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.DelBufRefCount(-1);
}

TEST_F(ScheduleOoOTest, TestDelBufCount_1) {
    Function function(Program::GetInstance(), "", "", nullptr);
    OoOScheduler oooSchedule(function);
    oooSchedule.bufRefCount[1] = -1;
    oooSchedule.DelBufRefCount(1);
}

TEST_F(ScheduleOoOTest, TestUpdateTensorAttr_DDR) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor1->memoryrange.memId = 1;

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    OoOScheduler oooSchedule(function);
    oooSchedule.UpdateTensorAttr(tensor1, MemoryType::MEM_DEVICE_DDR, tensor3, -1);
}

TEST_F(ScheduleOoOTest, TestUpdateTensorAttr_UB) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor1 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor1->SetMemoryTypeOriginal(MEM_DEVICE_DDR);
    tensor1->SetMemoryTypeToBe(MEM_DEVICE_DDR);
    tensor1->memoryrange.memId = 1;

    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    OoOScheduler oooSchedule(function);
    oooSchedule.UpdateTensorAttr(tensor1, MemoryType::MEM_UB, tensor3, -1);
}

TEST_F(ScheduleOoOTest, TestGetSpillTensor) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3}));
    alloc1.UpdateLatency(1);

    LogicalTensorPtr tensor = nullptr;
    auto allocIssue = std::make_shared<IssueEntry>(alloc1, 1);

    OoOScheduler oooSchedule(function);
    oooSchedule.GetSpillTensor(allocIssue, 1, tensor);
}

TEST_F(ScheduleOoOTest, TestCheckAllocIssue) {
    Function function(Program::GetInstance(), "", "", nullptr);
    std::vector<Operation *> scheduleOpList;

    std::vector<int64_t> shape = {128, 128};
    std::shared_ptr<LogicalTensor> tensor3 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 3;

    std::shared_ptr<LogicalTensor> tensor2 = std::make_shared<LogicalTensor>(function, DataType::DT_FP32, shape);
    tensor3->SetMemoryTypeOriginal(MEM_UB);
    tensor3->SetMemoryTypeToBe(MEM_UB);
    tensor3->memoryrange.memId = 1;

    auto &alloc1 = function.AddOperation(Opcode::OP_UB_ALLOC, {}, std::vector<std::shared_ptr<LogicalTensor>>({tensor3, tensor2}));
    alloc1.UpdateLatency(1);

    OoOScheduler oooSchedule(function);
    oooSchedule.Init(function.Operations().DuplicatedOpList());
}
} // namespace npu::tile_fwk
