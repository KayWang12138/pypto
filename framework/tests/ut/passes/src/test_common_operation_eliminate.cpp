/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_common_operation_eliminate.cpp
 * \brief Unit test for common_operation_eliminate pass.
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/graph_partition/common_operation_eliminate.h"
#include <fstream>
#include <vector>
#include <string>

namespace npu {
namespace tile_fwk {

class CommonOperationEliminateTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    }
    void TearDown() override {}
};

TEST_F(CommonOperationEliminateTest, EliminateRedundantOps) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2","t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 2;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantMultiInputOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t1", "t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 2;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantMultiOutputOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t4"}, {"t3", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2", "t3"}, {"t4", "t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"RowMax1", "RowMax2", "MUL1", "MUL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t6", "t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, EliminateRedundantCascadeOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_EXP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2"}, {"t3"}, {"t4", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "EXP1", "EXP2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
    std::shared_ptr<LogicalTensor> tensorPtr = G.GetTensor("t1");
    EXPECT_NE(tensorPtr, nullptr);
    EXPECT_EQ(tensorPtr->GetConsumers().size(), 1);
    tensorPtr = G.GetTensor("t6");
    EXPECT_NE(tensorPtr, nullptr);
    EXPECT_EQ(tensorPtr->GetProducers().size(), 1);
}

TEST_F(CommonOperationEliminateTest, IgnoreSingleInputOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t3"}, {"t2", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t4"}, {"t5"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t3"}), true);
    EXPECT_EQ(G.SetOutCast({"t5"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreMultiInputOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_MUL, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t1", "t4"}, {"t3", "t5"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t5"}, {"t6"}};
    std::vector<std::string> opNames{"MUL1", "MUL2", "MUL3"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2", "t4"}), true);
    EXPECT_EQ(G.SetOutCast({"t6"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentAttr) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ADDS, Opcode::OP_ADDS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ADDS1", "ADDS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Operation* opPtr = G.GetOp("ADDS1");
    EXPECT_NE(opPtr, nullptr);
    opPtr->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, 1.0));
    opPtr = G.GetOp("ADDS2");
    EXPECT_NE(opPtr, nullptr);
    const double value2 = 2.0;
    opPtr->SetAttribute(OpAttributeKey::scalar, Element(DataType::DT_FP32, value2));
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;//修复后有序遍历tensor，使得连续冗余场景正确消除
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_EXP, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS", "EXP", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreDifferentSubgraph) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ABS, Opcode::OP_ABS, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}};
    std::vector<std::string> opNames{"ABS1", "ABS2", "MUL"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    Operation* opPtr = G.GetOp("ABS1");
    EXPECT_NE(opPtr, nullptr);
    opPtr->UpdateSubgraphID(0);
    opPtr = G.GetOp("ABS2");
    EXPECT_NE(opPtr, nullptr);
    opPtr->UpdateSubgraphID(1);
    opPtr = G.GetOp("MUL");
    EXPECT_NE(opPtr, nullptr);
    const int subgraphID2 = 2;
    opPtr->UpdateSubgraphID(subgraphID2);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    const int subgraphNum = 3;
    function->SetTotalSubGraphCount(subgraphNum);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 3;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, IgnoreSpecialOp) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6", "t7"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_VIEW, Opcode::OP_MUL,
                                Opcode::OP_L1_TO_FIX, Opcode::OP_L1_TO_FIX, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t1"}, {"t2", "t3"}, {"t1"}, {"t1"}, {"t5", "t6"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}, {"t4"}, {"t5"}, {"t6"}, {"t7"}};
    std::vector<std::string> opNames{"VIEW1", "VIEW2", "MUL1", "COPY1", "COPY2", "MUL2"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1"}), true);
    EXPECT_EQ(G.SetOutCast({"t4", "t7"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    COE.Run(*function, "", "", 0);
    const int validOpNum = 6;
    EXPECT_EQ(function->Operations().size(), validOpNum);
}

TEST_F(CommonOperationEliminateTest, TestShmemGetGm2UBChecker){
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {1, 1}, {"dummy"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {1, 1, 4, 64}, {"shmemData"}), true);
    EXPECT_EQ(G.AddTensors(DataType::DT_INT32, {4, 64}, {"out"}), true);
    std::vector<Opcode> opCodes{Opcode::OP_SHMEM_GET_GM2UB};
    std::vector<std::vector<std::string>> ioperands{{"dummy", "shmemData"}};
    std::vector<std::vector<std::string>> ooperands{{"out"}};
    std::vector<std::string> opNames{"TILE_SHMEM_GET_GM2UB"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"dummy", "shmemData"}), true);
    EXPECT_EQ(G.SetOutCast({"out"}), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, SUCCESS) << "COE Precheck failed for OP_SHMEM_GET_GM2UB!";
}

TEST_F(CommonOperationEliminateTest, PreCheck_CopyIn_InvalidInputNum) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t2"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t3"), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"COPY_IN_InvalidInput"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(CommonOperationEliminateTest, PreCheck_CopyIn_OffsetShapeMismatch) {
    ComputationalGraphBuilder G;
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t1"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "t2"), true);
    std::vector<Opcode> opCodes{Opcode::OP_COPY_IN};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"COPY_IN_OffsetMismatch"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();
    ASSERT_NE(function, nullptr);
    Operation* copyOp = G.GetOp("COPY_IN_OffsetMismatch");
    ASSERT_NE(copyOp, nullptr);
    auto opAttr = copyOp->GetOpAttribute();
    ASSERT_NE(opAttr, nullptr);
    auto copyAttr = dynamic_cast<CopyOpAttribute*>(opAttr.get());
    ASSERT_NE(copyAttr, nullptr);
    auto [fromOffset, memType] = copyAttr->GetCopyInAttr();
    (void)memType;
    std::vector<OpImmediate> newFromOffset;
    newFromOffset.emplace_back(0);
    newFromOffset.emplace_back(1);
    newFromOffset.emplace_back(2);
    copyAttr->SetFromOffset(newFromOffset);
    G.GetTensor("t1")->offset = {0, 0};
    CommonOperationEliminate COE;
    Status preCheckStatus = COE.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

// ========== 测试用例：InsertAssembleCopy - 整体插入拷贝序列流程 ==========
TEST_F(CommonOperationEliminateTest, InsertAssembleCopy) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "InsertAssembleCopy", "InsertAssembleCopy", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("InsertAssembleCopy", currFunctionPtr);

    // 创建共享输入tensor (UB内存类型)
    std::vector<int64_t> shape = {32, 64};
    auto sharedInput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    sharedInput->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建多个输出tensor
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output1->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output2->SetMemoryTypeBoth(MemoryType::MEM_UB, true);
    auto output3 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output3->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建多个ASSEMBLE操作，共享同一个输入
    auto &assemble1 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output1});
    assemble1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &assemble2 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output2});
    assemble2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &assemble3 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output3});
    assemble3.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(sharedInput);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);
    currFunctionPtr->outCasts_.push_back(output3);

    // 调用InsertAssembleCopy
    CommonOperationEliminate commonOperation;
    commonOperation.InsertAssembleCopy(*currFunctionPtr);

    // 验证插入的拷贝序列
    int copyInNum = 0;
    int copyOutNum = 0;
    int assembleNum = 0;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInNum++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutNum++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleNum++;
        }
    }

    // 应该为2个ASSEMBLE操作插入拷贝序列（每个需要2个COPY操作）
    EXPECT_EQ(assembleNum, 3) << "Should have 3 ASSEMBLE operations";
    EXPECT_EQ(copyInNum, 2) << "Should insert 2 COPY_IN operations";
    EXPECT_EQ(copyOutNum, 2) << "Should insert 2 COPY_OUT operations";
}

// ========== 测试用例：InsertAssembleCopy - DDR内存类型场景 ==========
TEST_F(CommonOperationEliminateTest, InsertAssembleCopyDDR) {
    auto currFunctionPtr =
        std::make_shared<Function>(Program::GetInstance(), "InsertAssembleCopyDDR", "InsertAssembleCopyDDR", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("InsertAssembleCopyDDR", currFunctionPtr);

    // 创建共享输入tensor (DDR内存类型)
    std::vector<int64_t> shape = {32, 64};
    auto sharedInput = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    sharedInput->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // 创建多个输出tensor
    auto output1 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output1->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);
    auto output2 = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output2->SetMemoryTypeBoth(MemoryType::MEM_DEVICE_DDR, true);

    // 创建多个ASSEMBLE操作，共享同一个输入
    auto &assemble1 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output1});
    assemble1.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));
    auto &assemble2 = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {sharedInput}, {output2});
    assemble2.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(sharedInput);
    currFunctionPtr->outCasts_.push_back(output1);
    currFunctionPtr->outCasts_.push_back(output2);

    // 调用InsertAssembleCopy
    CommonOperationEliminate commonOperationEliminate;
    commonOperationEliminate.InsertAssembleCopy(*currFunctionPtr);

    // 验证插入的拷贝序列
    int copyInNum = 0;
    int copyOutNum = 0;
    int assembleNum = 0;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInNum++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutNum++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleNum++;
        }
    }

    // 应该为1个ASSEMBLE操作插入拷贝序列（DDR→UB→DDR）
    EXPECT_EQ(assembleNum, 2) << "Should have 2 ASSEMBLE operations";
    EXPECT_EQ(copyInNum, 1) << "Should insert 1 COPY_IN operation";
    EXPECT_EQ(copyOutNum, 1) << "Should insert 1 COPY_OUT operation";
}

// ========== 测试用例：InsertAssembleCopy - 单个ASSEMBLE不插入拷贝 ==========
TEST_F(CommonOperationEliminateTest, InsertAssembleCopySingleAssemble) {
    auto currFunctionPtr = std::make_shared<Function>(
        Program::GetInstance(), "InsertAssembleCopySingleAssemble", "InsertAssembleCopySingleAssemble", nullptr);
    EXPECT_TRUE(currFunctionPtr != nullptr);
    Program::GetInstance().InsertFuncToFunctionMap("InsertAssembleCopySingleAssemble", currFunctionPtr);

    // 创建输入tensor
    std::vector<int64_t> shape = {32, 64};
    auto input = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    input->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建输出tensor
    auto output = std::make_shared<LogicalTensor>(*currFunctionPtr, DT_FP32, shape);
    output->SetMemoryTypeBoth(MemoryType::MEM_UB, true);

    // 创建单个ASSEMBLE操作
    auto &assemble = currFunctionPtr->AddRawOperation(Opcode::OP_ASSEMBLE, {input}, {output});
    assemble.SetOpAttribute(std::make_shared<AssembleOpAttribute>(std::vector<int64_t>{0, 0}));

    currFunctionPtr->inCasts_.push_back(input);
    currFunctionPtr->outCasts_.push_back(output);

    // 调用InsertAssembleCopy
    CommonOperationEliminate commonOperationEliminateTest;
    commonOperationEliminateTest.InsertAssembleCopy(*currFunctionPtr);

    // 验证没有插入拷贝序列
    int copyInNum = 0;
    int copyOutNum = 0;
    int assembleNum = 0;
    for (const auto &op : currFunctionPtr->Operations()) {
        if (op.GetOpcode() == Opcode::OP_COPY_IN) {
            copyInNum++;
        } else if (op.GetOpcode() == Opcode::OP_COPY_OUT) {
            copyOutNum++;
        } else if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            assembleNum++;
        }
    }

    // 单个ASSEMBLE不应该插入拷贝序列
    EXPECT_EQ(assembleNum, 1) << "Should have 1 ASSEMBLE operation";
    EXPECT_EQ(copyInNum, 0) << "Should not insert COPY_IN operation";
    EXPECT_EQ(copyOutNum, 0) << "Should not insert COPY_OUT operation";
}
} // namespace tile_fwk
} // namespace npu