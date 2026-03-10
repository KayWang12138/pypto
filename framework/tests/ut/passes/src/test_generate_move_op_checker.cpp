/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_generate_move_op_checker.cpp
 * \brief Unit test for Generate_Move_Op pass checker - all error scenarios
 */

#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "ut_json/ut_json_tool.h"
#include "computational_graph_builder.h"
#include "passes/tile_graph_pass/data_path/generate_move_op.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/attribute.h"
#include <fstream>
#include <vector>
#include <string>

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class TestGenerateMoveOpChecker : public ::testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
        config::SetHostConfig(KEY_STRATEGY, "GenerateMoveOpCheckerTestStrategy");
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    void TearDown() override {}
};

template <typename OpType>
OpType *FindOpByOpcode(Function *function, Opcode targetOpcode) {
    OpType *targetOp = nullptr;
    for (auto &op : function->Operations()) {
        if (op.GetOpcode() == targetOpcode) {
            targetOp = &op;
            break;
        }
    }
    return targetOp;
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_AttrNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"VIEW_AttrNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_MoreThanOneInput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{
        {"t1", "t2"}
    };
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"VIEW_MultiInput"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_MoreThanOneOutput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{
        {"t2", "t3"}
    };
    std::vector<std::string> opNames{"VIEW_MultiOutput"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_InputIsNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"VIEW_InputNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *viewOp = FindOpByOpcode<Operation>(function, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);

    auto &inputOperands = const_cast<std::vector<std::shared_ptr<LogicalTensor>> &>(viewOp->GetIOperands());
    inputOperands[0] = nullptr;

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_OutputIsNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"VIEW_OutputNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *viewOp = FindOpByOpcode<Operation>(function, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);

    auto &outputOperands = const_cast<std::vector<std::shared_ptr<LogicalTensor>> &>(viewOp->GetOOperands());
    outputOperands[0] = nullptr;

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_OutputHasNullConsumer) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"VIEW_NullConsumer"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *viewOp = FindOpByOpcode<Operation>(function, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);

    auto viewOutputTensor = viewOp->GetOOperands().front();
    ASSERT_NE(viewOutputTensor, nullptr);
    auto &consumers = const_cast<std::set<Operation *, LogicalTensor::CompareOp> &>(viewOutputTensor->GetConsumers());
    consumers.clear();
    consumers.insert(nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_ConsumerNotSupportDDR) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}};
    std::vector<std::string> opNames{"VIEW_DDR", "MUL_NotSupportDDR"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *viewOp = FindOpByOpcode<Operation>(function, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);

    auto viewOutputTensor = viewOp->GetOOperands().front();
    ASSERT_NE(viewOutputTensor, nullptr);
    viewOutputTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_ConvertPathInvalid) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}};
    std::vector<std::string> opNames{"VIEW_DDR", "CONVERT_Invalid"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *viewOp = FindOpByOpcode<Operation>(function, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);

    auto viewOutputTensor = viewOp->GetOOperands().front();
    ASSERT_NE(viewOutputTensor, nullptr);
    viewOutputTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);

    auto convertAttr = std::make_shared<ConvertOpAttribute>(MemoryType::MEM_L1, MemoryType::MEM_L0A);
    Operation *convertOp = FindOpByOpcode<Operation>(function, Opcode::OP_CONVERT);
    if (convertOp != nullptr) {
        convertOp->SetOpAttribute(convertAttr);
    }

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_AttrNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"ASSEMBLE_AttrNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_MoreThanOneInput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{
        {"t1", "t2"}
    };
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"ASSEMBLE_MultiInput"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_MoreThanOneOutput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t0", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"t0"}};
    std::vector<std::vector<std::string>> ooperands{
        {"t2", "t3"}
    };
    std::vector<std::string> opNames{"ASSEMBLE_MultiOutput1"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_InputIsNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"ASSEMBLE_InputNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *assembleOp = FindOpByOpcode<Operation>(function, Opcode::OP_ASSEMBLE);
    ASSERT_NE(assembleOp, nullptr);

    auto &inputOperands = const_cast<std::vector<std::shared_ptr<LogicalTensor>> &>(assembleOp->GetIOperands());
    inputOperands[0] = nullptr;

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_OutputIsNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"ASSEMBLE_OutputNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *assembleOp = FindOpByOpcode<Operation>(function, Opcode::OP_ASSEMBLE);
    ASSERT_NE(assembleOp, nullptr);

    auto &outputOperands = const_cast<std::vector<std::shared_ptr<LogicalTensor>> &>(assembleOp->GetOOperands());
    outputOperands[0] = nullptr;

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_AttrNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"CONVERT_AttrNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_MoreThanOneInput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{
        {"t1", "t2"}
    };
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"CONVERT_MultiInput"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_MoreThanOneOutput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{
        {"t2", "t3"}
    };
    std::vector<std::string> opNames{"CONVERT_MultiOutput"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_InputIsNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"CONVERT_InputNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *convertOp = FindOpByOpcode<Operation>(function, Opcode::OP_CONVERT);
    ASSERT_NE(convertOp, nullptr);

    auto &inputOperands = const_cast<std::vector<std::shared_ptr<LogicalTensor>> &>(convertOp->GetIOperands());
    inputOperands[0] = nullptr;

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_OutputIsNull) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"CONVERT_OutputNull"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *convertOp = FindOpByOpcode<Operation>(function, Opcode::OP_CONVERT);
    ASSERT_NE(convertOp, nullptr);

    auto &outputOperands = const_cast<std::vector<std::shared_ptr<LogicalTensor>> &>(convertOp->GetOOperands());
    outputOperands[0] = nullptr;

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_SameMemType) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"CONVERT_SameMemType"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *convertOp = FindOpByOpcode<Operation>(function, Opcode::OP_CONVERT);
    ASSERT_NE(convertOp, nullptr);

    auto inputTensor = convertOp->GetIOperands().front();
    auto outputTensor = convertOp->GetOOperands().front();
    ASSERT_NE(inputTensor, nullptr);
    ASSERT_NE(outputTensor, nullptr);

    inputTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    outputTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_DiffShape) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames1{"t1"};
    std::vector<int64_t> shape1{16, 32};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape1, tensorNames1), true);

    std::vector<std::string> tensorNames2{"t2"};
    std::vector<int64_t> shape2{32, 64};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, shape2, tensorNames2), true);

    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"CONVERT_DiffShape1"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_Duplicate_Exist) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_DUPLICATE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"DUPLICATE_Exist"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status postCheckStatus = checker.PostCheck(*function);
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_Convert_Exist) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"CONVERT_Exist"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status postCheckStatus = checker.PostCheck(*function);
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_ViewOp_MoreThanOneInput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{
        {"t1", "t2"}
    };
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"VIEW_MultiInput"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    GenerateMoveOp checker;
    Status postCheckStatus = checker.PostCheck(*function);
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_ViewOp_MoreThanOneOutput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{
        {"t2", "t3"}
    };
    std::vector<std::string> opNames{"VIEW_MultiOutput"};
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    GenerateMoveOp checker;
    Status postCheckStatus = checker.PostCheck(*function);
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_ViewOp_InputMemTypeNotMatch) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"VIEW_MemTypeNotMatch"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *viewOp = FindOpByOpcode<Operation>(function, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);

    auto inputTensor = viewOp->GetIOperands().front();
    auto outputTensor = viewOp->GetOOperands().front();
    ASSERT_NE(inputTensor, nullptr);
    ASSERT_NE(outputTensor, nullptr);

    inputTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    outputTensor->SetMemoryTypeOriginal(MemoryType::MEM_L1);

    GenerateMoveOp checker;
    Status postCheckStatus = checker.PostCheck(*function);
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PostCheck_AssembleOp_MoreThanOneInput) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{
        {"t1", "t2"}
    };
    std::vector<std::vector<std::string>> ooperands{{"t3"}};
    std::vector<std::string> opNames{"ASSEMBLE_MultiInput2"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 32}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);
    GenerateMoveOp checker;
    Status postCheckStatus = checker.PostCheck(*function);
    EXPECT_EQ(postCheckStatus, FAILED);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ViewOp_Valid) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2", "t3"};
    std::vector<Opcode> opCodes{Opcode::OP_VIEW, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1"}, {"t2"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}, {"t3"}};
    std::vector<std::string> opNames{"VIEW_Valid", "MUL"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *viewOp = FindOpByOpcode<Operation>(function, Opcode::OP_VIEW);
    ASSERT_NE(viewOp, nullptr);

    auto viewAttr = std::make_shared<ViewOpAttribute>(Offset(0, 0));
    viewOp->SetOpAttribute(viewAttr);

    auto viewOutputTensor = viewOp->GetOOperands().front();
    ASSERT_NE(viewOutputTensor, nullptr);
    viewOutputTensor->SetMemoryTypeOriginal(MemoryType::MEM_L1);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, SUCCESS);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_AssembleOp_Valid) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_ASSEMBLE};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"ASSEMBLE_Valid"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *assembleOp = FindOpByOpcode<Operation>(function, Opcode::OP_ASSEMBLE);
    ASSERT_NE(assembleOp, nullptr);

    auto assembleAttr = std::make_shared<AssembleOpAttribute>(Offset(0, 0));
    assembleOp->SetOpAttribute(assembleAttr);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, SUCCESS);
}

TEST_F(TestGenerateMoveOpChecker, PreCheck_ConvertOp_Valid) {
    ComputationalGraphBuilder G;
    std::vector<std::string> tensorNames{"t1", "t2"};
    std::vector<Opcode> opCodes{Opcode::OP_CONVERT};
    std::vector<std::vector<std::string>> ioperands{{"t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t2"}};
    std::vector<std::string> opNames{"CONVERT_Valid"};

    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, {16, 16}, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    Function *function = G.GetFunction();
    EXPECT_NE(function, nullptr);

    Operation *convertOp = FindOpByOpcode<Operation>(function, Opcode::OP_CONVERT);
    ASSERT_NE(convertOp, nullptr);

    auto convertAttr = std::make_shared<ConvertOpAttribute>(MemoryType::MEM_DEVICE_DDR, MemoryType::MEM_L1);
    convertOp->SetOpAttribute(convertAttr);

    auto inputTensor = convertOp->GetIOperands().front();
    auto outputTensor = convertOp->GetOOperands().front();
    ASSERT_NE(inputTensor, nullptr);
    ASSERT_NE(outputTensor, nullptr);

    inputTensor->SetMemoryTypeOriginal(MemoryType::MEM_DEVICE_DDR);
    outputTensor->SetMemoryTypeOriginal(MemoryType::MEM_L1);

    GenerateMoveOp checker;
    Status preCheckStatus = checker.PreCheck(*function);
    EXPECT_EQ(preCheckStatus, SUCCESS);
}

} // namespace tile_fwk
} // namespace npu
