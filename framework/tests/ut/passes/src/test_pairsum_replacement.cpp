/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_pairsum_replacement.cpp
 * \brief Unit test for PairSumReplacementPass.
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "passes/block_graph_pass/pairsum_replacement.h"
#include "computational_graph_builder.h"
#include <vector>
#include <string>

using namespace npu::tile_fwk;

class PairSumReplacementTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
    }
    
    void TearDown() override {}

 protected:
    // Helper method to create a basic graph with single PAIRSUM operation
    Function* CreateBasicPairSumGraph(ComputationalGraphBuilder &G,
                                       const std::vector<int64_t> &tileShape = {16, 16}) {
        std::vector<std::string> tensorNames{"t1", "t2", "t3"};
        std::vector<Opcode> opCodes{Opcode::OP_PAIRSUM};
        std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}};
        std::vector<std::vector<std::string>> ooperands{{"t3"}};
        std::vector<std::string> opNames{"pairsum1"};
        
        EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
        EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
        EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
        EXPECT_EQ(G.SetOutCast({"t3"}), true);
        
        return G.GetFunction();
    }
    
    // Helper method to run pass and verify basic expectations
    void RunPassAndVerifySuccess(Function* func, PairSumReplacementPass &pass) {
        ASSERT_NE(func, nullptr);
        EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    }
    
    // Helper method to verify operation opcode
    void VerifyOpcode(Operation* op, Opcode expectedOpcode) {
        ASSERT_NE(op, nullptr);
        EXPECT_EQ(op->GetOpcode(), expectedOpcode);
    }
};

// Test Case 1: Static shape replacement - both inputs have no valid_shape
TEST_F(PairSumReplacementTest, TestStaticShapeReplacement) {
    ComputationalGraphBuilder G;
    Function* func = CreateBasicPairSumGraph(G);
    
    // Verify PAIRSUM exists before pass
    Operation* op = G.GetOp("pairsum1");
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
    
    // Run the pass
    PairSumReplacementPass pass;
    RunPassAndVerifySuccess(func, pass);
    
    // Verify PAIRSUM was replaced with ADD
    VerifyOpcode(op, Opcode::OP_ADD);
    
    // Verify inputs and outputs are unchanged
    EXPECT_EQ(op->GetIOperands().size(), 2);
    EXPECT_EQ(op->GetOOperands().size(), 1);
}

// Test Case 2: Same valid_shape replacement - both inputs have same valid_shape
TEST_F(PairSumReplacementTest, TestSameValidShapeReplacement) {
    ComputationalGraphBuilder G;
    Function* func = CreateBasicPairSumGraph(G);
    
    // Set same valid_shape for both inputs
    auto tensor1 = G.GetTensor("t1");
    auto tensor2 = G.GetTensor("t2");
    std::vector<SymbolicScalar> validShape{SymbolicScalar(8), SymbolicScalar(16)};
    tensor1->UpdateDynValidShape(validShape);
    tensor2->UpdateDynValidShape(validShape);
    
    // Verify PAIRSUM exists before pass
    Operation* op = G.GetOp("pairsum1");
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
    
    // Run the pass
    PairSumReplacementPass pass;
    RunPassAndVerifySuccess(func, pass);
    
    // Verify PAIRSUM was replaced with ADD
    VerifyOpcode(op, Opcode::OP_ADD);
}

// Test Case 3: Different valid_shape - should NOT replace
TEST_F(PairSumReplacementTest, TestDifferentValidShapeNoReplacement) {
    ComputationalGraphBuilder G;
    Function* func = CreateBasicPairSumGraph(G);
    
    // Set different valid_shape for inputs
    auto tensor1 = G.GetTensor("t1");
    auto tensor2 = G.GetTensor("t2");
    std::vector<SymbolicScalar> validShape1{SymbolicScalar(8), SymbolicScalar(16)};
    std::vector<SymbolicScalar> validShape2{SymbolicScalar(10), SymbolicScalar(16)};
    tensor1->UpdateDynValidShape(validShape1);
    tensor2->UpdateDynValidShape(validShape2);
    
    // Verify PAIRSUM exists before pass
    Operation* op = G.GetOp("pairsum1");
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
    
    // Run the pass
    PairSumReplacementPass pass;
    RunPassAndVerifySuccess(func, pass);
    
    // Verify PAIRSUM was NOT replaced (should still be PAIRSUM)
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
}

// Test Case 4: Mixed scenario (one static, one with valid_shape) - should NOT replace
TEST_F(PairSumReplacementTest, TestMixedScenarioNoReplacement) {
    ComputationalGraphBuilder G;
    Function* func = CreateBasicPairSumGraph(G);
    
    // Set valid_shape only for t1, t2 remains static
    auto tensor1 = G.GetTensor("t1");
    std::vector<SymbolicScalar> validShape{SymbolicScalar(8), SymbolicScalar(16)};
    tensor1->UpdateDynValidShape(validShape);
    // t2 has no valid_shape
    
    // Verify PAIRSUM exists before pass
    Operation* op = G.GetOp("pairsum1");
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
    
    // Run the pass
    PairSumReplacementPass pass;
    RunPassAndVerifySuccess(func, pass);
    
    // Verify PAIRSUM was NOT replaced (should still be PAIRSUM)
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
}

// Test Case 5: inputCombineAxis attribute preservation
TEST_F(PairSumReplacementTest, TestInputCombineAxisPreservation) {
    ComputationalGraphBuilder G;
    Function* func = CreateBasicPairSumGraph(G);
    
    // Set inputCombineAxis attribute
    Operation* op = G.GetOp("pairsum1");
    ASSERT_NE(op, nullptr);
    std::vector<bool> combineAxis{true};
    op->SetAttr(OpAttributeKey::inputCombineAxis, combineAxis);
    EXPECT_TRUE(op->HasAttribute(OpAttributeKey::inputCombineAxis));
    
    // Run the pass
    PairSumReplacementPass pass;
    RunPassAndVerifySuccess(func, pass);
    
    // Verify PAIRSUM was replaced with ADD
    VerifyOpcode(op, Opcode::OP_ADD);
    
    // Verify inputCombineAxis attribute is preserved
    EXPECT_TRUE(op->HasAttribute(OpAttributeKey::inputCombineAxis));
    std::vector<bool> combineAxisResult;
    EXPECT_TRUE(op->GetAttr(OpAttributeKey::inputCombineAxis, combineAxisResult));
    EXPECT_EQ(combineAxisResult.size(), 1);
    EXPECT_EQ(combineAxisResult[0], true);
}

// Test Case 6: excludeBufferReuse attribute removal
TEST_F(PairSumReplacementTest, TestExcludeBufferReuseRemoval) {
    ComputationalGraphBuilder G;
    Function* func = CreateBasicPairSumGraph(G);
    
    // Set excludeBufferReuse attribute
    Operation* op = G.GetOp("pairsum1");
    ASSERT_NE(op, nullptr);
    op->SetAttr(OpAttributeKey::excludeBufferReuse, true);
    EXPECT_TRUE(op->HasAttribute(OpAttributeKey::excludeBufferReuse));
    
    // Run the pass
    PairSumReplacementPass pass;
    RunPassAndVerifySuccess(func, pass);
    
    // Verify PAIRSUM was replaced with ADD
    VerifyOpcode(op, Opcode::OP_ADD);
    
    // Verify excludeBufferReuse attribute is removed
    EXPECT_FALSE(op->HasAttribute(OpAttributeKey::excludeBufferReuse));
}

// Test Case 7: Multiple OP_PAIRSUM operations (mixed)
TEST_F(PairSumReplacementTest, TestMultiplePairSumMixed) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    
    // Build a graph with multiple OP_PAIRSUM operations
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4", "t5", "t6"};
    std::vector<Opcode> opCodes{Opcode::OP_PAIRSUM, Opcode::OP_PAIRSUM};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3", "t4"}};
    std::vector<std::vector<std::string>> ooperands{{"t5"}, {"t6"}};
    std::vector<std::string> opNames{"pairsum1", "pairsum2"};
    
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2", "t3", "t4"}), true);
    EXPECT_EQ(G.SetOutCast({"t5", "t6"}), true);
    
    // pairsum1: both static (should replace)
    // pairsum2: different valid_shape (should NOT replace)
    auto tensor3 = G.GetTensor("t3");
    auto tensor4 = G.GetTensor("t4");
    std::vector<SymbolicScalar> validShape3{SymbolicScalar(8), SymbolicScalar(16)};
    std::vector<SymbolicScalar> validShape4{SymbolicScalar(10), SymbolicScalar(16)};
    tensor3->UpdateDynValidShape(validShape3);
    tensor4->UpdateDynValidShape(validShape4);
    
    Function* func = G.GetFunction();
    ASSERT_NE(func, nullptr);
    
    Operation* op1 = G.GetOp("pairsum1");
    Operation* op2 = G.GetOp("pairsum2");
    ASSERT_NE(op1, nullptr);
    ASSERT_NE(op2, nullptr);
    
    // Run the pass
    PairSumReplacementPass pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    
    // Verify pairsum1 was replaced
    EXPECT_EQ(op1->GetOpcode(), Opcode::OP_ADD);
    
    // Verify pairsum2 was NOT replaced
    EXPECT_EQ(op2->GetOpcode(), Opcode::OP_PAIRSUM);
}

// Test Case 8: Empty function
TEST_F(PairSumReplacementTest, TestEmptyFunction) {
    ComputationalGraphBuilder G;
    
    Function* func = G.GetFunction();
    ASSERT_NE(func, nullptr);
    
    PairSumReplacementPass pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    
    EXPECT_EQ(func->Operations(false).size(), 0);
}

// Test Case 9: Function with no PAIRSUM operations
TEST_F(PairSumReplacementTest, TestNoPairSumOperations) {
    ComputationalGraphBuilder G;
    std::vector<int64_t> tileShape{16, 16};
    
    // Build a graph with various operations but no PAIRSUM
    std::vector<std::string> tensorNames{"t1", "t2", "t3", "t4"};
    std::vector<Opcode> opCodes{Opcode::OP_ADD, Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"t1", "t2"}, {"t3", "t1"}};
    std::vector<std::vector<std::string>> ooperands{{"t3"}, {"t4"}};
    std::vector<std::string> opNames{"add1", "mul1"};
    
    EXPECT_EQ(G.AddTensors(DataType::DT_FP32, tileShape, tensorNames), true);
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);
    EXPECT_EQ(G.SetInCast({"t1", "t2"}), true);
    EXPECT_EQ(G.SetOutCast({"t4"}), true);
    
    Function* func = G.GetFunction();
    ASSERT_NE(func, nullptr);
    
    size_t opCountBefore = func->Operations(false).size();
    
    // Run the pass
    PairSumReplacementPass pass;
    EXPECT_EQ(pass.RunOnFunction(*func), SUCCESS);
    
    // Verify operation count unchanged
    EXPECT_EQ(func->Operations(false).size(), opCountBefore);
    
    // Verify no PAIRSUM operations
    for (const auto &op : func->Operations(false)) {
        EXPECT_NE(op.GetOpcode(), Opcode::OP_PAIRSUM);
    }
}

// Test Case 10: Symbolic expressions (same)
TEST_F(PairSumReplacementTest, TestSameSymbolicExpression) {
    ComputationalGraphBuilder G;
    Function* func = CreateBasicPairSumGraph(G);
    
    // Set same symbolic expression for both inputs
    auto tensor1 = G.GetTensor("t1");
    auto tensor2 = G.GetTensor("t2");
    std::vector<SymbolicScalar> validShape{SymbolicScalar("N"), SymbolicScalar(16)};
    tensor1->UpdateDynValidShape(validShape);
    tensor2->UpdateDynValidShape(validShape);
    
    Operation* op = G.GetOp("pairsum1");
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
    
    // Run the pass
    PairSumReplacementPass pass;
    RunPassAndVerifySuccess(func, pass);
    
    // Verify PAIRSUM was replaced with ADD
    VerifyOpcode(op, Opcode::OP_ADD);
}

// Test Case 11: Symbolic expressions (different)
TEST_F(PairSumReplacementTest, TestDifferentSymbolicExpression) {
    ComputationalGraphBuilder G;
    Function* func = CreateBasicPairSumGraph(G);
    
    // Set different symbolic expressions for inputs
    auto tensor1 = G.GetTensor("t1");
    auto tensor2 = G.GetTensor("t2");
    std::vector<SymbolicScalar> validShape1{SymbolicScalar("N"), SymbolicScalar(16)};
    std::vector<SymbolicScalar> validShape2{SymbolicScalar("M"), SymbolicScalar(16)};
    tensor1->UpdateDynValidShape(validShape1);
    tensor2->UpdateDynValidShape(validShape2);
    
    Operation* op = G.GetOp("pairsum1");
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
    
    // Run the pass
    PairSumReplacementPass pass;
    RunPassAndVerifySuccess(func, pass);
    
    // Verify PAIRSUM was NOT replaced
    VerifyOpcode(op, Opcode::OP_PAIRSUM);
}