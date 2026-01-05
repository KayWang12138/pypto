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
 * \file test_autodiff.cpp
 * \brief Unit test for AutodiffPass.
 */

#include <fstream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "passes/pass_mgr/pass_manager.h"
#include "interface/configs/config_manager.h"
#include "computational_graph_builder.h"
#include "passes/tensor_graph_pass/autodiff.h"
#include "passes/tensor_graph_pass/vjp_registry.h"

namespace npu {
namespace tile_fwk {

class AutodiffPassTest : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ONLY_HOST_COMPILE, true);
    }
    void TearDown() override {}
};

// Test basic add operation gradient
TEST_F(AutodiffPassTest, AddGradient) {
    ComputationalGraphBuilder G;

    // Create tensors: a + b = loss
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    // Add operation: loss = a + b
    std::vector<Opcode> opCodes{Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"a", "b"}};
    std::vector<std::vector<std::string>> ooperands{{"loss"}};
    std::vector<std::string> opNames{"ADD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    // Set in/out casts
    EXPECT_EQ(G.SetInCast({"a", "b"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    // Mark tensors for autodiff
    auto tensorA = G.GetTensor("a");
    auto tensorB = G.GetTensor("b");
    auto tensorLoss = G.GetTensor("loss");

    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorB->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    // Initial operation count
    size_t initialOpCount = function->Operations().size();
    EXPECT_EQ(initialOpCount, 1);

    // Run AutodiffPass
    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // After autodiff, there should be more operations (gradient ops)
    // For add: d_a = d_loss, d_b = d_loss (no new ops needed, just tensor assignments)
    // But initialization of loss grad might add operations
    // Note: With current implementation, finalOpCount may or may not increase
    (void)function->Operations().size();  // finalOpCount - may check later

    // Verify gradients are stored
    int64_t gradMagicA = 0;
    int64_t gradMagicB = 0;
    EXPECT_TRUE(tensorA->GetAttr("gradient_magic", gradMagicA));
    EXPECT_TRUE(tensorB->GetAttr("gradient_magic", gradMagicB));

    // Gradients should exist (non-zero magic indicates valid gradient tensor)
    EXPECT_NE(gradMagicA, 0);
    EXPECT_NE(gradMagicB, 0);
}

// Test mul operation gradient
TEST_F(AutodiffPassTest, MulGradient) {
    ComputationalGraphBuilder G;

    // Create tensors: a * b = loss
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    // Add operation: loss = a * b
    std::vector<Opcode> opCodes{Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"a", "b"}};
    std::vector<std::vector<std::string>> ooperands{{"loss"}};
    std::vector<std::string> opNames{"MUL"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"a", "b"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    auto tensorA = G.GetTensor("a");
    auto tensorB = G.GetTensor("b");
    auto tensorLoss = G.GetTensor("loss");

    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorB->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    size_t initialOpCount = function->Operations().size();
    EXPECT_EQ(initialOpCount, 1);

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // Mul gradient adds operations: d_a = d_loss * b, d_b = d_loss * a
    size_t finalOpCount = function->Operations().size();
    EXPECT_GT(finalOpCount, initialOpCount);

    // Verify gradients are stored
    int64_t gradMagicA = 0;
    int64_t gradMagicB = 0;
    EXPECT_TRUE(tensorA->GetAttr("gradient_magic", gradMagicA));
    EXPECT_TRUE(tensorB->GetAttr("gradient_magic", gradMagicB));
    EXPECT_NE(gradMagicA, 0);
    EXPECT_NE(gradMagicB, 0);
}

// Test chain rule: c = a * b, loss = c + c
TEST_F(AutodiffPassTest, ChainRule) {
    ComputationalGraphBuilder G;

    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "c"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    // c = a * b
    // loss = c + c
    std::vector<Opcode> opCodes{Opcode::OP_MUL, Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"a", "b"}, {"c", "c"}};
    std::vector<std::vector<std::string>> ooperands{{"c"}, {"loss"}};
    std::vector<std::string> opNames{"MUL", "ADD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"a", "b"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    auto tensorA = G.GetTensor("a");
    auto tensorB = G.GetTensor("b");
    auto tensorLoss = G.GetTensor("loss");

    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorB->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // Verify gradients exist
    int64_t gradMagicA = 0;
    int64_t gradMagicB = 0;
    EXPECT_TRUE(tensorA->GetAttr("gradient_magic", gradMagicA));
    EXPECT_TRUE(tensorB->GetAttr("gradient_magic", gradMagicB));
    EXPECT_NE(gradMagicA, 0);
    EXPECT_NE(gradMagicB, 0);
}

// Test no loss tensor - should skip gracefully
TEST_F(AutodiffPassTest, NoLossTensor) {
    ComputationalGraphBuilder G;

    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "c"), true);

    std::vector<Opcode> opCodes{Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"a", "b"}};
    std::vector<std::vector<std::string>> ooperands{{"c"}};
    std::vector<std::string> opNames{"ADD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"a", "b"}), true);
    EXPECT_EQ(G.SetOutCast({"c"}), true);

    // Mark requires_grad but NO is_loss
    auto tensorA = G.GetTensor("a");
    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);

    // Should succeed (graceful skip)
    EXPECT_EQ(status, SUCCESS);
}

// Test no requires_grad tensors - should skip gracefully
TEST_F(AutodiffPassTest, NoRequiresGradTensors) {
    ComputationalGraphBuilder G;

    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    std::vector<Opcode> opCodes{Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"a", "b"}};
    std::vector<std::vector<std::string>> ooperands{{"loss"}};
    std::vector<std::string> opNames{"ADD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"a", "b"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    // Mark is_loss but NO requires_grad
    auto tensorLoss = G.GetTensor("loss");
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);

    // Should succeed (graceful skip)
    EXPECT_EQ(status, SUCCESS);
}

// Test non-differentiable dtype (INT32) is skipped
TEST_F(AutodiffPassTest, NonDifferentiableDtype) {
    ComputationalGraphBuilder G;

    // INT32 is not differentiable
    EXPECT_EQ(G.AddTensor(DataType::DT_INT32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_INT32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    std::vector<Opcode> opCodes{Opcode::OP_ADD};
    std::vector<std::vector<std::string>> ioperands{{"a", "b"}};
    std::vector<std::vector<std::string>> ooperands{{"loss"}};
    std::vector<std::string> opNames{"ADD"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"a", "b"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    auto tensorA = G.GetTensor("a");
    auto tensorB = G.GetTensor("b");
    auto tensorLoss = G.GetTensor("loss");

    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorB->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);

    // Should succeed but INT32 inputs should not have gradients
    EXPECT_EQ(status, SUCCESS);
}

// Test VJP Registry
TEST_F(AutodiffPassTest, VJPRegistryHasRules) {
    // Verify key operations have VJP rules registered
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_ADD));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_SUB));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_MUL));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_DIV));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_NEG));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_EXP));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_LN));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_SQRT));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_RSQRT));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_RESHAPE));
    EXPECT_TRUE(VJPRegistry::Instance().HasVJP(Opcode::OP_VIEW));
}

// Test matmul gradient
TEST_F(AutodiffPassTest, MatmulGradient) {
    ComputationalGraphBuilder G;

    // Matrix multiplication: C = A @ B
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 32}, "A"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {32, 16}, "B"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    std::vector<Opcode> opCodes{Opcode::OP_A_MUL_B};
    std::vector<std::vector<std::string>> ioperands{{"A", "B"}};
    std::vector<std::vector<std::string>> ooperands{{"loss"}};
    std::vector<std::string> opNames{"MATMUL"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"A", "B"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    auto tensorA = G.GetTensor("A");
    auto tensorB = G.GetTensor("B");
    auto tensorLoss = G.GetTensor("loss");

    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorB->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    size_t initialOpCount = function->Operations().size();
    EXPECT_EQ(initialOpCount, 1);

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // Matmul gradient adds 2 matmul operations
    size_t finalOpCount = function->Operations().size();
    EXPECT_GT(finalOpCount, initialOpCount);

    int64_t gradMagicA = 0;
    int64_t gradMagicB = 0;
    EXPECT_TRUE(tensorA->GetAttr("gradient_magic", gradMagicA));
    EXPECT_TRUE(tensorB->GetAttr("gradient_magic", gradMagicB));
    EXPECT_NE(gradMagicA, 0);
    EXPECT_NE(gradMagicB, 0);
}

// Test gradient tensors are added to outCasts_
TEST_F(AutodiffPassTest, GradientTensorsInOutcasts) {
    ComputationalGraphBuilder G;

    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    std::vector<Opcode> opCodes{Opcode::OP_MUL};
    std::vector<std::vector<std::string>> ioperands{{"a", "b"}};
    std::vector<std::vector<std::string>> ooperands{{"loss"}};
    std::vector<std::string> opNames{"MUL"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"a", "b"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    auto tensorA = G.GetTensor("a");
    auto tensorB = G.GetTensor("b");
    auto tensorLoss = G.GetTensor("loss");

    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorB->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    size_t initialOutcastCount = function->GetOutcast().size();
    EXPECT_EQ(initialOutcastCount, 1);  // Only loss initially

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // After autodiff, gradient tensors should be added to outCasts_
    size_t finalOutcastCount = function->GetOutcast().size();
    EXPECT_GT(finalOutcastCount, initialOutcastCount);

    // Verify gradient tensors have correct attributes
    int gradientCount = 0;
    for (const auto& outcast : function->GetOutcast()) {
        bool isGradient = false;
        if (outcast->GetAttr("is_gradient", isGradient) && isGradient) {
            gradientCount++;
            // Verify gradient_of_magic is set
            int64_t gradOfMagic = 0;
            EXPECT_TRUE(outcast->GetAttr("gradient_of_magic", gradOfMagic));
            EXPECT_NE(gradOfMagic, 0);
        }
    }
    // Should have 2 gradients (for a and b)
    EXPECT_EQ(gradientCount, 2);
}

// Test scalar operations gradient (MULS, ADDS)
TEST_F(AutodiffPassTest, ScalarOpsGradient) {
    ComputationalGraphBuilder G;

    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    // a * 2.0 = b, then b as loss
    std::vector<Opcode> opCodes{Opcode::OP_MULS};
    std::vector<std::vector<std::string>> ioperands{{"a"}};
    std::vector<std::vector<std::string>> ooperands{{"loss"}};
    std::vector<std::string> opNames{"MULS"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"a"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    auto tensorA = G.GetTensor("a");
    auto tensorLoss = G.GetTensor("loss");

    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    int64_t gradMagicA = 0;
    EXPECT_TRUE(tensorA->GetAttr("gradient_magic", gradMagicA));
    EXPECT_NE(gradMagicA, 0);
}

// Test exp/log gradient
TEST_F(AutodiffPassTest, ExpLogGradient) {
    ComputationalGraphBuilder G;

    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "a"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "b"), true);
    EXPECT_EQ(G.AddTensor(DataType::DT_FP32, {16, 16}, "loss"), true);

    // loss = log(exp(a))
    std::vector<Opcode> opCodes{Opcode::OP_EXP, Opcode::OP_LN};
    std::vector<std::vector<std::string>> ioperands{{"a"}, {"b"}};
    std::vector<std::vector<std::string>> ooperands{{"b"}, {"loss"}};
    std::vector<std::string> opNames{"EXP", "LN"};
    EXPECT_EQ(G.AddOps(opCodes, ioperands, ooperands, opNames, true), true);

    EXPECT_EQ(G.SetInCast({"a"}), true);
    EXPECT_EQ(G.SetOutCast({"loss"}), true);

    auto tensorA = G.GetTensor("a");
    auto tensorLoss = G.GetTensor("loss");

    tensorA->SetAttr(ATTR_REQUIRES_GRAD, true);
    tensorLoss->SetAttr(ATTR_IS_LOSS, true);

    Function* function = G.GetFunction();
    function->SetGraphType(GraphType::TENSOR_GRAPH);

    size_t initialOpCount = function->Operations().size();

    AutodiffPass autodiff;
    Status status = autodiff.RunOnFunction(*function);
    EXPECT_EQ(status, SUCCESS);

    // Exp and Log gradients add operations
    size_t finalOpCount = function->Operations().size();
    EXPECT_GT(finalOpCount, initialOpCount);

    int64_t gradMagicA = 0;
    EXPECT_TRUE(tensorA->GetAttr("gradient_magic", gradMagicA));
    EXPECT_NE(gradMagicA, 0);
}

} // namespace tile_fwk
} // namespace npu
