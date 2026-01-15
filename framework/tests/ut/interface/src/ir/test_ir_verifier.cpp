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
 * \file test_ir_verifier.cpp
 * \brief Tests for IR verifier
 */

#include "gtest/gtest.h"
#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/verifier/verifier.h"
#include "ir/verifier/ssa_verify.h"
#include "ir/verifier/shape_verify.h"
#include "ir/opcode.h"
#include "ir/program.h"

#include <memory>

using namespace pto;

// ===== Verifier Class Tests =====

class IRVerifierTest : public ::testing::Test {
protected:
    void SetUp() override {
        module_ = std::make_shared<ProgramModule>("main");
        builder_ = std::make_unique<IRBuilder>();
        ctx_ = std::make_unique<IRBuilderContext>();
    }

    void TearDown() override {
        ctx_.reset();
        builder_.reset();
        module_.reset();
    }

    FunctionPtr CreateTestFunction(const FunctionSignature &sig, const std::string &name = "test_func") {
        auto func = builder_->CreateFunction(name, FunctionKind::Block, sig);
        module_->AddFunction(func);
        builder_->EnterFunctionBody(*ctx_, func);
        return func;
    }

    void FinishFunction() {
        builder_->CreateReturn(*ctx_, {});
        ctx_->PopScope();
    }

    ProgramModulePtr module_;
    std::unique_ptr<IRBuilder> builder_;
    std::unique_ptr<IRBuilderContext> ctx_;
};

TEST_F(IRVerifierTest, TestVerifySSASingleInput_ValidProgram) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // Create a simple operation: output = input (using unary op as identity)
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    auto unaryOp = builder_->CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder_->Emit(*ctx_, unaryOp);

    FinishFunction();

    VerifyResult result = VerifySSA(module_);
    EXPECT_TRUE(result.passed) << "Valid program should pass SSA verification";
}

TEST_F(IRVerifierTest, TestVerifySSASingleInput_InvalidProgram_MultipleDefinitions) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // Create a TileValue that will be used as output multiple times (violates SSA)
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    
    // First definition: output = neg(input)
    auto op1 = builder_->CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder_->Emit(*ctx_, op1);

    // Second definition: output = abs(input) - violates SSA (same TileValue defined twice)
    auto op2 = builder_->CreateUnaryOp(Opcode::OP_ABS, inputTile, outputTile);
    builder_->Emit(*ctx_, op2);

    FinishFunction();

    VerifyResult result = VerifySSA(module_);
    EXPECT_FALSE(result.passed) << "Program with multiple definitions of same TileValue should fail";
    EXPECT_FALSE(result.errorMsg.empty());
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidUnaryOp) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // UnaryOp with matching shapes
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    auto unaryOp = builder_->CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder_->Emit(*ctx_, unaryOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid UnaryOp should pass shape verification";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_InvalidUnaryOp_MismatchedShapes) {
    FunctionSignature sig;
    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> outputShape = {64, 128}; // Different shape
    auto inputTile = std::make_shared<TileValue>(inputShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // UnaryOp with mismatched shapes
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    auto unaryOp = builder_->CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder_->Emit(*ctx_, unaryOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_FALSE(result.passed) << "UnaryOp with mismatched shapes should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("UnaryOp"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidBinaryOp_MatchingShapes) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto input1 = std::make_shared<TileValue>(tileShape, DataType::FP32, "input1");
    auto input2 = std::make_shared<TileValue>(tileShape, DataType::FP32, "input2");
    sig.arguments = {input1, input2};

    CreateTestFunction(sig);

    // BinaryOp with all matching shapes
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    auto binaryOp = builder_->CreateBinaryOp(Opcode::OP_ADD, input1, input2, outputTile);
    builder_->Emit(*ctx_, binaryOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "BinaryOp with matching shapes should pass";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidBinaryOp_Broadcast) {
    FunctionSignature sig;
    std::vector<int64_t> lhsShape = {1, 64}; // Broadcastable shape
    std::vector<int64_t> rhsShape = {128, 64};
    std::vector<int64_t> outputShape = {128, 64};
    auto input1 = std::make_shared<TileValue>(lhsShape, DataType::FP32, "input1");
    auto input2 = std::make_shared<TileValue>(rhsShape, DataType::FP32, "input2");
    sig.arguments = {input1, input2};

    CreateTestFunction(sig);

    // BinaryOp with broadcastable shape (one input has dimension 1)
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    auto binaryOp = builder_->CreateBinaryOp(Opcode::OP_ADD, input1, input2, outputTile);
    builder_->Emit(*ctx_, binaryOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "BinaryOp with broadcastable shape should pass";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_InvalidBinaryOp_IncompatibleShapes) {
    FunctionSignature sig;
    std::vector<int64_t> lhsShape = {128, 32}; // Incompatible
    std::vector<int64_t> rhsShape = {64, 64};
    std::vector<int64_t> outputShape = {64, 64};
    auto input1 = std::make_shared<TileValue>(lhsShape, DataType::FP32, "input1");
    auto input2 = std::make_shared<TileValue>(rhsShape, DataType::FP32, "input2");
    sig.arguments = {input1, input2};

    CreateTestFunction(sig);

    // BinaryOp with incompatible shapes
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    auto binaryOp = builder_->CreateBinaryOp(Opcode::OP_SUB, input1, input2, outputTile);
    builder_->Emit(*ctx_, binaryOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_FALSE(result.passed) << "BinaryOp with incompatible shapes should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("BinaryOp"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidBinaryScalarMixOp) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto scalar = std::make_shared<ScalarValue>(DataType::FP32, "scale", ScalarValueKind::Symbolic);
    sig.arguments = {inputTile, scalar};

    CreateTestFunction(sig);

    // BinaryScalarMixOp with matching shapes
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    auto binaryScalarMixOp = builder_->CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scalar, outputTile);
    builder_->Emit(*ctx_, binaryScalarMixOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid BinaryScalarMixOp should pass shape verification";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_InvalidBinaryScalarMixOp_MismatchedShapes) {
    FunctionSignature sig;
    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> outputShape = {64, 128}; // Different shape
    auto inputTile = std::make_shared<TileValue>(inputShape, DataType::FP32, "input");
    auto scalar = std::make_shared<ScalarValue>(DataType::FP32, "scale", ScalarValueKind::Symbolic);
    sig.arguments = {inputTile, scalar};

    CreateTestFunction(sig);

    // BinaryScalarMixOp with mismatched shapes
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    auto binaryScalarMixOp = builder_->CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scalar, outputTile);
    builder_->Emit(*ctx_, binaryScalarMixOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_FALSE(result.passed) << "BinaryScalarMixOp with mismatched shapes should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("BinaryScalarMixOp"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifyOpShape_MixedOperations) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // Mix of valid operations
    auto temp1 = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "temp1");
    auto unaryOp = builder_->CreateUnaryOp(Opcode::OP_NEG, inputTile, temp1);
    builder_->Emit(*ctx_, unaryOp);

    auto temp2 = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "temp2");
    auto binaryOp = builder_->CreateBinaryOp(Opcode::OP_ADD, temp1, temp1, temp2);
    builder_->Emit(*ctx_, binaryOp);

    auto scalar = builder_->CreateConst(*ctx_, 2.0, "scale");
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    auto binaryScalarMixOp = builder_->CreateBinaryScalarMixOp(Opcode::OP_MULS, temp2, scalar, outputTile);
    builder_->Emit(*ctx_, binaryScalarMixOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Program with valid mixed operations should pass";
}
