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

TEST_F(IRVerifierTest, TestVerifySSA_ScalarValue_ValidProgram) {
    FunctionSignature sig;
    auto inputScalar = std::make_shared<ScalarValue>(DataType::FP32, "input", ScalarValueKind::Symbolic);
    sig.arguments = {inputScalar};

    CreateTestFunction(sig);

    // Create a simple scalar operation: output = neg(input)
    auto outputScalar = builder_->CreateScalar(*ctx_, DataType::FP32, "output");
    auto unaryScalarOp = builder_->CreateUnaryScalarOp(Opcode::OP_SCALAR_NEG, inputScalar, outputScalar);
    builder_->Emit(*ctx_, unaryScalarOp);

    FinishFunction();

    VerifyResult result = VerifySSA(module_);
    EXPECT_TRUE(result.passed) << "Valid program with ScalarValue should pass SSA verification";
}

TEST_F(IRVerifierTest, TestVerifySSA_ScalarValue_InvalidProgram_MultipleDefinitions) {
    FunctionSignature sig;
    auto inputScalar = std::make_shared<ScalarValue>(DataType::FP32, "input", ScalarValueKind::Symbolic);
    sig.arguments = {inputScalar};

    CreateTestFunction(sig);

    // Create a ScalarValue that will be used as output multiple times (violates SSA)
    auto outputScalar = builder_->CreateScalar(*ctx_, DataType::FP32, "output");
    
    // First definition: output = neg(input)
    auto op1 = builder_->CreateUnaryScalarOp(Opcode::OP_SCALAR_NEG, inputScalar, outputScalar);
    builder_->Emit(*ctx_, op1);

    // Second definition: output = pos(input) - violates SSA (same ScalarValue defined twice)
    auto op2 = builder_->CreateUnaryScalarOp(Opcode::OP_SCALAR_POS, inputScalar, outputScalar);
    builder_->Emit(*ctx_, op2);

    FinishFunction();

    VerifyResult result = VerifySSA(module_);
    EXPECT_FALSE(result.passed) << "Program with multiple definitions of same ScalarValue should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("ScalarValue"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifySSA_MixedTileAndScalar_ValidProgram) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto inputScalar = std::make_shared<ScalarValue>(DataType::FP32, "scale", ScalarValueKind::Symbolic);
    sig.arguments = {inputTile, inputScalar};

    CreateTestFunction(sig);

    // Create Tile operations
    auto tempTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "temp");
    auto unaryOp = builder_->CreateUnaryOp(Opcode::OP_NEG, inputTile, tempTile);
    builder_->Emit(*ctx_, unaryOp);

    // Create Scalar operations
    auto tempScalar = builder_->CreateScalar(*ctx_, DataType::FP32, "temp_scalar");
    auto unaryScalarOp = builder_->CreateUnaryScalarOp(Opcode::OP_SCALAR_NEG, inputScalar, tempScalar);
    builder_->Emit(*ctx_, unaryScalarOp);

    // Create BinaryScalarMixOp
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    auto binaryScalarMixOp = builder_->CreateBinaryScalarMixOp(Opcode::OP_MULS, tempTile, tempScalar, outputTile);
    builder_->Emit(*ctx_, binaryScalarMixOp);

    FinishFunction();

    VerifyResult result = VerifySSA(module_);
    EXPECT_TRUE(result.passed) << "Valid program with mixed TileValue and ScalarValue should pass SSA verification";
}

TEST_F(IRVerifierTest, TestVerifySSA_MixedTileAndScalar_InvalidProgram) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto inputScalar = std::make_shared<ScalarValue>(DataType::FP32, "scale", ScalarValueKind::Symbolic);
    sig.arguments = {inputTile, inputScalar};

    CreateTestFunction(sig);

    // Create a ScalarValue that will be used as output multiple times (violates SSA)
    auto outputScalar = builder_->CreateScalar(*ctx_, DataType::FP32, "output");
    
    // First definition: output = neg(input)
    auto op1 = builder_->CreateUnaryScalarOp(Opcode::OP_SCALAR_NEG, inputScalar, outputScalar);
    builder_->Emit(*ctx_, op1);

    // Second definition: output = pos(input) - violates SSA
    auto op2 = builder_->CreateUnaryScalarOp(Opcode::OP_SCALAR_POS, inputScalar, outputScalar);
    builder_->Emit(*ctx_, op2);

    // Also create a TileValue with multiple definitions
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output_tile");
    auto op3 = builder_->CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder_->Emit(*ctx_, op3);
    auto op4 = builder_->CreateUnaryOp(Opcode::OP_ABS, inputTile, outputTile);
    builder_->Emit(*ctx_, op4);

    FinishFunction();

    VerifyResult result = VerifySSA(module_);
    EXPECT_FALSE(result.passed) << "Program with multiple definitions of both TileValue and ScalarValue should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("ScalarValue"), std::string::npos);
    EXPECT_NE(result.errorMsg.find("TileValue"), std::string::npos);
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

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidMatmulLoadOp) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // MatmulLoadOp with matching input/output shapes
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    std::vector<ScalarValuePtr> offsets = {};
    auto matmulLoadOp = builder_->CreateMatmulLoadOp(Opcode::OP_L1_COPY_IN, inputTile, offsets, outputTile);
    builder_->Emit(*ctx_, matmulLoadOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid MatmulLoadOp should pass shape verification";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_InvalidMatmulLoadOp_MismatchedShapes) {
    FunctionSignature sig;
    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> outputShape = {64, 128}; // Different shape
    auto inputTile = std::make_shared<TileValue>(inputShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // MatmulLoadOp with mismatched shapes
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    std::vector<ScalarValuePtr> offsets = {};
    auto matmulLoadOp = builder_->CreateMatmulLoadOp(Opcode::OP_L1_COPY_IN, inputTile, offsets, outputTile);
    builder_->Emit(*ctx_, matmulLoadOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_FALSE(result.passed) << "MatmulLoadOp with mismatched shapes should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("MatmulLoadOp"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidMatmulExtractOp_NoTranspose) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // MatmulExtractOp without transpose (OP_L1_TO_L0A)
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    std::vector<ScalarValuePtr> offsets = {};
    auto matmulExtractOp = builder_->CreateMatmulExtractOp(Opcode::OP_L1_TO_L0A, inputTile, offsets, outputTile);
    builder_->Emit(*ctx_, matmulExtractOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid MatmulExtractOp without transpose should pass";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidMatmulExtractOp_WithTranspose) {
    FunctionSignature sig;
    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> outputShape = {64, 128}; // Transposed shape
    auto inputTile = std::make_shared<TileValue>(inputShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // MatmulExtractOp with transpose (OP_L1_TO_L0_AT - transpose A)
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    std::vector<ScalarValuePtr> offsets = {};
    auto matmulExtractOp = builder_->CreateMatmulExtractOp(Opcode::OP_L1_TO_L0_AT, inputTile, offsets, outputTile);
    builder_->Emit(*ctx_, matmulExtractOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid MatmulExtractOp with transpose should pass";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_InvalidMatmulExtractOp_WrongTransposeShape) {
    FunctionSignature sig;
    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> outputShape = {128, 64}; // Should be {64, 128} for transpose
    auto inputTile = std::make_shared<TileValue>(inputShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // MatmulExtractOp with transpose opcode but wrong output shape
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    std::vector<ScalarValuePtr> offsets = {};
    auto matmulExtractOp = builder_->CreateMatmulExtractOp(Opcode::OP_L1_TO_L0_AT, inputTile, offsets, outputTile);
    builder_->Emit(*ctx_, matmulExtractOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_FALSE(result.passed) << "MatmulExtractOp with wrong transpose shape should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("MatmulExtractOp"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidMatmulMmadOp) {
    FunctionSignature sig;
    // Matrix multiplication: A [M, K] * B [K, N] = C [M, N]
    std::vector<int64_t> lhsShape = {128, 64};  // [M, K]
    std::vector<int64_t> rhsShape = {64, 256};  // [K, N]
    std::vector<int64_t> outputShape = {128, 256}; // [M, N]
    auto lhsTile = std::make_shared<TileValue>(lhsShape, DataType::FP32, "lhs");
    auto rhsTile = std::make_shared<TileValue>(rhsShape, DataType::FP32, "rhs");
    sig.arguments = {lhsTile, rhsTile};

    CreateTestFunction(sig);

    // MatmulMmadOp with correct shapes
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    auto matmulMmadOp = builder_->CreateMatmulMmadOp(Opcode::OP_A_MUL_B, lhsTile, rhsTile, outputTile);
    builder_->Emit(*ctx_, matmulMmadOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid MatmulMmadOp should pass shape verification";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_InvalidMatmulMmadOp_KDimensionMismatch) {
    FunctionSignature sig;
    // K dimension mismatch: A [M, K1] * B [K2, N] where K1 != K2
    std::vector<int64_t> lhsShape = {128, 64};  // [M, K1=64]
    std::vector<int64_t> rhsShape = {32, 256};  // [K2=32, N] - K mismatch!
    std::vector<int64_t> outputShape = {128, 256};
    auto lhsTile = std::make_shared<TileValue>(lhsShape, DataType::FP32, "lhs");
    auto rhsTile = std::make_shared<TileValue>(rhsShape, DataType::FP32, "rhs");
    sig.arguments = {lhsTile, rhsTile};

    CreateTestFunction(sig);

    // MatmulMmadOp with K dimension mismatch
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    auto matmulMmadOp = builder_->CreateMatmulMmadOp(Opcode::OP_A_MUL_B, lhsTile, rhsTile, outputTile);
    builder_->Emit(*ctx_, matmulMmadOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_FALSE(result.passed) << "MatmulMmadOp with K dimension mismatch should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("K dimension mismatch"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifyOpShape_InvalidMatmulMmadOp_WrongOutputShape) {
    FunctionSignature sig;
    std::vector<int64_t> lhsShape = {128, 64};
    std::vector<int64_t> rhsShape = {64, 256};
    std::vector<int64_t> outputShape = {64, 128}; // Wrong! Should be [128, 256]
    auto lhsTile = std::make_shared<TileValue>(lhsShape, DataType::FP32, "lhs");
    auto rhsTile = std::make_shared<TileValue>(rhsShape, DataType::FP32, "rhs");
    sig.arguments = {lhsTile, rhsTile};

    CreateTestFunction(sig);

    // MatmulMmadOp with wrong output shape
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    auto matmulMmadOp = builder_->CreateMatmulMmadOp(Opcode::OP_A_MUL_B, lhsTile, rhsTile, outputTile);
    builder_->Emit(*ctx_, matmulMmadOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_FALSE(result.passed) << "MatmulMmadOp with wrong output shape should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("MatmulMmadOp"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidMatmulAccOp) {
    FunctionSignature sig;
    std::vector<int64_t> lhsShape = {128, 64};
    std::vector<int64_t> rhsShape = {64, 256};
    std::vector<int64_t> outputShape = {128, 256};
    auto lhsTile = std::make_shared<TileValue>(lhsShape, DataType::FP32, "lhs");
    auto rhsTile = std::make_shared<TileValue>(rhsShape, DataType::FP32, "rhs");
    sig.arguments = {lhsTile, rhsTile};

    CreateTestFunction(sig);

    // MatmulAccOp with correct shapes
    auto outputTile = builder_->CreateTile(*ctx_, outputShape, DataType::FP32, "output");
    auto matmulAccOp = builder_->CreateMatmulAccOp(Opcode::OP_A_MULACC_B, lhsTile, rhsTile, outputTile);
    builder_->Emit(*ctx_, matmulAccOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid MatmulAccOp should pass shape verification";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidMatmulStoreOp) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // MatmulStoreOp with matching shapes
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    std::vector<ScalarValuePtr> offsets = {};
    auto matmulStoreOp = builder_->CreateMatmulStoreOp(Opcode::OP_L0C_COPY_OUT, inputTile, offsets, outputTile);
    builder_->Emit(*ctx_, matmulStoreOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid MatmulStoreOp should pass shape verification";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidMatmulBiasOp) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // MatmulBiasOp with matching shapes
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    std::vector<ScalarValuePtr> offsets = {};
    auto matmulBiasOp = builder_->CreateMatmulBiasOp(Opcode::OP_L1_TO_BT, inputTile, offsets, outputTile);
    builder_->Emit(*ctx_, matmulBiasOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid MatmulBiasOp should pass shape verification";
}

TEST_F(IRVerifierTest, TestVerifyOpShape_ValidMatmulQuantOp) {
    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    CreateTestFunction(sig);

    // MatmulQuantOp with matching shapes
    auto outputTile = builder_->CreateTile(*ctx_, tileShape, DataType::FP32, "output");
    std::vector<ScalarValuePtr> offsets = {};
    auto matmulQuantOp = builder_->CreateMatmulQuantOp(Opcode::OP_L1_TO_FIX_QUANT_PRE, inputTile, offsets, outputTile);
    builder_->Emit(*ctx_, matmulQuantOp);

    FinishFunction();

    VerifyResult result = VerifyOpShape(module_);
    EXPECT_TRUE(result.passed) << "Valid MatmulQuantOp should pass shape verification";
}
