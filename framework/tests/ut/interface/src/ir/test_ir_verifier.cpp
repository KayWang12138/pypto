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

using namespace pto;

// ===== Verifier Class Tests =====

TEST(IRVerifierTest, TestVerifySSASingleInput_ValidProgram) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // Create a simple operation: output = input (using unary op as identity)
    auto outputTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    auto unaryOp = builder.CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder.Emit(ctx, unaryOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifySSA(module);
    EXPECT_TRUE(result.passed) << "Valid program should pass SSA verification";
}

TEST(IRVerifierTest, TestVerifySSASingleInput_InvalidProgram_MultipleDefinitions) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // Create a TileValue that will be used as output multiple times (violates SSA)
    auto outputTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    
    // First definition: output = neg(input)
    auto op1 = builder.CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder.Emit(ctx, op1);

    // Second definition: output = abs(input) - violates SSA (same TileValue defined twice)
    auto op2 = builder.CreateUnaryOp(Opcode::OP_ABS, inputTile, outputTile);
    builder.Emit(ctx, op2);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifySSA(module);
    EXPECT_FALSE(result.passed) << "Program with multiple definitions of same TileValue should fail";
    EXPECT_FALSE(result.errorMsg.empty());
}

TEST(IRVerifierTest, TestVerifyOpShape_ValidUnaryOp) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // UnaryOp with matching shapes
    auto outputTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    auto unaryOp = builder.CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder.Emit(ctx, unaryOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifyOpShape(module);
    EXPECT_TRUE(result.passed) << "Valid UnaryOp should pass shape verification";
}

TEST(IRVerifierTest, TestVerifyOpShape_InvalidUnaryOp_MismatchedShapes) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> outputShape = {64, 128}; // Different shape
    auto inputTile = std::make_shared<TileValue>(inputShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // UnaryOp with mismatched shapes
    auto outputTile = builder.CreateTile(ctx, outputShape, DataType::FP32, "output");
    auto unaryOp = builder.CreateUnaryOp(Opcode::OP_NEG, inputTile, outputTile);
    builder.Emit(ctx, unaryOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifyOpShape(module);
    EXPECT_FALSE(result.passed) << "UnaryOp with mismatched shapes should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("UnaryOp"), std::string::npos);
}

TEST(IRVerifierTest, TestVerifyOpShape_ValidBinaryOp_MatchingShapes) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto input1 = std::make_shared<TileValue>(tileShape, DataType::FP32, "input1");
    auto input2 = std::make_shared<TileValue>(tileShape, DataType::FP32, "input2");
    sig.arguments = {input1, input2};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // BinaryOp with all matching shapes
    auto outputTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    auto binaryOp = builder.CreateBinaryOp(Opcode::OP_ADD, input1, input2, outputTile);
    builder.Emit(ctx, binaryOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifyOpShape(module);
    EXPECT_TRUE(result.passed) << "BinaryOp with matching shapes should pass";
}

TEST(IRVerifierTest, TestVerifyOpShape_ValidBinaryOp_Broadcast) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> lhsShape = {1, 64}; // Broadcastable shape
    std::vector<int64_t> rhsShape = {128, 64};
    std::vector<int64_t> outputShape = {128, 64};
    auto input1 = std::make_shared<TileValue>(lhsShape, DataType::FP32, "input1");
    auto input2 = std::make_shared<TileValue>(rhsShape, DataType::FP32, "input2");
    sig.arguments = {input1, input2};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // BinaryOp with broadcastable shape (one input has dimension 1)
    auto outputTile = builder.CreateTile(ctx, outputShape, DataType::FP32, "output");
    auto binaryOp = builder.CreateBinaryOp(Opcode::OP_ADD, input1, input2, outputTile);
    builder.Emit(ctx, binaryOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifyOpShape(module);
    EXPECT_TRUE(result.passed) << "BinaryOp with broadcastable shape should pass";
}

TEST(IRVerifierTest, TestVerifyOpShape_InvalidBinaryOp_IncompatibleShapes) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> lhsShape = {128, 32}; // Incompatible
    std::vector<int64_t> rhsShape = {128, 64};
    std::vector<int64_t> outputShape = {128, 64};
    auto input1 = std::make_shared<TileValue>(lhsShape, DataType::FP32, "input1");
    auto input2 = std::make_shared<TileValue>(rhsShape, DataType::FP32, "input2");
    sig.arguments = {input1, input2};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // BinaryOp with incompatible shapes
    auto outputTile = builder.CreateTile(ctx, outputShape, DataType::FP32, "output");
    auto binaryOp = builder.CreateBinaryOp(Opcode::OP_ADD, input1, input2, outputTile);
    builder.Emit(ctx, binaryOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifyOpShape(module);
    EXPECT_FALSE(result.passed) << "BinaryOp with incompatible shapes should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("BinaryOp"), std::string::npos);
}

TEST(IRVerifierTest, TestVerifyOpShape_ValidBinaryScalarMixOp) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto scalar = std::make_shared<ScalarValue>(DataType::FP32, "scale", ScalarValueKind::Symbolic);
    sig.arguments = {inputTile, scalar};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // BinaryScalarMixOp with matching shapes
    auto outputTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    auto binaryScalarMixOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scalar, outputTile);
    builder.Emit(ctx, binaryScalarMixOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifyOpShape(module);
    EXPECT_TRUE(result.passed) << "Valid BinaryScalarMixOp should pass shape verification";
}

TEST(IRVerifierTest, TestVerifyOpShape_InvalidBinaryScalarMixOp_MismatchedShapes) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> inputShape = {128, 64};
    std::vector<int64_t> outputShape = {64, 128}; // Different shape
    auto inputTile = std::make_shared<TileValue>(inputShape, DataType::FP32, "input");
    auto scalar = std::make_shared<ScalarValue>(DataType::FP32, "scale", ScalarValueKind::Symbolic);
    sig.arguments = {inputTile, scalar};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // BinaryScalarMixOp with mismatched shapes
    auto outputTile = builder.CreateTile(ctx, outputShape, DataType::FP32, "output");
    auto binaryScalarMixOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scalar, outputTile);
    builder.Emit(ctx, binaryScalarMixOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifyOpShape(module);
    EXPECT_FALSE(result.passed) << "BinaryScalarMixOp with mismatched shapes should fail";
    EXPECT_FALSE(result.errorMsg.empty());
    EXPECT_NE(result.errorMsg.find("BinaryScalarMixOp"), std::string::npos);
}

TEST(IRVerifierTest, TestVerifyOpShape_MixedOperations) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 64};
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {inputTile};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // Mix of valid operations
    auto temp1 = builder.CreateTile(ctx, tileShape, DataType::FP32, "temp1");
    auto unaryOp = builder.CreateUnaryOp(Opcode::OP_NEG, inputTile, temp1);
    builder.Emit(ctx, unaryOp);

    auto temp2 = builder.CreateTile(ctx, tileShape, DataType::FP32, "temp2");
    auto binaryOp = builder.CreateBinaryOp(Opcode::OP_ADD, temp1, temp1, temp2);
    builder.Emit(ctx, binaryOp);

    auto scalar = builder.CreateConst(ctx, 2.0, "scale");
    auto outputTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "output");
    auto binaryScalarMixOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, temp2, scalar, outputTile);
    builder.Emit(ctx, binaryScalarMixOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    VerifyResult result = VerifyOpShape(module);
    EXPECT_TRUE(result.passed) << "Program with valid mixed operations should pass";
}
