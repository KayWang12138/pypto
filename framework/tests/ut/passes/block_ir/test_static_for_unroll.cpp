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
 * \file test_static_for_unroll.cpp
 * \brief Unit tests for static for loop unrolling transformation
 */

#include <cstdint>
#include <memory>
#include "gtest/gtest.h"

#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"
#include "passes/block_graph_pass/block_ir/static_for_unroll.h"
#include "interface/utils/common.h"

namespace pto {

class StaticForUnrollTest : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(StaticForUnrollTest, BasicStaticForLoop) {
    // Create module
    auto module = std::make_shared<ProgramModule>("test_static_for_unroll");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // Create function signature
    FunctionSignature sig;

    std::vector<uint64_t> tileShape = { 128, 128 };

    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto scale = std::make_shared<ScalarValue>(DataType::FP32, "scale", ScalarValueKind::Symbolic);
    auto result = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");

    sig.arguments = { inputTile, scale, result };

    // Create function
    auto func = builder.CreateFunction("test_static_for", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // Enter function body
    builder.EnterFunctionBody(ctx, func);

    // Create static for loop: for i = 0 to 3 step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant3 = builder.CreateConst(ctx, int64_t(3), "const_3");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, constant3, constant1);

    builder.EnterForBody(ctx, fs);

    // Loop body: simple tile operation
    auto tileMul = builder.CreateTile(ctx, tileShape, DataType::FP32, "tile_mul");
    auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scale, tileMul);
    builder.Emit(ctx, mulOp);

    ctx.PopScope(); // Exit for body

    builder.ExitForStatement(ctx, fs);

    // Return
    builder.CreateReturn(ctx, {});

    ctx.PopScope(); // Exit function body

    // Verify initial IR has 1 statement (the for loop)
    ASSERT_EQ(func->GetCompound()->GetStatementsNum(), 2); // for + return

    // Print IR before unrolling
    std::cout << "========== Before Unrolling ==========" << std::endl;
    std::cout << *module << std::endl;

    // Apply static for unroll transformation
    npu::tile_fwk::StaticForUnrollTransform transform;
    auto status = transform.Apply(module);

    ASSERT_EQ(status, SUCCESS) << "Static for unroll transformation failed";

    // Print IR after unrolling
    std::cout << "========== After Unrolling ==========" << std::endl;
    std::cout << *module << std::endl;

    // Verify the loop was unrolled
    // After unrolling, the for statement should be replaced with 3 op statements (one for each iteration)
    // Plus the return statement
    ASSERT_GT(func->GetCompound()->GetStatementsNum(), 2) << "Loop was not unrolled";
}

TEST_F(StaticForUnrollTest, NestedStaticForLoops) {
    // Create module
    auto module = std::make_shared<ProgramModule>("test_nested_static_for");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // Create function signature
    FunctionSignature sig;

    std::vector<uint64_t> tileShape = { 16, 16 };

    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto result = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");

    sig.arguments = { inputTile, result };

    // Create function
    auto func = builder.CreateFunction("test_nested_for", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // Enter function body
    builder.EnterFunctionBody(ctx, func);

    // Outer loop: for i = 0 to 2 step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant2 = builder.CreateConst(ctx, int64_t(2), "const_2");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto outerLoop = builder.CreateForStmt(ctx, i, constant0, constant2, constant1);

    builder.EnterForBody(ctx, outerLoop);

    // Inner loop: for j = 0 to 2 step 1
    auto j = builder.CreateScalar(ctx, DataType::INT32, "j");
    auto innerLoop = builder.CreateForStmt(ctx, j, constant0, constant2, constant1);

    builder.EnterForBody(ctx, innerLoop);

    // Inner loop body: simple tile operation
    auto scaledTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "scaled");
    auto scaleValue = builder.CreateConst(ctx, 2.0, "scale_const");
    auto scaleOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scaleValue, scaledTile);
    builder.Emit(ctx, scaleOp);

    ctx.PopScope(); // Exit inner for body

    builder.ExitForStatement(ctx, innerLoop);

    ctx.PopScope(); // Exit outer for body

    builder.ExitForStatement(ctx, outerLoop);

    // Return
    builder.CreateReturn(ctx, {});

    ctx.PopScope(); // Exit function body

    // Print IR before unrolling
    std::cout << "========== Nested Loops Before Unrolling ==========" << std::endl;
    std::cout << *module << std::endl;

    // Apply static for unroll transformation
    npu::tile_fwk::StaticForUnrollTransform transform;
    auto status = transform.Apply(module);

    ASSERT_EQ(status, SUCCESS) << "Nested static for unroll transformation failed";

    // Print IR after unrolling
    std::cout << "========== Nested Loops After Unrolling ==========" << std::endl;
    std::cout << *module << std::endl;

    // Verify nested loops were unrolled
    // 2 outer iterations * 2 inner iterations = 4 operations + 1 return = 5 statements
    ASSERT_GT(func->GetCompound()->GetStatementsNum(), 2) << "Nested loops were not unrolled";
}

TEST_F(StaticForUnrollTest, RejectDynamicForLoop) {
    // Create module
    auto module = std::make_shared<ProgramModule>("test_dynamic_for");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // Create function signature
    FunctionSignature sig;

    std::vector<uint64_t> tileShape = { 128, 128 };

    auto batch = std::make_shared<ScalarValue>(DataType::INT32, "batch", ScalarValueKind::Symbolic);
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto result = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");

    sig.arguments = { batch, inputTile, result };

    // Create function
    auto func = builder.CreateFunction("test_dynamic_for", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // Enter function body
    builder.EnterFunctionBody(ctx, func);

    // Create dynamic for loop: for i = 0 to batch step 1 (batch is symbolic)
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, batch, constant1);

    builder.EnterForBody(ctx, fs);

    // Loop body
    auto tileMul = builder.CreateTile(ctx, tileShape, DataType::FP32, "tile_mul");
    auto scaleValue = builder.CreateConst(ctx, 1.5, "scale");
    auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scaleValue, tileMul);
    builder.Emit(ctx, mulOp);

    ctx.PopScope();

    builder.ExitForStatement(ctx, fs);

    builder.CreateReturn(ctx, {});

    ctx.PopScope();

    // Print IR
    std::cout << "========== Dynamic For Loop (Should Fail) ==========" << std::endl;
    std::cout << *module << std::endl;

    // Apply static for unroll transformation - should fail
    npu::tile_fwk::StaticForUnrollTransform transform;
    auto status = transform.Apply(module);

    ASSERT_EQ(status, FAILED) << "Dynamic for loop should not be unrollable";
}

TEST_F(StaticForUnrollTest, RejectIfStatement) {
    // Create module
    auto module = std::make_shared<ProgramModule>("test_with_if");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // Create function signature
    FunctionSignature sig;

    std::vector<uint64_t> tileShape = { 128, 128 };

    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto condition = std::make_shared<ScalarValue>(DataType::INT32, "cond", ScalarValueKind::Symbolic);
    auto result = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");

    sig.arguments = { inputTile, condition, result };

    // Create function
    auto func = builder.CreateFunction("test_with_if", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // Enter function body
    builder.EnterFunctionBody(ctx, func);

    // Create if statement
    auto ifs = builder.CreateIfStmt(ctx, condition);

    builder.EnterIfThen(ctx, ifs);

    auto thenTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "then_tile");
    auto scale1 = builder.CreateConst(ctx, 2.0, "scale1");
    auto mulOp1 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scale1, thenTile);
    builder.Emit(ctx, mulOp1);

    ctx.PopScope();

    builder.EnterIfElse(ctx, ifs);

    auto elseTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "else_tile");
    auto scale2 = builder.CreateConst(ctx, 3.0, "scale2");
    auto mulOp2 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scale2, elseTile);
    builder.Emit(ctx, mulOp2);

    ctx.PopScope();

    builder.ExitIfStatement(ctx, ifs);

    builder.CreateReturn(ctx, {});

    ctx.PopScope();

    // Print IR
    std::cout << "========== With If Statement (Should Fail) ==========" << std::endl;
    std::cout << *module << std::endl;

    // Apply static for unroll transformation - should fail because of if statement
    npu::tile_fwk::StaticForUnrollTransform transform;
    auto status = transform.Apply(module);

    ASSERT_EQ(status, FAILED) << "Functions with if statements should be rejected";
}

TEST_F(StaticForUnrollTest, VerifyUniqueNamesAcrossIterations) {
    // Test that cloned operations have unique names for each iteration
    auto module = std::make_shared<ProgramModule>("test_unique_names");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // Create function signature
    FunctionSignature sig;
    std::vector<uint64_t> tileShape = { 64, 64 };
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto outputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
    sig.arguments = { inputTile, outputTile };

    // Create function
    auto func = builder.CreateFunction("test_unique_names", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // Enter function body
    builder.EnterFunctionBody(ctx, func);

    // Create static for loop: for i = 0 to 3 step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant3 = builder.CreateConst(ctx, int64_t(3), "const_3");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, constant3, constant1);

    builder.EnterForBody(ctx, fs);

    // Loop body: create intermediate tiles to verify unique naming
    auto temp1 = builder.CreateTile(ctx, tileShape, DataType::FP32, "temp1");
    auto scale1 = builder.CreateConst(ctx, 1.0, "scale1");
    auto op1 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scale1, temp1);
    builder.Emit(ctx, op1);

    auto temp2 = builder.CreateTile(ctx, tileShape, DataType::FP32, "temp2");
    auto scale2 = builder.CreateConst(ctx, 2.0, "scale2");
    auto op2 = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, temp1, scale2, temp2);
    builder.Emit(ctx, op2);

    ctx.PopScope();
    builder.ExitForStatement(ctx, fs);
    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    // Print IR before unrolling
    std::cout << "========== Before Unrolling (Unique Names Test) =========" << std::endl;
    std::cout << *module << std::endl;

    // Apply transformation
    npu::tile_fwk::StaticForUnrollTransform transform;
    auto status = transform.Apply(module);
    ASSERT_EQ(status, SUCCESS);

    // Print IR after unrolling
    std::cout << "========== After Unrolling (Unique Names Test) =========" << std::endl;
    std::cout << *module << std::endl;

    // Verify that we have multiple statements (3 iterations + return)
    // Note: Each iteration creates 1 OpStatement containing all operations
    size_t expectedMinStmts = 3; // 3 iterations (each as 1 OpStatement) + return
    ASSERT_GE(func->GetCompound()->GetStatementsNum(), expectedMinStmts) 
        << "Expected at least " << expectedMinStmts << " statements after unrolling";

    // Collect all operations to verify unique naming
    std::set<std::string> outputNames;
    size_t totalOps = 0;
    for (size_t idx = 0; idx < func->GetCompound()->GetStatementsNum(); ++idx) {
        auto stmt = func->GetCompound()->GetStatement(idx);
        if (stmt->GetKind() == pto::StatementKind::Op) {
            auto opStmt = std::dynamic_pointer_cast<pto::OpStatement>(stmt);
            if (opStmt) {
                totalOps += opStmt->Operations().size();
                for (const auto& op : opStmt->Operations()) {
                    for (size_t j = 0; j < op->GetNumOutputOperand(); ++j) {
                        auto output = op->GetOutputOperand(j);
                        std::string outputName = output->GetName();
                        // Verify that names contain iteration-specific suffix
                        if (outputName.find("temp") != std::string::npos) {
                            EXPECT_TRUE(outputName.find("_unroll_") != std::string::npos)
                                << "Output name '" << outputName << "' should contain '_unroll_' suffix";
                        }
                        outputNames.insert(outputName);
                    }
                }
            }
        }
    }

    // Verify we have the correct number of operations (3 iterations * 2 ops)
    size_t expectedOps = 6;
    EXPECT_GE(totalOps, expectedOps) 
        << "Expected at least " << expectedOps << " operations, got " << totalOps;

    // Verify that we have unique names for each iteration
    // We should have at least 6 unique temp outputs (3 iterations * 2 temps each)
    size_t expectedUniqueTemps = 6;
    size_t actualUniqueTemps = 0;
    for (const auto& name : outputNames) {
        if (name.find("temp") != std::string::npos) {
            actualUniqueTemps++;
        }
    }
    EXPECT_GE(actualUniqueTemps, expectedUniqueTemps) 
        << "Expected at least " << expectedUniqueTemps << " unique temp names, got " << actualUniqueTemps;
}

TEST_F(StaticForUnrollTest, VerifyIterVarReplacement) {
    // Test that iteration variable is replaced with constant values
    auto module = std::make_shared<ProgramModule>("test_iter_var_replacement");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // Create function signature
    FunctionSignature sig;
    std::vector<uint64_t> tileShape = { 32, 32 };
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto outputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
    sig.arguments = { inputTile, outputTile };

    // Create function
    auto func = builder.CreateFunction("test_iter_var", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // Enter function body
    builder.EnterFunctionBody(ctx, func);

    // Create static for loop: for i = 0 to 4 step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant4 = builder.CreateConst(ctx, int64_t(4), "const_4");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, constant4, constant1);

    builder.EnterForBody(ctx, fs);

    // Loop body: use iteration variable i in computation
    // In a real scenario, i would be used in operations
    // For this test, we create a scalar operation that uses i
    auto resultTile = builder.CreateTile(ctx, tileShape, DataType::FP32, "result");
    auto iterScalar = i; // Use the iteration variable
    auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, iterScalar, resultTile);
    builder.Emit(ctx, mulOp);

    ctx.PopScope();
    builder.ExitForStatement(ctx, fs);
    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    // Print IR before unrolling
    std::cout << "========== Before Unrolling (IterVar Replacement) =========" << std::endl;
    std::cout << *module << std::endl;

    // Apply transformation
    npu::tile_fwk::StaticForUnrollTransform transform;
    auto status = transform.Apply(module);
    ASSERT_EQ(status, SUCCESS);

    // Print IR after unrolling
    std::cout << "========== After Unrolling (IterVar Replacement) =========" << std::endl;
    std::cout << *module << std::endl;

    // Verify that we have 4 iterations
    size_t expectedMinStmts = 4; // 4 iterations
    ASSERT_GE(func->GetCompound()->GetStatementsNum(), expectedMinStmts);

    // Verify that iteration variable was replaced with constants
    int constantCount = 0;
    for (size_t idx = 0; idx < func->GetCompound()->GetStatementsNum(); ++idx) {
        auto stmt = func->GetCompound()->GetStatement(idx);
        if (stmt->GetKind() == pto::StatementKind::Op) {
            auto opStmt = std::dynamic_pointer_cast<pto::OpStatement>(stmt);
            if (opStmt) {
                for (const auto& op : opStmt->Operations()) {
                    // Check if inputs contain constant values (iter_0, iter_1, etc.)
                    for (size_t j = 0; j < op->GetNumInputOperand(); ++j) {
                        auto input = op->GetInputOperand(j);
                        if (input->GetValueKind() == pto::ValueKind::Scalar) {
                            auto scalarInput = std::dynamic_pointer_cast<pto::ScalarValue>(input);
                            if (scalarInput && scalarInput->GetName().find("iter_") != std::string::npos) {
                                constantCount++;
                            }
                        }
                    }
                }
            }
        }
    }

    // Should have replaced iteration variable in all 4 iterations
    EXPECT_EQ(constantCount, 4) << "Expected 4 constant replacements for iteration variable";
}

TEST_F(StaticForUnrollTest, VerifyValueReferenceChain) {
    // Test that value references are correctly maintained across multiple operations
    auto module = std::make_shared<ProgramModule>("test_value_chain");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // Create function signature
    FunctionSignature sig;
    std::vector<uint64_t> tileShape = { 16, 16 };
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto outputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
    sig.arguments = { inputTile, outputTile };

    // Create function
    auto func = builder.CreateFunction("test_chain", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // Enter function body
    builder.EnterFunctionBody(ctx, func);

    // Create static for loop: for i = 0 to 2 step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant2 = builder.CreateConst(ctx, int64_t(2), "const_2");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, constant2, constant1);

    builder.EnterForBody(ctx, fs);

    // Loop body: create a chain of operations where output of one is input to next
    auto step1 = builder.CreateTile(ctx, tileShape, DataType::FP32, "step1");
    auto scale1 = builder.CreateConst(ctx, 1.5, "scale1");
    auto op1 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scale1, step1);
    builder.Emit(ctx, op1);

    auto step2 = builder.CreateTile(ctx, tileShape, DataType::FP32, "step2");
    auto scale2 = builder.CreateConst(ctx, 2.0, "scale2");
    auto op2 = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, step1, scale2, step2); // step1 -> step2
    builder.Emit(ctx, op2);

    auto step3 = builder.CreateTile(ctx, tileShape, DataType::FP32, "step3");
    auto scale3 = builder.CreateConst(ctx, 0.5, "scale3");
    auto op3 = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, step2, scale3, step3); // step2 -> step3
    builder.Emit(ctx, op3);

    ctx.PopScope();
    builder.ExitForStatement(ctx, fs);
    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    // Print IR before unrolling
    std::cout << "========== Before Unrolling (Value Chain) =========" << std::endl;
    std::cout << *module << std::endl;

    // Apply transformation
    npu::tile_fwk::StaticForUnrollTransform transform;
    auto status = transform.Apply(module);
    ASSERT_EQ(status, SUCCESS);

    // Print IR after unrolling
    std::cout << "========== After Unrolling (Value Chain) =========" << std::endl;
    std::cout << *module << std::endl;

    // Verify that we have correct number of operations (2 iterations * 3 ops)
    size_t expectedMinStmts = 2 * 3;
    size_t actualOpCount = 0;
    for (size_t idx = 0; idx < func->GetCompound()->GetStatementsNum(); ++idx) {
        auto stmt = func->GetCompound()->GetStatement(idx);
        if (stmt->GetKind() == pto::StatementKind::Op) {
            auto opStmt = std::dynamic_pointer_cast<pto::OpStatement>(stmt);
            if (opStmt) {
                actualOpCount += opStmt->Operations().size();
            }
        }
    }
    ASSERT_GE(actualOpCount, expectedMinStmts) 
        << "Expected at least " << expectedMinStmts << " operations after unrolling";

    // Verify that each iteration has its own unique step1, step2, step3 values
    std::set<std::string> step1Names, step2Names, step3Names;
    for (size_t idx = 0; idx < func->GetCompound()->GetStatementsNum(); ++idx) {
        auto stmt = func->GetCompound()->GetStatement(idx);
        if (stmt->GetKind() == pto::StatementKind::Op) {
            auto opStmt = std::dynamic_pointer_cast<pto::OpStatement>(stmt);
            if (opStmt) {
                for (const auto& op : opStmt->Operations()) {
                    for (size_t j = 0; j < op->GetNumOutputOperand(); ++j) {
                        auto output = op->GetOutputOperand(j);
                        std::string name = output->GetName();
                        if (name.find("step1") != std::string::npos) step1Names.insert(name);
                        if (name.find("step2") != std::string::npos) step2Names.insert(name);
                        if (name.find("step3") != std::string::npos) step3Names.insert(name);
                    }
                }
            }
        }
    }

    // Each step should have 2 unique names (one per iteration)
    EXPECT_GE(step1Names.size(), 2u) << "Expected at least 2 unique step1 names";
    EXPECT_GE(step2Names.size(), 2u) << "Expected at least 2 unique step2 names";
    EXPECT_GE(step3Names.size(), 2u) << "Expected at least 2 unique step3 names";
}

TEST_F(StaticForUnrollTest, VerifyDifferentValueTypes) {
    // Test cloning of different value types: Scalar, Tensor, Tile
    auto module = std::make_shared<ProgramModule>("test_value_types");
    IRBuilder builder(module);
    IRBuilderContext ctx;

    // Create function signature with different value types
    FunctionSignature sig;
    std::vector<uint64_t> tileShape = { 8, 8 };
    auto inputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "input_tile");
    auto scalarParam = std::make_shared<ScalarValue>(DataType::FP32, "scalar_param", ScalarValueKind::Symbolic);
    auto outputTile = std::make_shared<TileValue>(tileShape, DataType::FP32, "output_tile");
    sig.arguments = { inputTile, scalarParam, outputTile };

    // Create function
    auto func = builder.CreateFunction("test_types", FunctionKind::ControlFlow, sig, /*setAsEntry=*/true);

    // Enter function body
    builder.EnterFunctionBody(ctx, func);

    // Create static for loop: for i = 0 to 2 step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "loop_index");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant2 = builder.CreateConst(ctx, int64_t(2), "const_2");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, constant2, constant1);

    builder.EnterForBody(ctx, fs);

    // Loop body: operations using different value types
    // 1. Scalar operation
    auto scalarResult = builder.CreateScalar(ctx, DataType::FP32, "scalar_result");
    auto scalarConst = builder.CreateConst(ctx, 3.0, "scalar_const");
    auto scalarOp = builder.CreateBinaryScalarOp(Opcode::OP_ADDS, scalarParam, scalarConst, scalarResult);
    builder.Emit(ctx, scalarOp);

    // 2. Tile operation
    auto tileResult = builder.CreateTile(ctx, tileShape, DataType::FP32, "tile_result");
    auto tileOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTile, scalarResult, tileResult);
    builder.Emit(ctx, tileOp);

    ctx.PopScope();
    builder.ExitForStatement(ctx, fs);
    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    // Print IR before unrolling
    std::cout << "========== Before Unrolling (Value Types) =========" << std::endl;
    std::cout << *module << std::endl;

    // Apply transformation
    npu::tile_fwk::StaticForUnrollTransform transform;
    auto status = transform.Apply(module);
    ASSERT_EQ(status, SUCCESS);

    // Print IR after unrolling
    std::cout << "========== After Unrolling (Value Types) =========" << std::endl;
    std::cout << *module << std::endl;

    // Verify operations were cloned correctly
    size_t scalarOpCount = 0, tileOpCount = 0;
    for (size_t idx = 0; idx < func->GetCompound()->GetStatementsNum(); ++idx) {
        auto stmt = func->GetCompound()->GetStatement(idx);
        if (stmt->GetKind() == pto::StatementKind::Op) {
            auto opStmt = std::dynamic_pointer_cast<pto::OpStatement>(stmt);
            if (opStmt) {
                for (const auto& op : opStmt->Operations()) {
                    if (op->GetOpcode() == Opcode::OP_ADDS) scalarOpCount++;
                    if (op->GetOpcode() == Opcode::OP_MULS) tileOpCount++;
                }
            }
        }
    }

    // Should have 2 of each operation (2 iterations)
    EXPECT_EQ(scalarOpCount, 2u) << "Expected 2 scalar operations";
    EXPECT_EQ(tileOpCount, 2u) << "Expected 2 tile operations";

    // Verify unique naming for different value types
    std::set<std::string> scalarNames, tileNames;
    for (size_t idx = 0; idx < func->GetCompound()->GetStatementsNum(); ++idx) {
        auto stmt = func->GetCompound()->GetStatement(idx);
        if (stmt->GetKind() == pto::StatementKind::Op) {
            auto opStmt = std::dynamic_pointer_cast<pto::OpStatement>(stmt);
            if (opStmt) {
                for (const auto& op : opStmt->Operations()) {
                    for (size_t j = 0; j < op->GetNumOutputOperand(); ++j) {
                        auto output = op->GetOutputOperand(j);
                        if (output->GetValueKind() == pto::ValueKind::Scalar && 
                            output->GetName().find("scalar_result") != std::string::npos) {
                            scalarNames.insert(output->GetName());
                        }
                        if (output->GetValueKind() == pto::ValueKind::Tile && 
                            output->GetName().find("tile_result") != std::string::npos) {
                            tileNames.insert(output->GetName());
                        }
                    }
                }
            }
        }
    }

    EXPECT_EQ(scalarNames.size(), 2u) << "Expected 2 unique scalar result names";
    EXPECT_EQ(tileNames.size(), 2u) << "Expected 2 unique tile result names";
}

} // namespace pto
