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
 * \file test_visitor.cpp
 * \brief Unit tests for IRVisitor
 */

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>
#include "gtest/gtest.h"

#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/opcode.h"
#include "ir/program.h"
#include "ir/function.h"
#include "ir/statement.h"
#include "ir/value.h"
#include "ir/operation_base.h"
#include "ir/transform/visitor.h"

namespace pto {

/**
 * @brief Test visitor that counts visited nodes
 */
class CountingVisitor : public IRVisitor {
public:
    int programCount = 0;
    int functionCount = 0;
    int compoundStmtCount = 0;
    int opStmtCount = 0;
    int forStmtCount = 0;
    int ifStmtCount = 0;
    int yieldStmtCount = 0;
    int returnStmtCount = 0;
    int operationCount = 0;
    int scalarValueCount = 0;
    int tileValueCount = 0;
    int tensorValueCount = 0;

    // Bring in base class methods to avoid overloaded-virtual warnings
    using IRVisitor::VisitImplOp;
    using IRVisitor::VisitImplStmt;
    using IRVisitor::VisitImplValue;

    void VisitImplProgram(ProgramModulePtr &program) override {
        programCount++;
        IRVisitor::VisitImplProgram(program);
    }

    void VisitImplFunction(FunctionPtr &func) override {
        functionCount++;
        IRVisitor::VisitImplFunction(func);
    }

    void VisitImplStmt(CompoundStatementPtr &stmt) override {
        compoundStmtCount++;
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplStmt(OpStatementPtr &stmt) override {
        opStmtCount++;
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplStmt(ForStatementPtr &stmt) override {
        forStmtCount++;
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplStmt(IfStatementPtr &stmt) override {
        ifStmtCount++;
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplStmt(YieldStatementPtr &stmt) override {
        yieldStmtCount++;
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplStmt(ReturnStatementPtr &stmt) override {
        returnStmtCount++;
        IRVisitor::VisitImplStmt(stmt);
    }

    // ---- operations ----
#define DEFOP(name, inherit, opcode, ...)   \
    void VisitImplOp(name##Ptr &op) override { \
        operationCount++;                   \
        IRVisitor::VisitImplOp(op);            \
    }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

    void VisitImplOp(OperationPtr &op) override {
        operationCount++;
        IRVisitor::VisitImplOp(op);
    }

    void VisitImplValue(ScalarValuePtr &value) override {
        scalarValueCount++;
        IRVisitor::VisitImplValue(value);
    }

    void VisitImplValue(TileValuePtr &value) override {
        tileValueCount++;
        IRVisitor::VisitImplValue(value);
    }

    void VisitImplValue(TensorValuePtr &value) override {
        tensorValueCount++;
        IRVisitor::VisitImplValue(value);
    }
};

/**
 * @brief Test visitor that collects visited values
 */
class ValueCollectorVisitor : public IRVisitor {
public:
    std::unordered_set<ValuePtr> visitedValues;

    // Bring in base class methods to avoid overloaded-virtual warnings
    using IRVisitor::VisitImplValue;

    void VisitImplValue(ScalarValuePtr &value) override {
        if (value) {
            visitedValues.insert(value);
        }
        IRVisitor::VisitImplValue(value);
    }

    void VisitImplValue(TileValuePtr &value) override {
        if (value) {
            visitedValues.insert(value);
        }
        IRVisitor::VisitImplValue(value);
    }

    void VisitImplValue(TensorValuePtr &value) override {
        if (value) {
            visitedValues.insert(value);
        }
        IRVisitor::VisitImplValue(value);
    }
};

TEST(IRVisitorTest, TestBasicTraversal) {
    // Create a simple IR program
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {128, 128};
    auto inputTensor = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto outputTensor = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
    sig.arguments = {inputTensor, outputTensor};

    auto func = builder.CreateFunction("test_func", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    auto c2 = builder.CreateConst(ctx, 2.0, "c2");
    auto mulRes = builder.CreateTile(ctx, tileShape, DataType::FP32, "mul_res");
    auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, inputTensor, c2, mulRes);
    builder.Emit(ctx, mulOp);

    builder.CreateReturn(ctx, {});

    ctx.PopScope();

    // Traverse using CountingVisitor
    CountingVisitor visitor;
    ProgramModulePtr modulePtr = module;
    visitor.VisitProgram(modulePtr);

    // Verify visit counts
    ASSERT_EQ(visitor.programCount, 1);
    // VisitImplProgram first visits the entry function, then all functions in the functions list
    ASSERT_GE(visitor.functionCount, 1);
    ASSERT_GE(visitor.compoundStmtCount, 1); // at least one compound statement
    ASSERT_GE(visitor.opStmtCount, 1);       // at least one op statement
    // return statement will also be visited twice if the function is visited twice
    ASSERT_GE(visitor.returnStmtCount, 1);
    ASSERT_GE(visitor.operationCount, 1);   // at least one operation
    ASSERT_GE(visitor.scalarValueCount, 1); // at least one scalar value (constant c2)
    ASSERT_GE(visitor.tileValueCount, 3);   // inputTensor, outputTensor, mulRes
}

TEST(IRVisitorTest, TestValueCollection) {
    // Create an IR program with multiple values
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {64, 64};
    auto in1 = std::make_shared<TileValue>(tileShape, DataType::FP32, "in1");
    auto in2 = std::make_shared<TileValue>(tileShape, DataType::FP32, "in2");
    sig.arguments = {in1, in2};

    auto func = builder.CreateFunction("test_collect", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    auto c1 = builder.CreateConst(ctx, 1.0, "c1");
    auto c2 = builder.CreateConst(ctx, 2.0, "c2");
    auto c3 = builder.CreateConst(ctx, 3.0, "c3");

    auto addRes = builder.CreateTile(ctx, tileShape, DataType::FP32, "add_res");
    auto addOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, in1, c1, addRes);
    builder.Emit(ctx, addOp);

    auto mulRes = builder.CreateTile(ctx, tileShape, DataType::FP32, "mul_res");
    auto mulOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, addRes, c2, mulRes);
    builder.Emit(ctx, mulOp);

    // Use c3 to ensure it is visited
    ctx.ResetInsertionPoint();
    auto finalRes = builder.CreateTile(ctx, tileShape, DataType::FP32, "final_res");
    auto finalOp = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, mulRes, c3, finalRes);
    builder.Emit(ctx, finalOp);

    builder.CreateReturn(ctx, {});

    ctx.PopScope();

    // Collect all values using ValueCollectorVisitor
    ValueCollectorVisitor collector;
    ProgramModulePtr modulePtr = module;
    collector.VisitProgram(modulePtr);

    // Verify collected values
    ASSERT_GT(collector.visitedValues.size(), 0);
    ASSERT_TRUE(collector.visitedValues.find(in1) != collector.visitedValues.end());
    ASSERT_TRUE(collector.visitedValues.find(in2) != collector.visitedValues.end());
    ASSERT_TRUE(collector.visitedValues.find(c1) != collector.visitedValues.end());
    ASSERT_TRUE(collector.visitedValues.find(c2) != collector.visitedValues.end());
    ASSERT_TRUE(collector.visitedValues.find(c3) != collector.visitedValues.end());
    ASSERT_TRUE(collector.visitedValues.find(addRes) != collector.visitedValues.end());
    ASSERT_TRUE(collector.visitedValues.find(mulRes) != collector.visitedValues.end());
    ASSERT_TRUE(collector.visitedValues.find(finalRes) != collector.visitedValues.end());
}

TEST(IRVisitorTest, TestControlFlowTraversal) {
    // Create an IR program with control flow
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    auto batch = std::make_shared<ScalarValue>(DataType::INT32, "batch", ScalarValueKind::Symbolic);
    auto constant128 = std::make_shared<ScalarValue>(int64_t(128), "const_128");
    std::vector<ScalarValuePtr> tensorShape = {batch, constant128};
    std::vector<int64_t> tileShape = {128, 128};

    auto inputX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "inputX");
    auto scale1 = std::make_shared<ScalarValue>(DataType::FP32, "scale1", ScalarValueKind::Symbolic);
    auto resultX = std::make_shared<TensorValue>(tensorShape, DataType::FP32, "outputX");

    sig.arguments = {inputX, scale1, resultX};

    auto func = builder.CreateFunction("test_control", FunctionKind::ControlFlow, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    // for i = 0 to batch step 1
    auto i = builder.CreateScalar(ctx, DataType::INT32, "i");
    auto constant0 = builder.CreateConst(ctx, int64_t(0), "const_0");
    auto constant1 = builder.CreateConst(ctx, int64_t(1), "const_1");
    auto fs = builder.CreateForStmt(ctx, i, constant0, batch, constant1);

    builder.EnterForBody(ctx, fs);

    auto resLoopX = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputX");
    auto addOpX = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, resLoopX, scale1, resLoopX);
    builder.Emit(ctx, addOpX);

    // if i then ...
    auto ifs = builder.CreateIfStmt(ctx, i);
    builder.EnterIfThen(ctx, ifs);
    auto resIfX = builder.CreateTile(ctx, tileShape, DataType::FP32, "outputX");
    auto mulOpX = builder.CreateBinaryScalarMixOp(Opcode::OP_MULS, resLoopX, scale1, resIfX);
    builder.Emit(ctx, mulOpX);
    ctx.PopScope();

    builder.EnterIfElse(ctx, ifs);
    ctx.PopScope();

    builder.ExitIfStatement(ctx, ifs);
    ctx.PopScope(); // for-body
    builder.ExitForStatement(ctx, fs);

    builder.CreateReturn(ctx, {constant0});
    ctx.PopScope(); // function-body

    // Traverse using CountingVisitor
    CountingVisitor visitor;
    ProgramModulePtr modulePtr = module;
    visitor.VisitProgram(modulePtr);

    // Verify control flow nodes are visited
    ASSERT_EQ(visitor.programCount, 1);
    // VisitImplProgram first visits the entry function, then all functions in the functions list
    // If the entry function is also in the functions list, it will be visited twice
    ASSERT_GE(visitor.functionCount, 1);
    ASSERT_GE(visitor.forStmtCount, 1);      // at least one for statement
    ASSERT_GE(visitor.ifStmtCount, 1);       // at least one if statement
    ASSERT_GE(visitor.compoundStmtCount, 3); // function body, for body, if branches
    ASSERT_GE(visitor.operationCount, 2);    // addOpX and mulOpX
}

/**
 * @brief Test visitor for default traversal methods
 */
class DefaultVisitTester : public IRVisitor {
public:
    int visitedCount = 0;
    std::unordered_set<void *> visitedNodes;

    // Bring in base class methods to avoid overloaded-virtual warnings
    using IRVisitor::VisitImplFunction;
    using IRVisitor::VisitImplOp;
    using IRVisitor::VisitImplProgram;
    using IRVisitor::VisitImplStmt;
    using IRVisitor::VisitImplValue;

    // Override VisitImplXXX methods to track visits, but call base class default implementation
    void VisitImplFunction(FunctionPtr &func) override {
        if (func) {
            visitedNodes.insert(func.get());
            visitedCount++;
        }
        IRVisitor::VisitImplFunction(func);
    }

    void VisitImplStmt(CompoundStatementPtr &stmt) override {
        if (stmt) {
            visitedNodes.insert(stmt.get());
            visitedCount++;
        }
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplStmt(OpStatementPtr &stmt) override {
        if (stmt) {
            visitedNodes.insert(stmt.get());
            visitedCount++;
        }
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplStmt(ForStatementPtr &stmt) override {
        if (stmt) {
            visitedNodes.insert(stmt.get());
            visitedCount++;
        }
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplStmt(IfStatementPtr &stmt) override {
        if (stmt) {
            visitedNodes.insert(stmt.get());
            visitedCount++;
        }
        IRVisitor::VisitImplStmt(stmt);
    }

    void VisitImplOp(OperationPtr &op) override {
        if (op) {
            visitedNodes.insert(op.get());
            visitedCount++;
        }
        IRVisitor::VisitImplOp(op);
    }

    void VisitImplValue(ScalarValuePtr &value) override {
        if (value) {
            visitedNodes.insert(value.get());
            visitedCount++;
        }
        IRVisitor::VisitImplValue(value);
    }

    void VisitImplValue(TileValuePtr &value) override {
        if (value) {
            visitedNodes.insert(value.get());
            visitedCount++;
        }
        IRVisitor::VisitImplValue(value);
    }

    void VisitImplValue(TensorValuePtr &value) override {
        if (value) {
            visitedNodes.insert(value.get());
            visitedCount++;
        }
        IRVisitor::VisitImplValue(value);
    }
};

/**
 * @brief Test visitor for null handling
 */
class NullHandlingVisitor : public IRVisitor {
public:
    // Use using declarations to bring in protected methods for testing
    using IRVisitor::VisitImplFunction;
    using IRVisitor::VisitImplOp;
    using IRVisitor::VisitImplStmt;
    using IRVisitor::VisitImplValue;
};

TEST(IRVisitorTest, TestDefaultTraversal) {
    // Test default traversal behavior
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {32, 32};
    auto input = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    sig.arguments = {input};

    auto func = builder.CreateFunction("test_default", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    auto c1 = builder.CreateConst(ctx, 1.0, "c1");
    auto result = builder.CreateTile(ctx, tileShape, DataType::FP32, "result");
    auto op = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, input, c1, result);
    builder.Emit(ctx, op);

    builder.CreateReturn(ctx, {});

    ctx.PopScope();

    // Use default IRVisitor (without overriding any methods)
    IRVisitor visitor;
    ProgramModulePtr modulePtr = module;
    visitor.VisitProgram(modulePtr);

    // Default visitor should be able to traverse all nodes without crashing
    // Although there's no counting, it should complete traversal normally
    ASSERT_TRUE(true); // If we reach here, traversal succeeded
}

TEST(IRVisitorTest, TestDefaultVisitMethods) {
    // Test default traversal method behavior
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    FunctionSignature sig;
    std::vector<int64_t> tileShape = {64, 64};
    auto input = std::make_shared<TileValue>(tileShape, DataType::FP32, "input");
    auto output = std::make_shared<TileValue>(tileShape, DataType::FP32, "output");
    sig.arguments = {input, output};

    auto func = builder.CreateFunction("test_default_visit", FunctionKind::Block, sig);
    module->AddFunction(func);
    builder.EnterFunctionBody(ctx, func);

    auto c1 = builder.CreateConst(ctx, 1.0, "c1");
    auto result = builder.CreateTile(ctx, tileShape, DataType::FP32, "result");
    auto addOp = builder.CreateBinaryScalarMixOp(Opcode::OP_ADDS, input, c1, result);
    builder.Emit(ctx, addOp);

    builder.CreateReturn(ctx, {});
    ctx.PopScope();

    // Use DefaultVisitTester, which calls default Visit*_ methods
    DefaultVisitTester tester;
    ProgramModulePtr modulePtr = module;
    tester.VisitImplProgram(modulePtr);

    // Verify VisitImplProgram correctly traverses all child nodes
    // It should have visited functions, statements, operations, values
    ASSERT_GT(tester.visitedCount, 0);
    ASSERT_NE(tester.visitedNodes.find(func.get()), tester.visitedNodes.end());

    // Verify VisitImplFunction is called
    FunctionPtr funcPtr = func;
    tester.VisitImplFunction(funcPtr);
    // Should have visited function arguments, results, body, etc.

    // Verify VisitImplOp is called
    OperationPtr opPtr = addOp;
    tester.VisitImplOp(opPtr);
    // Should have visited operation operands

    // Verify VisitImplValue is called
    ScalarValuePtr c1Ptr = c1;
    tester.VisitImplValue(c1Ptr);

    TileValuePtr inputPtr = input;
    tester.VisitImplValue(inputPtr);

    // Verify VisitImplStmt is called
    auto compound = func->GetCompound();
    CompoundStatementPtr compoundPtr = compound;
    tester.VisitImplStmt(compoundPtr);
    // Should have visited all statements in compound
}

TEST(IRVisitorTest, TestNullHandling) {
    NullHandlingVisitor visitor;

    FunctionPtr nullFunc = nullptr;
    visitor.VisitImplFunction(nullFunc);

    CompoundStatementPtr nullCompound = nullptr;
    visitor.VisitImplStmt(nullCompound);

    OperationPtr nullOp = nullptr;
    visitor.VisitImplOp(nullOp);

    ScalarValuePtr nullScalar = nullptr;
    visitor.VisitImplValue(nullScalar);

    TileValuePtr nullTile = nullptr;
    visitor.VisitImplValue(nullTile);

    TensorValuePtr nullTensor = nullptr;
    visitor.VisitImplValue(nullTensor);

    ASSERT_TRUE(true);
}

} // namespace pto
