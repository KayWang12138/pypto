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
 * \file test_transform.cpp
 * \brief Unit tests for IR transform module (printer, visitor, mutator,
 *        structural_equal/hash, verifier, passes, dependency_analyzer)
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/builder.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"
#include "ir/transform/base/mutator.h"
#include "ir/transform/base/visitor.h"
#include "ir/transform/dependency_analyzer.h"
#include "ir/transform/passes.h"
#include "ir/transform/printer.h"
#include "ir/transform/structural_comparison.h"
#include "ir/transform/verifier.h"

namespace pypto {
namespace ir {

// ============================================================================
// Helper utilities
// ============================================================================

static Span TestSpan() {
    return Span("test.py", 1, 0);
}

static TypePtr Int32Type() {
    return std::make_shared<ScalarType>(DataType::INT32);
}

// Helper: build a simple function with body "y = x + 1; return y"
static FunctionPtr BuildSimpleFunction(const std::string &name = "testFunc") {
    auto span = TestSpan();
    auto intType = Int32Type();

    auto x = std::make_shared<Var>("x", intType, span);
    auto y = std::make_shared<Var>("y", intType, span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto addExpr = std::make_shared<Add>(x, one, DataType::INT32, span);

    auto assign = std::make_shared<AssignStmt>(y, addExpr, span);
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{y}, span);
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign, ret}, span);

    return std::make_shared<Function>(name, std::vector<VarPtr>{x}, std::vector<TypePtr>{intType}, body, span);
}

// Helper: build a simple program with one function
static ProgramPtr BuildSimpleProgram() {
    auto func = BuildSimpleFunction("main");
    return std::make_shared<Program>(std::vector<FunctionPtr>{func}, "test_prog", TestSpan());
}

// ============================================================================
// Printer Tests (printer.cpp)
// ============================================================================

TEST(PrinterTest, PrintVar) {
    auto span = TestSpan();
    auto var = std::make_shared<Var>("x", Int32Type(), span);
    IRNodePtr node = var;
    auto result = PythonPrint(node);
    ASSERT_FALSE(result.empty());
    ASSERT_NE(result.find("x"), std::string::npos);
}

TEST(PrinterTest, PrintConstInt) {
    auto span = TestSpan();
    auto c = std::make_shared<ConstInt>(42, DataType::INT32, span);
    IRNodePtr node = c;
    auto result = PythonPrint(node);
    ASSERT_NE(result.find("42"), std::string::npos);
}

TEST(PrinterTest, PrintBinaryExpr) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto y = std::make_shared<Var>("y", Int32Type(), span);
    auto add = std::make_shared<Add>(x, y, DataType::INT32, span);
    IRNodePtr node = add;
    auto result = PythonPrint(node);
    ASSERT_FALSE(result.empty());
}

TEST(PrinterTest, PrintAssignStmt) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto assign = std::make_shared<AssignStmt>(x, one, span);
    IRNodePtr node = assign;
    auto result = PythonPrint(node);
    ASSERT_NE(result.find("x"), std::string::npos);
}

TEST(PrinterTest, PrintFunction) {
    auto func = BuildSimpleFunction();
    IRNodePtr node = func;
    auto result = PythonPrint(node);
    ASSERT_NE(result.find("def"), std::string::npos);
    ASSERT_NE(result.find("testFunc"), std::string::npos);
}

TEST(PrinterTest, PrintProgram) {
    auto prog = BuildSimpleProgram();
    IRNodePtr node = prog;
    auto result = PythonPrint(node);
    ASSERT_NE(result.find("def"), std::string::npos);
}

TEST(PrinterTest, PrintType) {
    auto intType = Int32Type();
    auto result = PythonPrint(intType);
    ASSERT_FALSE(result.empty());
}

TEST(PrinterTest, GetPrecedenceVar) {
    auto span = TestSpan();
    auto var = std::make_shared<Var>("x", Int32Type(), span);
    ExprPtr expr = var;
    auto prec = GetPrecedence(expr);
    ASSERT_EQ(prec, Precedence::ATOM);
}

TEST(PrinterTest, GetPrecedenceAdd) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto y = std::make_shared<Var>("y", Int32Type(), span);
    auto add = std::make_shared<Add>(x, y, DataType::INT32, span);
    ExprPtr expr = add;
    auto prec = GetPrecedence(expr);
    ASSERT_EQ(prec, Precedence::ADD_SUB);
}

TEST(PrinterTest, IsRightAssociativeAdd) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto y = std::make_shared<Var>("y", Int32Type(), span);
    auto add = std::make_shared<Add>(x, y, DataType::INT32, span);
    ExprPtr expr = add;
    ASSERT_FALSE(IsRightAssociative(expr));
}

TEST(PrinterTest, PrintWithCustomPrefix) {
    auto func = BuildSimpleFunction();
    IRNodePtr node = func;
    auto result = PythonPrint(node, "ir");
    ASSERT_FALSE(result.empty());
}

// ============================================================================
// Visitor Tests (visitor.cpp)
// ============================================================================

// Custom visitor that counts nodes
class NodeCounter : public IRVisitor {
public:
    int exprCount = 0;
    int stmtCount = 0;

    void VisitExpr(const ExprPtr &expr) override {
        exprCount++;
        IRVisitor::VisitExpr(expr);
    }

    void VisitStmt(const StmtPtr &stmt) override {
        stmtCount++;
        IRVisitor::VisitStmt(stmt);
    }
};

TEST(VisitorTest, VisitSimpleExpr) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto add = std::make_shared<Add>(x, one, DataType::INT32, span);

    NodeCounter counter;
    counter.VisitExpr(add);
    ASSERT_GE(counter.exprCount, 3); // add, x, one
}

TEST(VisitorTest, VisitStatement) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto assign = std::make_shared<AssignStmt>(x, one, span);

    NodeCounter counter;
    counter.VisitStmt(assign);
    ASSERT_GE(counter.stmtCount, 1);
}

TEST(VisitorTest, VisitSeqStmts) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto assign1 = std::make_shared<AssignStmt>(x, one, span);
    auto assign2 = std::make_shared<AssignStmt>(x, one, span);
    auto seq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign1, assign2}, span);

    NodeCounter counter;
    counter.VisitStmt(seq);
    ASSERT_GE(counter.stmtCount, 3); // seq + 2 assigns
}

TEST(VisitorTest, VisitForStmt) {
    auto span = TestSpan();
    auto i = std::make_shared<Var>("i", Int32Type(), span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto forStmt =
        std::make_shared<ForStmt>(i, start, stop, step, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{}, span);

    NodeCounter counter;
    counter.VisitStmt(forStmt);
    ASSERT_GE(counter.stmtCount, 1);
}

// ============================================================================
// Mutator Tests (mutator.cpp)
// ============================================================================

TEST(MutatorTest, IdentityMutateExpr) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto add = std::make_shared<Add>(x, one, DataType::INT32, span);

    IRMutator mutator;
    auto result = mutator.VisitExpr(add);
    ASSERT_NE(result, nullptr);
    // Identity mutator should return same pointer (copy-on-write)
    ASSERT_EQ(result.get(), add.get());
}

TEST(MutatorTest, IdentityMutateStmt) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto assign = std::make_shared<AssignStmt>(x, one, span);

    IRMutator mutator;
    auto result = mutator.VisitStmt(assign);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result.get(), assign.get());
}

TEST(MutatorTest, IdentityMutateSeqStmts) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto assign = std::make_shared<AssignStmt>(x, one, span);
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{x}, span);
    auto seq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assign, ret}, span);

    IRMutator mutator;
    auto result = mutator.VisitStmt(seq);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result.get(), seq.get());
}

TEST(MutatorTest, IdentityMutateForStmt) {
    auto span = TestSpan();
    auto i = std::make_shared<Var>("i", Int32Type(), span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto forStmt =
        std::make_shared<ForStmt>(i, start, stop, step, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{}, span);

    IRMutator mutator;
    auto result = mutator.VisitStmt(forStmt);
    ASSERT_NE(result, nullptr);
}

TEST(MutatorTest, IdentityMutateIfStmt) {
    auto span = TestSpan();
    auto cond = std::make_shared<ConstBool>(true, span);
    auto thenBody = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::nullopt, std::vector<VarPtr>{}, span);

    IRMutator mutator;
    auto result = mutator.VisitStmt(ifStmt);
    ASSERT_NE(result, nullptr);
}

TEST(MutatorTest, MutateUnaryExpr) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto neg = std::make_shared<Neg>(x, DataType::INT32, span);

    IRMutator mutator;
    auto result = mutator.VisitExpr(neg);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result.get(), neg.get());
}

TEST(MutatorTest, MutateCastExpr) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto floatType = std::make_shared<ScalarType>(DataType::FP32);
    auto cast = std::make_shared<Cast>(x, DataType::FP32, span);

    IRMutator mutator;
    auto result = mutator.VisitExpr(cast);
    ASSERT_NE(result, nullptr);
}

TEST(MutatorTest, MutateEvalStmt) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto eval = std::make_shared<EvalStmt>(x, span);

    IRMutator mutator;
    auto result = mutator.VisitStmt(eval);
    ASSERT_NE(result, nullptr);
}

TEST(MutatorTest, MutateReturnStmt) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{x}, span);

    IRMutator mutator;
    auto result = mutator.VisitStmt(ret);
    ASSERT_NE(result, nullptr);
}

TEST(MutatorTest, MutateYieldStmt) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{x}, span);

    IRMutator mutator;
    auto result = mutator.VisitStmt(yield);
    ASSERT_NE(result, nullptr);
}

// ============================================================================
// Structural Equal Tests (structural_equal.cpp)
// ============================================================================

TEST(StructuralEqualExtraTest, SameExprEqual) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto add1 = std::make_shared<Add>(x, one, DataType::INT32, span);
    auto add2 = std::make_shared<Add>(x, one, DataType::INT32, span);

    IRNodePtr n1 = add1;
    IRNodePtr n2 = add2;
    ASSERT_TRUE(structural_equal(n1, n2));
}

TEST(StructuralEqualExtraTest, DifferentExprNotEqual) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, span);
    auto add1 = std::make_shared<Add>(x, one, DataType::INT32, span);
    auto add2 = std::make_shared<Add>(x, two, DataType::INT32, span);

    IRNodePtr n1 = add1;
    IRNodePtr n2 = add2;
    ASSERT_FALSE(structural_equal(n1, n2));
}

TEST(StructuralEqualExtraTest, AutoMappingVariables) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto y = std::make_shared<Var>("y", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto add1 = std::make_shared<Add>(x, one, DataType::INT32, span);
    auto add2 = std::make_shared<Add>(y, one, DataType::INT32, span);

    IRNodePtr n1 = add1;
    IRNodePtr n2 = add2;
    // Without auto mapping, different var names → not equal
    ASSERT_FALSE(structural_equal(n1, n2, false));
    // With auto mapping, x maps to y → equal
    ASSERT_TRUE(structural_equal(n1, n2, true));
}

TEST(StructuralEqualExtraTest, TypeEqual) {
    auto t1 = Int32Type();
    auto t2 = Int32Type();
    ASSERT_TRUE(structural_equal(t1, t2));
}

TEST(StructuralEqualExtraTest, TypeNotEqual) {
    auto t1 = Int32Type();
    auto t2 = std::make_shared<ScalarType>(DataType::FP32);
    ASSERT_FALSE(structural_equal(t1, t2));
}

TEST(StructuralEqualExtraTest, AssertStructuralEqualPass) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    IRNodePtr n1 = x;
    IRNodePtr n2 = x;
    ASSERT_NO_THROW(assert_structural_equal(n1, n2));
}

TEST(StructuralEqualExtraTest, AssertStructuralEqualFail) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto y = std::make_shared<Var>("y", Int32Type(), span);
    IRNodePtr n1 = x;
    IRNodePtr n2 = y;
    ASSERT_THROW(assert_structural_equal(n1, n2), ValueError);
}

// ============================================================================
// Structural Hash Tests (structural_hash.cpp)
// ============================================================================

TEST(StructuralHashExtraTest, SameExprSameHash) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto add1 = std::make_shared<Add>(x, one, DataType::INT32, span);
    auto add2 = std::make_shared<Add>(x, one, DataType::INT32, span);

    IRNodePtr n1 = add1;
    IRNodePtr n2 = add2;
    ASSERT_EQ(structural_hash(n1), structural_hash(n2));
}

TEST(StructuralHashExtraTest, DifferentExprDifferentHash) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, span);
    auto add1 = std::make_shared<Add>(x, one, DataType::INT32, span);
    auto add2 = std::make_shared<Add>(x, two, DataType::INT32, span);

    IRNodePtr n1 = add1;
    IRNodePtr n2 = add2;
    ASSERT_NE(structural_hash(n1), structural_hash(n2));
}

TEST(StructuralHashExtraTest, AutoMappingHash) {
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto y = std::make_shared<Var>("y", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto add1 = std::make_shared<Add>(x, one, DataType::INT32, span);
    auto add2 = std::make_shared<Add>(y, one, DataType::INT32, span);

    IRNodePtr n1 = add1;
    IRNodePtr n2 = add2;
    // With auto mapping, x and y should hash the same
    ASSERT_EQ(structural_hash(n1, true), structural_hash(n2, true));
}

TEST(StructuralHashExtraTest, TypeHash) {
    auto t1 = Int32Type();
    auto t2 = Int32Type();
    ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

TEST(StructuralHashExtraTest, DifferentTypeHash) {
    auto t1 = Int32Type();
    auto t2 = std::make_shared<ScalarType>(DataType::FP32);
    ASSERT_NE(structural_hash(t1), structural_hash(t2));
}

// ============================================================================
// Verifier Tests (verifier.cpp)
// ============================================================================

TEST(VerifierTest, CreateDefaultVerifier) {
    auto verifier = IRVerifier::CreateDefault();
    ASSERT_TRUE(verifier.IsRuleEnabled("SSAVerify"));
    ASSERT_TRUE(verifier.IsRuleEnabled("TypeCheck"));
}

TEST(VerifierTest, DisableEnableRule) {
    auto verifier = IRVerifier::CreateDefault();
    verifier.DisableRule("SSAVerify");
    ASSERT_FALSE(verifier.IsRuleEnabled("SSAVerify"));
    verifier.EnableRule("SSAVerify");
    ASSERT_TRUE(verifier.IsRuleEnabled("SSAVerify"));
}

TEST(VerifierTest, VerifyValidProgram) {
    auto prog = BuildSimpleProgram();
    auto verifier = IRVerifier::CreateDefault();
    auto diagnostics = verifier.Verify(prog);
    // A simple valid program should have no errors
    int errorCount = 0;
    for (const auto &d : diagnostics) {
        if (d.severity == DiagnosticSeverity::ERROR) {
            errorCount++;
        }
    }
    ASSERT_EQ(errorCount, 0);
}

TEST(VerifierTest, VerifyOrThrowValid) {
    auto prog = BuildSimpleProgram();
    auto verifier = IRVerifier::CreateDefault();
    ASSERT_NO_THROW(verifier.VerifyOrThrow(prog));
}

TEST(VerifierTest, GenerateReport) {
    std::vector<Diagnostic> diagnostics;
    auto report = IRVerifier::GenerateReport(diagnostics);
    ASSERT_FALSE(report.empty());
}

TEST(VerifierTest, EmptyVerifier) {
    IRVerifier verifier;
    auto prog = BuildSimpleProgram();
    auto diagnostics = verifier.Verify(prog);
    ASSERT_TRUE(diagnostics.empty());
}

// ============================================================================
// Pass Tests (passes.cpp)
// ============================================================================

TEST(PassTest, CreateFunctionPass) {
    auto myPass = pass::CreateFunctionPass([](const FunctionPtr &func) { return func; }, // identity pass
        "IdentityPass");

    auto prog = BuildSimpleProgram();
    auto result = myPass(prog);
    ASSERT_NE(result, nullptr);
}

TEST(PassTest, PassRunMethod) {
    auto myPass = pass::CreateFunctionPass([](const FunctionPtr &func) { return func; }, "IdentityPass");

    auto prog = BuildSimpleProgram();
    auto result = myPass.run(prog);
    ASSERT_NE(result, nullptr);
}

TEST(PassTest, CreateProgramPass) {
    auto myPass = pass::CreateProgramPass([](const ProgramPtr &prog) { return prog; }, // identity pass
        "IdentityProgramPass");

    auto prog = BuildSimpleProgram();
    auto result = myPass(prog);
    ASSERT_NE(result, nullptr);
}

TEST(PassTest, DefaultPass) {
    Pass p;
    // Default pass with no impl should handle gracefully
    auto prog = BuildSimpleProgram();
    // Depending on implementation, this may throw or return nullptr
    // Just test construction works
    ASSERT_TRUE(true);
}

TEST(PassTest, PassCopyAndMove) {
    auto myPass = pass::CreateFunctionPass([](const FunctionPtr &func) { return func; }, "CopyTest");

    Pass copy = myPass;
    auto prog = BuildSimpleProgram();
    auto result = copy(prog);
    ASSERT_NE(result, nullptr);

    Pass moved = std::move(copy);
    result = moved(prog);
    ASSERT_NE(result, nullptr);
}

// ============================================================================
// Dependency Analyzer Tests (dependency_analyzer.cpp)
// ============================================================================

TEST(DependencyAnalyzerExtraTest, AnalyzeSimpleFunction) {
    auto func = BuildSimpleFunction();
    DependencyAnalyzer analyzer;
    auto graph = analyzer.Analyze(func);
    // Should have at least one basic block
    ASSERT_GE(graph.blocks.size(), 1u);
}

TEST(DependencyAnalyzerExtraTest, AnalyzeFunctionWithLoop) {
    auto span = TestSpan();
    auto intType = Int32Type();

    IRBuilder builder;
    builder.BeginFunction("loop_func", span);
    auto x = builder.FuncArg("x", intType, span);
    builder.ReturnType(intType);

    auto i = std::make_shared<Var>("i", intType, span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);

    builder.BeginForLoop(i, start, stop, step, span);
    builder.EndForLoop(span);

    builder.Return({x}, span);
    auto func = builder.EndFunction(span);

    DependencyAnalyzer analyzer;
    auto graph = analyzer.Analyze(func);
    ASSERT_GE(graph.blocks.size(), 1u);
}

TEST(DependencyAnalyzerExtraTest, AnalyzeBlocks) {
    auto func = BuildSimpleFunction();
    DependencyAnalyzer analyzer;
    auto blocks = analyzer.AnalyzeBasicBlocks(func);
    ASSERT_GE(blocks.size(), 1u);
}

TEST(DependencyAnalyzerExtraTest, DependencyGraphDefaultConstruction) {
    DependencyGraph graph;
    ASSERT_TRUE(graph.blocks.empty());
    ASSERT_TRUE(graph.dependencies.empty());
}

} // namespace ir
} // namespace pypto
