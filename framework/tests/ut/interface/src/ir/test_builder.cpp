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
 * \file test_builder.cpp
 * \brief Unit tests for IRBuilder
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/builder.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ============================================================================
// Helper utilities
// ============================================================================

static Span TestSpan() {
    return Span("test.py", 1, 0);
}
static Span TestSpan2() {
    return Span("test.py", 10, 0);
}

static TypePtr Int32Type() {
    return std::make_shared<ScalarType>(DataType::INT32);
}
static TypePtr Float32Type() {
    return std::make_shared<ScalarType>(DataType::FP32);
}

// ============================================================================
// IRBuilder Construction Tests
// ============================================================================

TEST(IRBuilderTest, DefaultConstruction) {
    IRBuilder builder;
    ASSERT_FALSE(builder.InFunction());
    ASSERT_FALSE(builder.InIf());
    ASSERT_FALSE(builder.InProgram());
    ASSERT_EQ(builder.CurrentContext(), nullptr);
}

// ============================================================================
// Function Building Tests
// ============================================================================

TEST(IRBuilderTest, BuildSimpleFunction) {
    IRBuilder builder;
    auto span = TestSpan();
    auto endSpan = TestSpan2();

    builder.BeginFunction("my_func", span);
    ASSERT_TRUE(builder.InFunction());

    auto x = builder.FuncArg("x", Int32Type(), span);
    ASSERT_NE(x, nullptr);
    ASSERT_EQ(x->name_, "x");

    builder.ReturnType(Int32Type());

    auto func = builder.EndFunction(endSpan);
    ASSERT_NE(func, nullptr);
    ASSERT_EQ(func->name_, "my_func");
    ASSERT_EQ(func->params_.size(), 1);
    ASSERT_EQ(func->returnTypes_.size(), 1);
    ASSERT_FALSE(builder.InFunction());
}

TEST(IRBuilderTest, BuildFunctionWithMultipleParams) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("multi_param", span);
    auto a = builder.FuncArg("a", Int32Type(), span);
    auto b = builder.FuncArg("b", Float32Type(), span);
    builder.ReturnType(Float32Type());

    auto func = builder.EndFunction(span);
    ASSERT_EQ(func->params_.size(), 2);
    ASSERT_EQ(func->params_[0]->name_, "a");
    ASSERT_EQ(func->params_[1]->name_, "b");
}

TEST(IRBuilderTest, BuildFunctionWithEmptyBody) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("empty_func", span);
    auto func = builder.EndFunction(span);
    ASSERT_NE(func, nullptr);
    ASSERT_NE(func->body_, nullptr);
    // Empty body should be SeqStmts
    ASSERT_EQ(func->body_->GetKind(), ObjectKind::SeqStmts);
}

TEST(IRBuilderTest, NestedFunctionThrows) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("outer", span);
    ASSERT_THROW(builder.BeginFunction("inner", span), RuntimeError);
    builder.EndFunction(span);
}

// ============================================================================
// Statement Emission Tests
// ============================================================================

TEST(IRBuilderTest, EmitAssignment) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("assign_func", span);
    auto x = builder.FuncArg("x", Int32Type(), span);
    auto y = builder.Var("y", Int32Type(), span);
    auto assign = builder.Assign(y, x, span);

    ASSERT_NE(assign, nullptr);
    ASSERT_EQ(assign->GetKind(), ObjectKind::AssignStmt);

    auto func = builder.EndFunction(span);
    ASSERT_NE(func->body_, nullptr);
}

TEST(IRBuilderTest, EmitReturn) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("ret_func", span);
    auto x = builder.FuncArg("x", Int32Type(), span);
    builder.ReturnType(Int32Type());
    auto ret = builder.Return({x}, span);

    ASSERT_NE(ret, nullptr);
    ASSERT_EQ(ret->GetKind(), ObjectKind::ReturnStmt);
    ASSERT_EQ(ret->value_.size(), 1);

    builder.EndFunction(span);
}

TEST(IRBuilderTest, EmitEmptyReturn) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("void_func", span);
    auto ret = builder.Return(span);
    ASSERT_NE(ret, nullptr);
    ASSERT_TRUE(ret->value_.empty());

    builder.EndFunction(span);
}

TEST(IRBuilderTest, EmitOutsideContextThrows) {
    IRBuilder builder;
    auto span = TestSpan();
    auto stmt = std::make_shared<ReturnStmt>(span);
    ASSERT_THROW(builder.Emit(stmt), RuntimeError);
}

// ============================================================================
// For Loop Building Tests
// ============================================================================

TEST(IRBuilderTest, BuildSimpleForLoop) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("loop_func", span);
    ASSERT_TRUE(builder.InFunction());

    auto i = std::make_shared<Var>("i", Int32Type(), span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);

    auto forLoop = builder.CreateForLoop(i, start, stop, step, span);
    auto forStmt = forLoop.Build(span);
    ASSERT_NE(forStmt, nullptr);
    ASSERT_EQ(forStmt->GetKind(), ObjectKind::ForStmt);
    builder.Emit(forStmt);

    builder.EndFunction(span);
}

TEST(IRBuilderTest, BuildForLoopWithIterArgs) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("iter_func", span);

    auto i = std::make_shared<Var>("i", Int32Type(), span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);

    auto forLoop = builder.CreateForLoop(i, start, stop, step, span);

    auto initVal = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto iterArg = std::make_shared<IterArg>("sum", Int32Type(), initVal, span);
    forLoop.AddIterArg(iterArg);

    auto retVar = std::make_shared<Var>("sum_final", Int32Type(), span);
    forLoop.AddReturnVar(retVar);

    auto forStmt = forLoop.Build(span);
    ASSERT_NE(forStmt, nullptr);
    builder.Emit(forStmt);

    builder.EndFunction(span);
}

TEST(IRBuilderTest, ForLoopMismatchedIterArgsThrows) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("mismatch_func", span);

    auto i = std::make_shared<Var>("i", Int32Type(), span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);

    auto forLoop = builder.CreateForLoop(i, start, stop, step, span);

    auto initVal = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto iterArg = std::make_shared<IterArg>("sum", Int32Type(), initVal, span);
    forLoop.AddIterArg(iterArg);
    // No AddReturnVar — mismatch

    ASSERT_THROW(forLoop.Build(span), RuntimeError);
    builder.EndFunction(span);
}

TEST(IRBuilderTest, ForLoopOutsideContextThrows) {
    IRBuilder builder;
    auto span = TestSpan();

    auto i = std::make_shared<Var>("i", Int32Type(), span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);

    // CreateForLoop is stateless - it can be called without a context
    auto forLoop = builder.CreateForLoop(i, start, stop, step, span);
    auto forStmt = forLoop.Build(span);
    // Emitting outside any context should throw
    ASSERT_THROW(builder.Emit(forStmt), RuntimeError);
}

// ============================================================================
// If Statement Building Tests
// ============================================================================

TEST(IRBuilderTest, BuildSimpleIf) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("if_func", span);

    auto cond = std::make_shared<ConstBool>(true, span);
    builder.BeginIf(cond, span);
    ASSERT_TRUE(builder.InIf());

    auto ifStmt = builder.EndIf(span);
    ASSERT_NE(ifStmt, nullptr);
    ASSERT_EQ(ifStmt->GetKind(), ObjectKind::IfStmt);
    ASSERT_FALSE(builder.InIf());

    builder.EndFunction(span);
}

TEST(IRBuilderTest, BuildIfElse) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("if_else_func", span);

    auto cond = std::make_shared<ConstBool>(true, span);
    builder.BeginIf(cond, span);

    // Then branch: assign x = 1
    auto x = builder.Var("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    builder.Assign(x, one, span);

    builder.BeginElse(span);

    // Else branch: assign x = 2
    auto two = std::make_shared<ConstInt>(2, DataType::INT32, span);
    builder.Assign(x, two, span);

    auto ifStmt = builder.EndIf(span);
    ASSERT_NE(ifStmt, nullptr);

    builder.EndFunction(span);
}

TEST(IRBuilderTest, DoubleElseThrows) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("double_else", span);

    auto cond = std::make_shared<ConstBool>(true, span);
    builder.BeginIf(cond, span);
    builder.BeginElse(span);
    ASSERT_THROW(builder.BeginElse(span), InternalError);

    builder.EndIf(span);
    builder.EndFunction(span);
}

TEST(IRBuilderTest, BuildIfWithReturnVars) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("if_ret_func", span);

    auto cond = std::make_shared<ConstBool>(true, span);
    builder.BeginIf(cond, span);

    auto retVar = std::make_shared<Var>("result", Int32Type(), span);
    builder.AddIfReturnVar(retVar);

    auto ifStmt = builder.EndIf(span);
    ASSERT_NE(ifStmt, nullptr);

    builder.EndFunction(span);
}

// ============================================================================
// Program Building Tests
// ============================================================================

TEST(IRBuilderTest, BuildSimpleProgram) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginProgram("my_program", span);
    ASSERT_TRUE(builder.InProgram());

    // Build a function
    builder.BeginFunction("func1", span);
    builder.ReturnType(Int32Type());
    auto func1 = builder.EndFunction(span);

    builder.AddFunction(func1);

    auto program = builder.EndProgram(span);
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->name_, "my_program");
    ASSERT_FALSE(builder.InProgram());
}

TEST(IRBuilderTest, ProgramDeclareFunction) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginProgram("prog", span);

    auto gvar = builder.DeclareFunction("func1");
    ASSERT_NE(gvar, nullptr);
    ASSERT_EQ(gvar->name_, "func1");

    // Declare same function again should return same GlobalVar
    auto gvar2 = builder.DeclareFunction("func1");
    ASSERT_EQ(gvar->name_, gvar2->name_);

    builder.EndProgram(span);
}

TEST(IRBuilderTest, ProgramGetGlobalVar) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginProgram("prog", span);
    builder.DeclareFunction("func1");

    auto gvar = builder.GetGlobalVar("func1");
    ASSERT_NE(gvar, nullptr);

    ASSERT_THROW(builder.GetGlobalVar("nonexistent"), RuntimeError);

    builder.EndProgram(span);
}

TEST(IRBuilderTest, NestedProgramThrows) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginProgram("prog1", span);
    ASSERT_THROW(builder.BeginProgram("prog2", span), RuntimeError);
    builder.EndProgram(span);
}

TEST(IRBuilderTest, ProgramGetFunctionReturnTypes) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginProgram("prog", span);
    auto gvar = builder.DeclareFunction("func1");

    builder.BeginFunction("func1", span);
    builder.ReturnType(Int32Type());
    builder.ReturnType(Float32Type());
    auto func = builder.EndFunction(span);
    builder.AddFunction(func);

    auto retTypes = builder.GetFunctionReturnTypes(gvar);
    ASSERT_EQ(retTypes.size(), 2);

    builder.EndProgram(span);
}

// ============================================================================
// Context State Query Tests
// ============================================================================

TEST(IRBuilderTest, VarCreation) {
    IRBuilder builder;
    auto span = TestSpan();
    auto var = builder.Var("test_var", Int32Type(), span);
    ASSERT_NE(var, nullptr);
    ASSERT_EQ(var->name_, "test_var");
}

TEST(IRBuilderTest, MultipleStatementsInFunction) {
    IRBuilder builder;
    auto span = TestSpan();

    builder.BeginFunction("multi_stmt", span);
    auto x = builder.FuncArg("x", Int32Type(), span);
    auto y = builder.Var("y", Int32Type(), span);
    builder.Assign(y, x, span);
    builder.Return({y}, span);
    builder.ReturnType(Int32Type());

    auto func = builder.EndFunction(span);
    ASSERT_NE(func, nullptr);
    // Body should be SeqStmts with 2 statements
    ASSERT_EQ(func->body_->GetKind(), ObjectKind::SeqStmts);
}

// ============================================================================
// ProgramContext Tests
// ============================================================================

TEST(IRBuilderTest, ProgramContextAddStmtThrows) {
    ProgramContext ctx("test", TestSpan());
    auto stmt = std::make_shared<ReturnStmt>(TestSpan());
    ASSERT_THROW(ctx.AddStmt(stmt), InternalError);
}

TEST(IRBuilderTest, ProgramContextGetReturnTypesEmpty) {
    ProgramContext ctx("test", TestSpan());
    auto gvar = std::make_shared<GlobalVar>("nonexistent");
    auto retTypes = ctx.GetReturnTypes(gvar);
    ASSERT_TRUE(retTypes.empty());
}

// ============================================================================
// IRBuilder FuncArg Tests
// ============================================================================

TEST(IRBuilderTest, TestFuncArg) {
    IRBuilder builder;
    auto span = Span::Unknown();
    auto intType = std::make_shared<ScalarType>(DataType::INT32);

    builder.BeginFunction("test", span);
    auto param = builder.FuncArg("x", intType, span);
    ASSERT_NE(param, nullptr);
    ASSERT_EQ(param->name_, "x");

    builder.Return({param}, span);
    auto func = builder.EndFunction(span);
    ASSERT_NE(func, nullptr);
    ASSERT_EQ(func->params_.size(), 1);
}

// ============================================================================
// IRBuilder Error Path Tests
// ============================================================================

TEST(IRBuilderTest, TestBeginFunctionWhileInFunction) {
    IRBuilder builder;
    auto span = Span::Unknown();
    builder.BeginFunction("f1", span);
    ASSERT_THROW(builder.BeginFunction("f2", span), RuntimeError);
    // Clean up
    builder.Return(span);
    builder.EndFunction(span);
}

TEST(IRBuilderTest, TestEmitOutsideContext) {
    IRBuilder builder;
    auto span = Span::Unknown();
    auto val = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stmt = std::make_shared<EvalStmt>(val, span);
    ASSERT_THROW(builder.Emit(stmt), RuntimeError);
}

TEST(IRBuilderTest, TestBeginProgramWhileInProgram) {
    IRBuilder builder;
    auto span = Span::Unknown();
    builder.BeginProgram("p1", span);
    ASSERT_THROW(builder.BeginProgram("p2", span), RuntimeError);
    builder.EndProgram(span);
}

TEST(IRBuilderTest, TestEmitForLoopOutsideContext) {
    IRBuilder builder;
    auto span = Span::Unknown();
    auto intType = std::make_shared<ScalarType>(DataType::INT32);
    auto lv = builder.Var("i", intType, span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);
    // CreateForLoop is stateless - no context required
    auto forLoop = builder.CreateForLoop(lv, start, stop, step, span);
    auto forStmt = forLoop.Build(span);
    // Emitting outside context throws
    ASSERT_THROW(builder.Emit(forStmt), RuntimeError);
}

// ============================================================================
// IRBuilder Empty Return Tests
// ============================================================================

TEST(IRBuilderTest, TestEmptyReturn) {
    IRBuilder builder;
    auto span = Span::Unknown();
    builder.BeginFunction("test", span);
    builder.Return(span);
    auto func = builder.EndFunction(span);
    ASSERT_NE(func, nullptr);
}

// ============================================================================
// IRBuilder Multiple Statements in Body
// ============================================================================

TEST(IRBuilderTest, TestMultipleStatementsInFunction) {
    IRBuilder builder;
    auto span = Span::Unknown();
    auto intType = std::make_shared<ScalarType>(DataType::INT32);

    builder.BeginFunction("test", span);
    auto var = builder.Var("x", intType, span);
    auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, span);
    builder.Assign(var, val1, span);
    builder.Assign(var, val2, span);
    builder.Return({var}, span);
    auto func = builder.EndFunction(span);
    ASSERT_NE(func, nullptr);
}

TEST(IRBuilderTest, TestEmptyFunctionBody) {
    IRBuilder builder;
    auto span = Span::Unknown();
    builder.BeginFunction("empty", span);
    auto func = builder.EndFunction(span);
    ASSERT_NE(func, nullptr);
}

// ============================================================================
// IRBuilder Program with DeclareFunction/GetGlobalVar
// ============================================================================

TEST(IRBuilderTest, TestDeclareFunctionAndGetGlobalVar) {
    IRBuilder builder;
    auto span = Span::Unknown();

    builder.BeginProgram("prog", span);
    auto gvar = builder.DeclareFunction("helper");
    ASSERT_NE(gvar, nullptr);
    ASSERT_EQ(gvar->name_, "helper");

    auto gvar2 = builder.GetGlobalVar("helper");
    ASSERT_NE(gvar2, nullptr);
    ASSERT_EQ(gvar2->name_, "helper");

    builder.EndProgram(span);
}

TEST(IRBuilderTest, TestGetGlobalVarNotFound) {
    IRBuilder builder;
    auto span = Span::Unknown();
    builder.BeginProgram("prog", span);
    ASSERT_THROW(builder.GetGlobalVar("nonexistent"), RuntimeError);
    builder.EndProgram(span);
}

TEST(IRBuilderTest, TestProgramAddFunction) {
    IRBuilder builder;
    auto span = Span::Unknown();
    auto intType = std::make_shared<ScalarType>(DataType::INT32);

    builder.BeginProgram("prog", span);
    builder.DeclareFunction("main");

    builder.BeginFunction("main", span);
    builder.ReturnType(intType);
    auto val = std::make_shared<ConstInt>(0, DataType::INT32, span);
    builder.Return({val}, span);
    auto func = builder.EndFunction(span);
    builder.AddFunction(func);

    auto prog = builder.EndProgram(span);
    ASSERT_NE(prog, nullptr);
    ASSERT_EQ(prog->functions_.size(), 1);
}

TEST(IRBuilderTest, TestGetFunctionReturnTypes) {
    IRBuilder builder;
    auto span = Span::Unknown();
    auto intType = std::make_shared<ScalarType>(DataType::INT32);

    builder.BeginProgram("prog", span);
    auto gvar = builder.DeclareFunction("main");

    builder.BeginFunction("main", span);
    builder.ReturnType(intType);
    auto val = std::make_shared<ConstInt>(0, DataType::INT32, span);
    builder.Return({val}, span);
    auto func = builder.EndFunction(span);
    builder.AddFunction(func);

    auto retTypes = builder.GetFunctionReturnTypes(gvar);
    ASSERT_EQ(retTypes.size(), 1);

    builder.EndProgram(span);
}

// ============================================================================
// IRBuilder ForLoop with IterArgs
// ============================================================================

TEST(IRBuilderTest, TestForLoopWithIterArgs) {
    IRBuilder builder;
    auto span = Span::Unknown();
    auto intType = std::make_shared<ScalarType>(DataType::INT32);

    builder.BeginFunction("test", span);

    auto loopVar = builder.Var("i", intType, span);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, span);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, span);

    auto forLoop = builder.CreateForLoop(loopVar, start, stop, step, span);

    auto init = std::make_shared<ConstInt>(0, DataType::INT32, span);
    auto iterArg = std::make_shared<IterArg>("acc", intType, init, span);
    forLoop.AddIterArg(iterArg);

    auto retVar = builder.Var("acc_out", intType, span);
    forLoop.AddReturnVar(retVar);

    auto forStmt = forLoop.Build(span);
    builder.Emit(forStmt);

    builder.Return(span);
    auto func = builder.EndFunction(span);
    ASSERT_NE(func, nullptr);
}

// ============================================================================
// IRBuilder If with ReturnVars and multiple stmts
// ============================================================================

TEST(IRBuilderTest, TestIfWithReturnVarsAndMultipleStmts) {
    IRBuilder builder;
    auto span = Span::Unknown();
    auto intType = std::make_shared<ScalarType>(DataType::INT32);

    builder.BeginFunction("test", span);

    auto cond = std::make_shared<ConstBool>(true, span);
    builder.BeginIf(cond, span);

    auto var = builder.Var("x", intType, span);
    builder.AddIfReturnVar(var);

    auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, span);
    builder.Assign(var, val1, span);
    builder.Assign(var, val2, span);

    builder.BeginElse(span);
    auto val3 = std::make_shared<ConstInt>(3, DataType::INT32, span);
    auto val4 = std::make_shared<ConstInt>(4, DataType::INT32, span);
    builder.Assign(var, val3, span);
    builder.Assign(var, val4, span);

    builder.EndIf(span);

    builder.Return(span);
    auto func = builder.EndFunction(span);
    ASSERT_NE(func, nullptr);
}

TEST(IRBuilderTest, TestCurrentContextNull) {
    IRBuilder builder;
    ASSERT_FALSE(builder.InFunction());
    ASSERT_FALSE(builder.InIf());
}

} // namespace ir
} // namespace pypto
