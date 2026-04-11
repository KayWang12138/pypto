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
 * \file test_program_stmt_error.cpp
 * \brief Unit tests for program.cpp, core.cpp, expr.cpp, stmt.cpp, error.cpp, memref.cpp
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

static Span TestSpan() { return Span("test.py", 1, 0); }
static TypePtr Int32Type() { return std::make_shared<ScalarType>(DataType::INT32); }

// ============================================================================
// Program Tests (program.cpp)
// ============================================================================

TEST(ProgramTest, ConstructFromFunctionVector)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto x = std::make_shared<Var>("x", intType, span);
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto func = std::make_shared<Function>("func1", std::vector<VarPtr>{x}, std::vector<TypePtr>{intType}, body, span);

    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "test_prog", span);
    ASSERT_NE(prog, nullptr);
    ASSERT_EQ(prog->name_, "test_prog");
    ASSERT_EQ(prog->functions_.size(), 1u);
}

TEST(ProgramTest, ConstructFromMultipleFunctions)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);

    auto func1 = std::make_shared<Function>("func_a", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);
    auto func2 = std::make_shared<Function>("func_b", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);

    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func1, func2}, "multi_prog", span);
    ASSERT_EQ(prog->functions_.size(), 2u);
}

TEST(ProgramTest, GetFunctionByName)
{
    auto span = TestSpan();
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto func = std::make_shared<Function>("my_func", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "prog", span);

    auto found = prog->GetFunction("my_func");
    ASSERT_NE(found, nullptr);
    ASSERT_EQ(found->name_, "my_func");

    auto notFound = prog->GetFunction("nonexistent");
    ASSERT_EQ(notFound, nullptr);
}

TEST(ProgramTest, GetGlobalVarByName)
{
    auto span = TestSpan();
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto func = std::make_shared<Function>("my_func", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "prog", span);

    auto gvar = prog->GetGlobalVar("my_func");
    ASSERT_NE(gvar, nullptr);
    ASSERT_EQ(gvar->name_, "my_func");

    auto notFound = prog->GetGlobalVar("nonexistent");
    ASSERT_EQ(notFound, nullptr);
}

TEST(ProgramTest, ProgramKindAndTypeName)
{
    auto span = TestSpan();
    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span);
    auto func = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, span);
    auto prog = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "p", span);

    ASSERT_EQ(prog->GetKind(), ObjectKind::Program);
    ASSERT_EQ(prog->TypeName(), "Program");
}

// ============================================================================
// Core / Span Tests (core.cpp)
// ============================================================================

TEST(SpanTest, Construction)
{
    Span span("file.py", 10, 5);
    ASSERT_EQ(span.filename_, "file.py");
    ASSERT_EQ(span.beginLine_, 10);
    ASSERT_EQ(span.beginColumn_, 5);
    ASSERT_EQ(span.endLine_, -1);
    ASSERT_EQ(span.endColumn_, -1);
}

TEST(SpanTest, ConstructionWithEndPos)
{
    Span span("file.py", 10, 5, 20, 15);
    ASSERT_EQ(span.endLine_, 20);
    ASSERT_EQ(span.endColumn_, 15);
}

TEST(SpanTest, ToString)
{
    Span span("file.py", 10, 5);
    auto str = span.ToString();
    ASSERT_NE(str.find("file.py"), std::string::npos);
    ASSERT_NE(str.find("10"), std::string::npos);
    ASSERT_NE(str.find("5"), std::string::npos);
}

TEST(SpanTest, IsValidTrue)
{
    Span span("file.py", 1, 1);
    ASSERT_TRUE(span.IsValid());
}

TEST(SpanTest, IsValidWithEndPos)
{
    Span span("file.py", 1, 1, 5, 10);
    ASSERT_TRUE(span.IsValid());
}

TEST(SpanTest, IsValidFalseNegativeLine)
{
    Span span("file.py", -1, -1, -1, -1);
    ASSERT_FALSE(span.IsValid());
}

TEST(SpanTest, UnknownSpan)
{
    auto span = Span::Unknown();
    ASSERT_FALSE(span.IsValid());
    ASSERT_TRUE(span.filename_.empty());
}

// ============================================================================
// Expr Tests (expr.cpp)
// ============================================================================

TEST(ExprTest, MakeTupleConstruction)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto x = std::make_shared<Var>("x", intType, span);
    auto y = std::make_shared<Var>("y", intType, span);

    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{x, y}, span);
    ASSERT_NE(tuple, nullptr);
    ASSERT_EQ(tuple->elements_.size(), 2u);

    // Type should be TupleType
    auto tupleType = As<TupleType>(tuple->GetType());
    ASSERT_NE(tupleType, nullptr);
    ASSERT_EQ(tupleType->types_.size(), 2u);
}

TEST(ExprTest, TupleGetItemExpr)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto floatType = std::make_shared<ScalarType>(DataType::FP32);
    auto x = std::make_shared<Var>("x", intType, span);
    auto y = std::make_shared<Var>("y", floatType, span);

    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{x, y}, span);
    auto getItem = std::make_shared<TupleGetItemExpr>(tuple, 0, span);

    ASSERT_NE(getItem, nullptr);
    ASSERT_EQ(getItem->index_, 0);
    // Type should be the first element's type (INT32)
    auto ResultType = As<ScalarType>(getItem->GetType());
    ASSERT_NE(ResultType, nullptr);
    ASSERT_EQ(ResultType->dtype_, DataType::INT32);
}

TEST(ExprTest, TupleGetItemSecondElement)
{
    auto span = TestSpan();
    auto intType = Int32Type();
    auto floatType = std::make_shared<ScalarType>(DataType::FP32);
    auto x = std::make_shared<Var>("x", intType, span);
    auto y = std::make_shared<Var>("y", floatType, span);

    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{x, y}, span);
    auto getItem = std::make_shared<TupleGetItemExpr>(tuple, 1, span);

    auto ResultType = As<ScalarType>(getItem->GetType());
    ASSERT_NE(ResultType, nullptr);
    ASSERT_EQ(ResultType->dtype_, DataType::FP32);
}

// ============================================================================
// Stmt Tests (stmt.cpp)
// ============================================================================

TEST(StmtTest, OpStmtsWithAssignAndEval)
{
    auto span = TestSpan();
    auto x = std::make_shared<Var>("x", Int32Type(), span);
    auto one = std::make_shared<ConstInt>(1, DataType::INT32, span);
    auto assign = std::make_shared<AssignStmt>(x, one, span);
    auto eval = std::make_shared<EvalStmt>(one, span);

    auto opStmts = std::make_shared<OpStmts>(std::vector<StmtPtr>{assign, eval}, span);
    ASSERT_NE(opStmts, nullptr);
    ASSERT_EQ(opStmts->stmts_.size(), 2u);
    ASSERT_EQ(opStmts->GetKind(), ObjectKind::OpStmts);
    ASSERT_EQ(opStmts->TypeName(), "OpStmts");
}

TEST(StmtTest, OpStmtsRejectsInvalidStmt)
{
    auto span = TestSpan();
    auto ret = std::make_shared<ReturnStmt>(span);
    // ReturnStmt is not AssignStmt or EvalStmt, should throw
    ASSERT_THROW(std::make_shared<OpStmts>(std::vector<StmtPtr>{ret}, span), InternalError);
}

TEST(StmtTest, OpStmtsEmpty)
{
    auto span = TestSpan();
    auto opStmts = std::make_shared<OpStmts>(std::vector<StmtPtr>{}, span);
    ASSERT_NE(opStmts, nullptr);
    ASSERT_TRUE(opStmts->stmts_.empty());
}

// ============================================================================
// Error Tests (error.cpp)
// ============================================================================

TEST(ErrorTest, RuntimeErrorConstruction)
{
    try {
        throw RuntimeError("test error message");
    } catch (const RuntimeError& e) {
        std::string msg = e.what();
        ASSERT_NE(msg.find("test error message"), std::string::npos);
    }
}

TEST(ErrorTest, ValueErrorConstruction)
{
    try {
        throw ValueError("value error");
    } catch (const ValueError& e) {
        std::string msg = e.what();
        ASSERT_NE(msg.find("value error"), std::string::npos);
    }
}

TEST(ErrorTest, InternalErrorConstruction)
{
    try {
        throw InternalError("internal error");
    } catch (const InternalError& e) {
        std::string msg = e.what();
        ASSERT_NE(msg.find("internal error"), std::string::npos);
    }
}

TEST(ErrorTest, ErrorGetFullMessage)
{
    try {
        throw RuntimeError("test full message");
    } catch (const Error& e) {
        auto fullMsg = e.GetFullMessage();
        ASSERT_NE(fullMsg.find("test full message"), std::string::npos);
    }
}

TEST(ErrorTest, ErrorGetFormattedStackTrace)
{
    try {
        throw RuntimeError("stack trace test");
    } catch (const Error& e) {
        // GetFormattedStackTrace should not crash
        auto trace = e.GetFormattedStackTrace();
        // trace may be empty in release builds
        ASSERT_TRUE(true);
    }
}

// ============================================================================
// MemRef Tests (memref.cpp)
// ============================================================================

TEST(MemRefTest, Construction)
{
    auto span = TestSpan();
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, span);
    auto memref = std::make_shared<MemRef>(MemorySpace::UB, addr, 1024, 0, span);

    ASSERT_NE(memref, nullptr);
    ASSERT_EQ(memref->memorySpace_, MemorySpace::UB);
    ASSERT_EQ(memref->size_, 1024u);
    ASSERT_EQ(memref->id_, 0u);
    ASSERT_EQ(memref->GetKind(), ObjectKind::MemRef);
}

TEST(MemRefTest, MemorySpaceToString)
{
    ASSERT_EQ(MemorySpaceToString(MemorySpace::DDR), "DDR");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::UB), "UB");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::L1), "L1");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::L0A), "L0A");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::L0B), "L0B");
    ASSERT_EQ(MemorySpaceToString(MemorySpace::L0C), "L0C");
}

TEST(MemRefTest, NameContainsMemorySpace)
{
    auto span = TestSpan();
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, span);
    auto memref = std::make_shared<MemRef>(MemorySpace::L1, addr, 512, 1, span);
    // Name should contain lowercase memory space
    ASSERT_NE(memref->name_.find("l1"), std::string::npos);
}

TEST(MemRefTest, DifferentMemorySpaces)
{
    auto span = TestSpan();
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, span);

    auto ub = std::make_shared<MemRef>(MemorySpace::UB, addr, 1024, 0, span);
    auto l0a = std::make_shared<MemRef>(MemorySpace::L0A, addr, 1024, 1, span);
    auto l0b = std::make_shared<MemRef>(MemorySpace::L0B, addr, 1024, 2, span);

    ASSERT_NE(ub->name_, l0a->name_);
    ASSERT_NE(l0a->name_, l0b->name_);
}

} // namespace ir
} // namespace pypto
