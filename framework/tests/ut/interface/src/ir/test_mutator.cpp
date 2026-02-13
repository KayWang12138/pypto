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
 * \file test_mutator.cpp
 * \brief Unit tests for IR mutator (copy-on-write transformation)
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/base/mutator.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class IRMutatorTest : public testing::Test {};

// ============================================================================
// Identity Mutator Tests (no changes, should return same pointers)
// ============================================================================

TEST_F(IRMutatorTest, TestIdentityMutatorConstInt) {
  IRMutator mutator;
  auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto result = mutator.VisitExpr(expr);
  ASSERT_EQ(result.get(), expr.get());  // Same pointer (no copy)
}

TEST_F(IRMutatorTest, TestIdentityMutatorVar) {
  IRMutator mutator;
  auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto result = mutator.VisitExpr(var);
  ASSERT_EQ(result.get(), var.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorAdd) {
  IRMutator mutator;
  auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto add = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

  auto result = mutator.VisitExpr(add);
  ASSERT_EQ(result.get(), add.get());  // No change, same pointer
}

TEST_F(IRMutatorTest, TestIdentityMutatorEvalStmt) {
  IRMutator mutator;
  auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto stmt = std::make_shared<EvalStmt>(expr, Span::unknown());

  auto result = mutator.VisitStmt(stmt);
  ASSERT_EQ(result.get(), stmt.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorAssignStmt) {
  IRMutator mutator;
  auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto val = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto stmt = std::make_shared<AssignStmt>(var, val, Span::unknown());

  auto result = mutator.VisitStmt(stmt);
  ASSERT_EQ(result.get(), stmt.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorSeqStmts) {
  IRMutator mutator;
  auto expr1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto eval1 = std::make_shared<EvalStmt>(expr1, Span::unknown());
  auto expr2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto eval2 = std::make_shared<EvalStmt>(expr2, Span::unknown());

  std::vector<StmtPtr> stmts = {eval1, eval2};
  auto seq = std::make_shared<SeqStmts>(stmts, Span::unknown());

  auto result = mutator.VisitStmt(seq);
  ASSERT_EQ(result.get(), seq.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorReturnStmt) {
  IRMutator mutator;
  auto val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> values = {val};
  auto ret = std::make_shared<ReturnStmt>(values, Span::unknown());

  auto result = mutator.VisitStmt(ret);
  ASSERT_EQ(result.get(), ret.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorYieldStmt) {
  IRMutator mutator;
  auto val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> values = {val};
  auto yield_stmt = std::make_shared<YieldStmt>(values, Span::unknown());

  auto result = mutator.VisitStmt(yield_stmt);
  ASSERT_EQ(result.get(), yield_stmt.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorForStmt) {
  IRMutator mutator;
  auto loop_var = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto start = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto stop = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto step = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto body_expr = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_expr, Span::unknown());

  std::vector<IterArgPtr> iter_args;
  std::vector<VarPtr> return_vars;
  auto for_stmt = std::make_shared<ForStmt>(loop_var, start, stop, step, iter_args, body, return_vars,
                                            Span::unknown());

  auto result = mutator.VisitStmt(for_stmt);
  ASSERT_EQ(result.get(), for_stmt.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorIfStmt) {
  IRMutator mutator;
  auto cond = std::make_shared<ConstBool>(true, Span::unknown());
  auto then_expr = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto then_body = std::make_shared<EvalStmt>(then_expr, Span::unknown());

  std::vector<VarPtr> return_vars;
  auto if_stmt = std::make_shared<IfStmt>(cond, then_body, std::nullopt, return_vars, Span::unknown());

  auto result = mutator.VisitStmt(if_stmt);
  ASSERT_EQ(result.get(), if_stmt.get());
}

// ============================================================================
// Mutator Binary/Unary Expression Tests
// ============================================================================

TEST_F(IRMutatorTest, TestIdentityMutatorSub) {
  IRMutator mutator;
  auto left = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
  auto sub = std::make_shared<Sub>(left, right, DataType::INT32, Span::unknown());

  auto result = mutator.VisitExpr(sub);
  ASSERT_EQ(result.get(), sub.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorMul) {
  IRMutator mutator;
  auto left = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
  auto mul = std::make_shared<Mul>(left, right, DataType::INT32, Span::unknown());

  auto result = mutator.VisitExpr(mul);
  ASSERT_EQ(result.get(), mul.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorNeg) {
  IRMutator mutator;
  auto operand = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
  auto neg = std::make_shared<Neg>(operand, DataType::INT32, Span::unknown());

  auto result = mutator.VisitExpr(neg);
  ASSERT_EQ(result.get(), neg.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorConstFloat) {
  IRMutator mutator;
  auto expr = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  auto result = mutator.VisitExpr(expr);
  ASSERT_EQ(result.get(), expr.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorConstBool) {
  IRMutator mutator;
  auto expr = std::make_shared<ConstBool>(true, Span::unknown());
  auto result = mutator.VisitExpr(expr);
  ASSERT_EQ(result.get(), expr.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorMakeTuple) {
  IRMutator mutator;
  auto elem1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto elem2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> elements = {elem1, elem2};
  auto tuple = std::make_shared<MakeTuple>(elements, Span::unknown());

  auto result = mutator.VisitExpr(tuple);
  ASSERT_EQ(result.get(), tuple.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorOpStmts) {
  IRMutator mutator;
  auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto val = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto assign = std::make_shared<AssignStmt>(var, val, Span::unknown());

  std::vector<StmtPtr> stmts = {assign};
  auto op_stmts = std::make_shared<OpStmts>(stmts, Span::unknown());

  auto result = mutator.VisitStmt(op_stmts);
  ASSERT_EQ(result.get(), op_stmts.get());
}

}  // namespace ir
}  // namespace pypto
