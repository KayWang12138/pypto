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
 * \file test_stmt.cpp
 * \brief Unit tests for IR statement types
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class IRStmtTest : public testing::Test {};

// ============================================================================
// OpStmts Tests
// ============================================================================

TEST_F(IRStmtTest, TestOpStmtsWithAssignStmts) {
  auto var1 = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::FP32), Span::unknown());
  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto assign1 = std::make_shared<AssignStmt>(var1, val1, Span::unknown());

  auto var2 = std::make_shared<Var>("y", std::make_shared<ScalarType>(DataType::FP32), Span::unknown());
  auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto assign2 = std::make_shared<AssignStmt>(var2, val2, Span::unknown());

  std::vector<StmtPtr> stmts = {assign1, assign2};
  auto opStmts = std::make_shared<OpStmts>(stmts, Span::unknown());

  ASSERT_EQ(opStmts->stmts_.size(), 2);
}

TEST_F(IRStmtTest, TestOpStmtsWithEvalStmt) {
  auto val = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto evalStmt = std::make_shared<EvalStmt>(val, Span::unknown());

  std::vector<StmtPtr> stmts = {evalStmt};
  auto opStmts = std::make_shared<OpStmts>(stmts, Span::unknown());

  ASSERT_EQ(opStmts->stmts_.size(), 1);
}

TEST_F(IRStmtTest, TestOpStmtsEmpty) {
  std::vector<StmtPtr> stmts;
  auto opStmts = std::make_shared<OpStmts>(stmts, Span::unknown());
  ASSERT_EQ(opStmts->stmts_.size(), 0);
}

// ============================================================================
// AssignStmt Tests
// ============================================================================

TEST_F(IRStmtTest, TestAssignStmtBasic) {
  auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto val = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto stmt = std::make_shared<AssignStmt>(var, val, Span::unknown());

  ASSERT_EQ(stmt->var_, var);
  ASSERT_EQ(stmt->value_, val);
  ASSERT_EQ(stmt->GetKind(), ObjectKind::AssignStmt);
}

// ============================================================================
// EvalStmt Tests
// ============================================================================

TEST_F(IRStmtTest, TestEvalStmtBasic) {
  auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto stmt = std::make_shared<EvalStmt>(expr, Span::unknown());

  ASSERT_EQ(stmt->expr_, expr);
  ASSERT_EQ(stmt->GetKind(), ObjectKind::EvalStmt);
}

// ============================================================================
// SeqStmts Tests
// ============================================================================

TEST_F(IRStmtTest, TestSeqStmtsBasic) {
  auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto val = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto assign = std::make_shared<AssignStmt>(var, val, Span::unknown());

  auto expr = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto eval = std::make_shared<EvalStmt>(expr, Span::unknown());

  std::vector<StmtPtr> stmts = {assign, eval};
  auto seq = std::make_shared<SeqStmts>(stmts, Span::unknown());

  ASSERT_EQ(seq->stmts_.size(), 2);
  ASSERT_EQ(seq->GetKind(), ObjectKind::SeqStmts);
}

// ============================================================================
// ReturnStmt Tests
// ============================================================================

TEST_F(IRStmtTest, TestReturnStmtBasic) {
  auto val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> values = {val};
  auto ret = std::make_shared<ReturnStmt>(values, Span::unknown());

  ASSERT_EQ(ret->value_.size(), 1);
  ASSERT_EQ(ret->GetKind(), ObjectKind::ReturnStmt);
}

TEST_F(IRStmtTest, TestReturnStmtMultipleValues) {
  auto val1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto val2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> values = {val1, val2};
  auto ret = std::make_shared<ReturnStmt>(values, Span::unknown());

  ASSERT_EQ(ret->value_.size(), 2);
}

// ============================================================================
// YieldStmt Tests
// ============================================================================

TEST_F(IRStmtTest, TestYieldStmtBasic) {
  auto val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  std::vector<ExprPtr> values = {val};
  auto yieldStmt = std::make_shared<YieldStmt>(values, Span::unknown());

  ASSERT_EQ(yieldStmt->value_.size(), 1);
  ASSERT_EQ(yieldStmt->GetKind(), ObjectKind::YieldStmt);
}

// ============================================================================
// ForStmt Tests
// ============================================================================

TEST_F(IRStmtTest, TestForStmtBasic) {
  auto loopVar = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto start = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto stop = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto step = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());

  auto bodyExpr = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(bodyExpr, Span::unknown());

  std::vector<IterArgPtr> iterArgs;
  std::vector<VarPtr> returnVars;

  auto forStmt = std::make_shared<ForStmt>(loopVar, start, stop, step, iterArgs, body, returnVars,
                                            Span::unknown());

  ASSERT_EQ(forStmt->loopVar_, loopVar);
  ASSERT_EQ(forStmt->GetKind(), ObjectKind::ForStmt);
}

// ============================================================================
// IfStmt Tests
// ============================================================================

TEST_F(IRStmtTest, TestIfStmtBasic) {
  auto cond = std::make_shared<ConstBool>(true, Span::unknown());
  auto thenExpr = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto thenBody = std::make_shared<EvalStmt>(thenExpr, Span::unknown());

  std::vector<VarPtr> returnVars;
  auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::nullopt, returnVars, Span::unknown());

  ASSERT_EQ(ifStmt->condition_, cond);
  ASSERT_EQ(ifStmt->GetKind(), ObjectKind::IfStmt);
  ASSERT_FALSE(ifStmt->elseBody_.has_value());
}

TEST_F(IRStmtTest, TestIfStmtWithElse) {
  auto cond = std::make_shared<ConstBool>(true, Span::unknown());
  auto thenExpr = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto thenBody = std::make_shared<EvalStmt>(thenExpr, Span::unknown());
  auto elseExpr = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto elseBody = std::make_shared<EvalStmt>(elseExpr, Span::unknown());

  std::vector<VarPtr> returnVars;
  auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, elseBody, returnVars, Span::unknown());

  ASSERT_TRUE(ifStmt->elseBody_.has_value());
}

}  // namespace ir
}  // namespace pypto
