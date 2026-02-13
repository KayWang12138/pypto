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
 * \file test_visitor.cpp
 * \brief Unit tests for IR visitor traversal
 */

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/base/visitor.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// A counting visitor that tracks how many nodes are visited
class CountingVisitor : public IRVisitor {
 public:
  using IRVisitor::VisitExpr_;
  using IRVisitor::VisitStmt_;

  int expr_count = 0;
  int stmt_count = 0;

  void VisitExpr_(const ConstIntPtr& op) override {
    expr_count++;
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarPtr& op) override {
    expr_count++;
    IRVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const AddPtr& op) override {
    expr_count++;
    IRVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    stmt_count++;
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    stmt_count++;
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    stmt_count++;
    IRVisitor::VisitStmt_(op);
  }
};

class IRVisitorTest : public testing::Test {};

// ============================================================================
// Visitor Expression Tests
// ============================================================================

TEST_F(IRVisitorTest, TestVisitConstInt) {
  CountingVisitor visitor;
  auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  visitor.VisitExpr(expr);
  ASSERT_EQ(visitor.expr_count, 1);
}

TEST_F(IRVisitorTest, TestVisitBinaryExpr) {
  CountingVisitor visitor;
  auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto add = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

  visitor.VisitExpr(add);
  // Should visit Add + left ConstInt + right ConstInt
  ASSERT_EQ(visitor.expr_count, 3);
}

TEST_F(IRVisitorTest, TestVisitVar) {
  CountingVisitor visitor;
  auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  visitor.VisitExpr(var);
  ASSERT_EQ(visitor.expr_count, 1);
}

// ============================================================================
// Visitor Statement Tests
// ============================================================================

TEST_F(IRVisitorTest, TestVisitEvalStmt) {
  CountingVisitor visitor;
  auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto stmt = std::make_shared<EvalStmt>(expr, Span::unknown());
  visitor.VisitStmt(stmt);
  ASSERT_EQ(visitor.stmt_count, 1);
  ASSERT_EQ(visitor.expr_count, 1);
}

TEST_F(IRVisitorTest, TestVisitAssignStmt) {
  CountingVisitor visitor;
  auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto val = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto stmt = std::make_shared<AssignStmt>(var, val, Span::unknown());
  visitor.VisitStmt(stmt);
  ASSERT_EQ(visitor.stmt_count, 1);
  ASSERT_EQ(visitor.expr_count, 2);  // var + val
}

TEST_F(IRVisitorTest, TestVisitSeqStmts) {
  CountingVisitor visitor;
  auto expr1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto eval1 = std::make_shared<EvalStmt>(expr1, Span::unknown());
  auto expr2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto eval2 = std::make_shared<EvalStmt>(expr2, Span::unknown());

  std::vector<StmtPtr> stmts = {eval1, eval2};
  auto seq = std::make_shared<SeqStmts>(stmts, Span::unknown());
  visitor.VisitStmt(seq);

  ASSERT_EQ(visitor.stmt_count, 3);  // SeqStmts + 2 EvalStmts
  ASSERT_EQ(visitor.expr_count, 2);  // 2 ConstInts
}

}  // namespace ir
}  // namespace pypto
