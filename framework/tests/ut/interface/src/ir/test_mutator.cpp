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
#include <optional>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
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

// ============================================================================
// Additional Binary/Unary/Node Mutator Tests
// ============================================================================

TEST_F(IRMutatorTest, TestIdentityFloorDiv) {
  IRMutator mutator;
  auto l = std::make_shared<ConstInt>(
      10, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      3, DataType::INT32, Span::unknown());
  auto e = std::make_shared<FloorDiv>(
      l, r, DataType::INT32, Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(e).get(), e.get());
}

TEST_F(IRMutatorTest, TestIdentityFloorMod) {
  IRMutator mutator;
  auto l = std::make_shared<ConstInt>(
      10, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      3, DataType::INT32, Span::unknown());
  auto e = std::make_shared<FloorMod>(
      l, r, DataType::INT32, Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(e).get(), e.get());
}

TEST_F(IRMutatorTest, TestIdentityFloatDiv) {
  IRMutator mutator;
  auto l = std::make_shared<ConstFloat>(
      1.0, DataType::FP32, Span::unknown());
  auto r = std::make_shared<ConstFloat>(
      2.0, DataType::FP32, Span::unknown());
  auto e = std::make_shared<FloatDiv>(
      l, r, DataType::FP32, Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(e).get(), e.get());
}

TEST_F(IRMutatorTest, TestIdentityMinMax) {
  IRMutator mutator;
  auto l = std::make_shared<ConstInt>(
      1, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      2, DataType::INT32, Span::unknown());
  auto mn = std::make_shared<Min>(
      l, r, DataType::INT32, Span::unknown());
  auto mx = std::make_shared<Max>(
      l, r, DataType::INT32, Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(mn).get(), mn.get());
  ASSERT_EQ(mutator.VisitExpr(mx).get(), mx.get());
}

TEST_F(IRMutatorTest, TestIdentityPow) {
  IRMutator mutator;
  auto l = std::make_shared<ConstInt>(
      2, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      3, DataType::INT32, Span::unknown());
  auto e = std::make_shared<Pow>(
      l, r, DataType::INT32, Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(e).get(), e.get());
}

TEST_F(IRMutatorTest, TestIdentityComparisonOps) {
  IRMutator mutator;
  auto l = std::make_shared<ConstInt>(
      1, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      2, DataType::INT32, Span::unknown());
  auto eq = std::make_shared<Eq>(
      l, r, DataType::BOOL, Span::unknown());
  auto ne = std::make_shared<Ne>(
      l, r, DataType::BOOL, Span::unknown());
  auto lt = std::make_shared<Lt>(
      l, r, DataType::BOOL, Span::unknown());
  auto le = std::make_shared<Le>(
      l, r, DataType::BOOL, Span::unknown());
  auto gt = std::make_shared<Gt>(
      l, r, DataType::BOOL, Span::unknown());
  auto ge = std::make_shared<Ge>(
      l, r, DataType::BOOL, Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(eq).get(), eq.get());
  ASSERT_EQ(mutator.VisitExpr(ne).get(), ne.get());
  ASSERT_EQ(mutator.VisitExpr(lt).get(), lt.get());
  ASSERT_EQ(mutator.VisitExpr(le).get(), le.get());
  ASSERT_EQ(mutator.VisitExpr(gt).get(), gt.get());
  ASSERT_EQ(mutator.VisitExpr(ge).get(), ge.get());
}

TEST_F(IRMutatorTest, TestIdentityLogicalOps) {
  IRMutator mutator;
  auto l = std::make_shared<ConstBool>(
      true, Span::unknown());
  auto r = std::make_shared<ConstBool>(
      false, Span::unknown());
  auto a = std::make_shared<And>(
      l, r, DataType::BOOL, Span::unknown());
  auto o = std::make_shared<Or>(
      l, r, DataType::BOOL, Span::unknown());
  auto x = std::make_shared<Xor>(
      l, r, DataType::BOOL, Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(a).get(), a.get());
  ASSERT_EQ(mutator.VisitExpr(o).get(), o.get());
  ASSERT_EQ(mutator.VisitExpr(x).get(), x.get());
}

TEST_F(IRMutatorTest, TestIdentityBitwiseOps) {
  IRMutator mutator;
  auto l = std::make_shared<ConstInt>(
      0xFF, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      0x0F, DataType::INT32, Span::unknown());
  auto ba = std::make_shared<BitAnd>(
      l, r, DataType::INT32, Span::unknown());
  auto bo = std::make_shared<BitOr>(
      l, r, DataType::INT32, Span::unknown());
  auto bx = std::make_shared<BitXor>(
      l, r, DataType::INT32, Span::unknown());
  auto sl = std::make_shared<BitShiftLeft>(
      l, r, DataType::INT32, Span::unknown());
  auto sr = std::make_shared<BitShiftRight>(
      l, r, DataType::INT32, Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(ba).get(), ba.get());
  ASSERT_EQ(mutator.VisitExpr(bo).get(), bo.get());
  ASSERT_EQ(mutator.VisitExpr(bx).get(), bx.get());
  ASSERT_EQ(mutator.VisitExpr(sl).get(), sl.get());
  ASSERT_EQ(mutator.VisitExpr(sr).get(), sr.get());
}

TEST_F(IRMutatorTest, TestIdentityUnaryOps) {
  IRMutator mutator;
  auto v = std::make_shared<ConstInt>(
      5, DataType::INT32, Span::unknown());
  auto abs_e = std::make_shared<Abs>(
      v, DataType::INT32, Span::unknown());
  auto not_e = std::make_shared<Not>(
      std::make_shared<ConstBool>(true, Span::unknown()),
      DataType::BOOL, Span::unknown());
  auto bn = std::make_shared<BitNot>(
      v, DataType::INT32, Span::unknown());
  auto cast_e = std::make_shared<Cast>(
      v, DataType::FP32, Span::unknown());
  ASSERT_EQ(
      mutator.VisitExpr(abs_e).get(), abs_e.get());
  ASSERT_EQ(
      mutator.VisitExpr(not_e).get(), not_e.get());
  ASSERT_EQ(mutator.VisitExpr(bn).get(), bn.get());
  ASSERT_EQ(
      mutator.VisitExpr(cast_e).get(), cast_e.get());
}

TEST_F(IRMutatorTest, TestIdentityTupleGetItem) {
  IRMutator mutator;
  auto e1 = std::make_shared<ConstInt>(
      1, DataType::INT32, Span::unknown());
  auto tup = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e1}, Span::unknown());
  auto get = std::make_shared<TupleGetItemExpr>(
      tup, 0,
      Span::unknown());
  ASSERT_EQ(mutator.VisitExpr(get).get(), get.get());
}

TEST_F(IRMutatorTest, TestIdentityCall) {
  IRMutator mutator;
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto a = std::make_shared<Var>(
      "a",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(
                  4, DataType::INT64, sp)},
          DataType::FP32),
      sp);
  auto b = std::make_shared<Var>(
      "b",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(
                  4, DataType::INT64, sp)},
          DataType::FP32),
      sp);
  auto call = reg.Create("tensor.add", {a, b}, sp);
  ASSERT_EQ(mutator.VisitExpr(call).get(), call.get());
}

TEST_F(IRMutatorTest, TestIdentityMemRef) {
  IRMutator mutator;
  auto addr = std::make_shared<ConstInt>(
      0, DataType::INT64, Span::unknown());
  auto mr = std::make_shared<MemRef>(
      MemorySpace::UB, addr, 1024, 0);
  ASSERT_EQ(mutator.VisitExpr(mr).get(), mr.get());
}

TEST_F(IRMutatorTest, TestIdentityIfStmtWithElse) {
  IRMutator mutator;
  Span sp = Span::unknown();
  auto cond = std::make_shared<ConstBool>(true, sp);
  auto then_body = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(1, DataType::INT32, sp),
      sp);
  auto else_body = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(2, DataType::INT32, sp),
      sp);
  auto if_stmt = std::make_shared<IfStmt>(
      cond, then_body,
      std::optional<StmtPtr>(else_body),
      std::vector<VarPtr>{}, sp);
  ASSERT_EQ(
      mutator.VisitStmt(if_stmt).get(), if_stmt.get());
}

}  // namespace ir
}  // namespace pypto
