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
 * \file test_structural_equal.cpp
 * \brief Unit tests for IR structural equality comparison
 */

#include "gtest/gtest.h"

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class StructuralEqualTest : public testing::Test {};

// ============================================================================
// Scalar Expression Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestConstIntEqual) {
  auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto b = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  ASSERT_TRUE(structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestConstIntNotEqual) {
  auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto b = std::make_shared<ConstInt>(99, DataType::INT32, Span::unknown());
  ASSERT_FALSE(structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestConstFloatEqual) {
  auto a = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  auto b = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  ASSERT_TRUE(structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestConstBoolEqual) {
  auto a = std::make_shared<ConstBool>(true, Span::unknown());
  auto b = std::make_shared<ConstBool>(true, Span::unknown());
  ASSERT_TRUE(structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestConstBoolNotEqual) {
  auto a = std::make_shared<ConstBool>(true, Span::unknown());
  auto b = std::make_shared<ConstBool>(false, Span::unknown());
  ASSERT_FALSE(structural_equal(a, b));
}

// ============================================================================
// Binary Expression Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestAddEqual) {
  auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto a = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

  auto left2 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto b = std::make_shared<Add>(left2, right2, DataType::INT32, Span::unknown());

  ASSERT_TRUE(structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestAddNotEqual) {
  auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto a = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

  auto left2 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right2 = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
  auto b = std::make_shared<Add>(left2, right2, DataType::INT32, Span::unknown());

  ASSERT_FALSE(structural_equal(a, b));
}

// ============================================================================
// Statement Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestEvalStmtEqual) {
  auto expr1 = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto stmt1 = std::make_shared<EvalStmt>(expr1, Span::unknown());

  auto expr2 = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto stmt2 = std::make_shared<EvalStmt>(expr2, Span::unknown());

  ASSERT_TRUE(structural_equal(stmt1, stmt2));
}

TEST_F(StructuralEqualTest, TestSeqStmtsEqual) {
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto s1 = std::make_shared<EvalStmt>(e1, Span::unknown());
  auto e2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto s2 = std::make_shared<EvalStmt>(e2, Span::unknown());
  auto seq1 = std::make_shared<SeqStmts>(std::vector<StmtPtr>{s1, s2}, Span::unknown());

  auto e3 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto s3 = std::make_shared<EvalStmt>(e3, Span::unknown());
  auto e4 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto s4 = std::make_shared<EvalStmt>(e4, Span::unknown());
  auto seq2 = std::make_shared<SeqStmts>(std::vector<StmtPtr>{s3, s4}, Span::unknown());

  ASSERT_TRUE(structural_equal(seq1, seq2));
}

// ============================================================================
// Type Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestScalarTypeEqual) {
  auto t1 = std::make_shared<ScalarType>(DataType::FP32);
  auto t2 = std::make_shared<ScalarType>(DataType::FP32);
  ASSERT_TRUE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestScalarTypeNotEqual) {
  auto t1 = std::make_shared<ScalarType>(DataType::FP32);
  auto t2 = std::make_shared<ScalarType>(DataType::INT32);
  ASSERT_FALSE(structural_equal(t1, t2));
}

// ============================================================================
// Function Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestFunctionEqual) {
  auto body1 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown()), Span::unknown());
  auto func1 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body1,
                                          Span::unknown());

  auto body2 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown()), Span::unknown());
  auto func2 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body2,
                                          Span::unknown());

  ASSERT_TRUE(structural_equal(func1, func2));
}

// ============================================================================
// assert_structural_equal Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestAssertStructuralEqualPass) {
  auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto b = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  ASSERT_NO_THROW(assert_structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestAssertStructuralEqualFail) {
  auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto b = std::make_shared<ConstInt>(99, DataType::INT32, Span::unknown());
  ASSERT_THROW(assert_structural_equal(a, b), ValueError);
}

// ============================================================================
// Null Pointer Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestBothNull) {
  ExprPtr a = nullptr;
  ExprPtr b = nullptr;
  ASSERT_TRUE(structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestOneNull) {
  auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  ExprPtr b = nullptr;
  ASSERT_FALSE(structural_equal(a, b));
}

// ============================================================================
// Type Equality Tests (TensorType, TileType, TupleType, UnknownType)
// ============================================================================

TEST_F(StructuralEqualTest, TestTensorTypeEqual) {
  Span sp = Span::unknown();
  std::vector<ExprPtr> shape1 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp),
      std::make_shared<ConstInt>(8, DataType::INT64, sp)};
  auto t1 = std::make_shared<TensorType>(shape1, DataType::FP32);
  std::vector<ExprPtr> shape2 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp),
      std::make_shared<ConstInt>(8, DataType::INT64, sp)};
  auto t2 = std::make_shared<TensorType>(shape2, DataType::FP32);
  ASSERT_TRUE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTensorTypeDtypeMismatch) {
  Span sp = Span::unknown();
  std::vector<ExprPtr> shape = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  auto t1 = std::make_shared<TensorType>(shape, DataType::FP32);
  std::vector<ExprPtr> shape2 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  auto t2 = std::make_shared<TensorType>(shape2, DataType::INT32);
  ASSERT_FALSE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTensorTypeRankMismatch) {
  Span sp = Span::unknown();
  std::vector<ExprPtr> shape1 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  auto t1 = std::make_shared<TensorType>(shape1, DataType::FP32);
  std::vector<ExprPtr> shape2 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp),
      std::make_shared<ConstInt>(8, DataType::INT64, sp)};
  auto t2 = std::make_shared<TensorType>(shape2, DataType::FP32);
  ASSERT_FALSE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTileTypeEqual) {
  Span sp = Span::unknown();
  std::vector<ExprPtr> shape1 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp),
      std::make_shared<ConstInt>(8, DataType::INT64, sp)};
  auto t1 = std::make_shared<TileType>(shape1, DataType::FP32);
  std::vector<ExprPtr> shape2 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp),
      std::make_shared<ConstInt>(8, DataType::INT64, sp)};
  auto t2 = std::make_shared<TileType>(shape2, DataType::FP32);
  ASSERT_TRUE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTileTypeDtypeMismatch) {
  Span sp = Span::unknown();
  std::vector<ExprPtr> s1 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  auto t1 = std::make_shared<TileType>(s1, DataType::FP32);
  std::vector<ExprPtr> s2 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  auto t2 = std::make_shared<TileType>(s2, DataType::INT32);
  ASSERT_FALSE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTileTypeShapeRankMismatch) {
  Span sp = Span::unknown();
  std::vector<ExprPtr> s1 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  auto t1 = std::make_shared<TileType>(s1, DataType::FP32);
  std::vector<ExprPtr> s2 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp),
      std::make_shared<ConstInt>(8, DataType::INT64, sp)};
  auto t2 = std::make_shared<TileType>(s2, DataType::FP32);
  ASSERT_FALSE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTileTypeWithTileView) {
  Span sp = Span::unknown();
  std::vector<ExprPtr> shape = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp),
      std::make_shared<ConstInt>(8, DataType::INT64, sp)};
  std::vector<ExprPtr> vs = {
      std::make_shared<ConstInt>(2, DataType::INT64, sp),
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  std::vector<ExprPtr> stride = {
      std::make_shared<ConstInt>(8, DataType::INT64, sp),
      std::make_shared<ConstInt>(1, DataType::INT64, sp)};
  auto offset = std::make_shared<ConstInt>(0, DataType::INT64, sp);
  TileView tv1(vs, stride, offset);

  std::vector<ExprPtr> shape2 = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp),
      std::make_shared<ConstInt>(8, DataType::INT64, sp)};
  std::vector<ExprPtr> vs2 = {
      std::make_shared<ConstInt>(2, DataType::INT64, sp),
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  std::vector<ExprPtr> stride2 = {
      std::make_shared<ConstInt>(8, DataType::INT64, sp),
      std::make_shared<ConstInt>(1, DataType::INT64, sp)};
  auto offset2 = std::make_shared<ConstInt>(0, DataType::INT64, sp);
  TileView tv2(vs2, stride2, offset2);

  auto t1 = std::make_shared<TileType>(
      shape, DataType::FP32, std::nullopt, std::optional<TileView>(tv1));
  auto t2 = std::make_shared<TileType>(
      shape2, DataType::FP32, std::nullopt, std::optional<TileView>(tv2));
  ASSERT_TRUE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTupleTypeEqual) {
  auto s1 = std::make_shared<ScalarType>(DataType::INT32);
  auto s2 = std::make_shared<ScalarType>(DataType::FP32);
  auto t1 = std::make_shared<TupleType>(std::vector<TypePtr>{s1, s2});
  auto s3 = std::make_shared<ScalarType>(DataType::INT32);
  auto s4 = std::make_shared<ScalarType>(DataType::FP32);
  auto t2 = std::make_shared<TupleType>(std::vector<TypePtr>{s3, s4});
  ASSERT_TRUE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTupleTypeSizeMismatch) {
  auto s1 = std::make_shared<ScalarType>(DataType::INT32);
  auto t1 = std::make_shared<TupleType>(std::vector<TypePtr>{s1});
  auto s2 = std::make_shared<ScalarType>(DataType::INT32);
  auto s3 = std::make_shared<ScalarType>(DataType::FP32);
  auto t2 = std::make_shared<TupleType>(std::vector<TypePtr>{s2, s3});
  ASSERT_FALSE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestUnknownTypeEqual) {
  auto t1 = std::make_shared<UnknownType>();
  auto t2 = std::make_shared<UnknownType>();
  ASSERT_TRUE(structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestTypeMismatchScalarVsTensor) {
  auto t1 = std::make_shared<ScalarType>(DataType::FP32);
  Span sp = Span::unknown();
  std::vector<ExprPtr> shape = {
      std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  auto t2 = std::make_shared<TensorType>(shape, DataType::FP32);
  ASSERT_FALSE(structural_equal(t1, t2));
}

// ============================================================================
// Variable Auto-Mapping Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestVarAutoMapping) {
  Span sp = Span::unknown();
  auto var_x = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto var_y = std::make_shared<Var>(
      "y", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto add1 = std::make_shared<Add>(
      var_x, var_x, DataType::INT32, sp);
  auto add2 = std::make_shared<Add>(
      var_y, var_y, DataType::INT32, sp);
  ASSERT_TRUE(structural_equal(add1, add2, true));
}

TEST_F(StructuralEqualTest, TestVarNoAutoMapping) {
  Span sp = Span::unknown();
  auto var_x = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto var_y = std::make_shared<Var>(
      "y", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto add1 = std::make_shared<Add>(
      var_x, var_x, DataType::INT32, sp);
  auto add2 = std::make_shared<Add>(
      var_y, var_y, DataType::INT32, sp);
  ASSERT_FALSE(structural_equal(add1, add2, false));
}

// ============================================================================
// Additional Expression Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestConstFloatNotEqual) {
  auto a = std::make_shared<ConstFloat>(1.0, DataType::FP32, Span::unknown());
  auto b = std::make_shared<ConstFloat>(2.0, DataType::FP32, Span::unknown());
  ASSERT_FALSE(structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestNodeTypeMismatch) {
  auto a = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto b = std::make_shared<ConstBool>(true, Span::unknown());
  ASSERT_FALSE(structural_equal(a, b));
}

TEST_F(StructuralEqualTest, TestUnaryExprEqual) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<ConstInt>(5, DataType::INT32, sp);
  auto neg1 = std::make_shared<Neg>(v1, DataType::INT32, sp);
  auto v2 = std::make_shared<ConstInt>(5, DataType::INT32, sp);
  auto neg2 = std::make_shared<Neg>(v2, DataType::INT32, sp);
  ASSERT_TRUE(structural_equal(neg1, neg2));
}

TEST_F(StructuralEqualTest, TestMakeTupleEqual) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto e2 = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto tup1 = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e1, e2}, sp);
  auto e3 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto e4 = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto tup2 = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e3, e4}, sp);
  ASSERT_TRUE(structural_equal(tup1, tup2));
}

TEST_F(StructuralEqualTest, TestTupleGetItemEqual) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto tup1 = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e1}, sp);
  auto get1 = std::make_shared<TupleGetItemExpr>(tup1, 0, sp);
  auto e2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto tup2 = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e2}, sp);
  auto get2 = std::make_shared<TupleGetItemExpr>(tup2, 0, sp);
  ASSERT_TRUE(structural_equal(get1, get2));
}

TEST_F(StructuralEqualTest, TestCallEqual) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto a1 = std::make_shared<Var>(
      "a", std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b1 = std::make_shared<Var>(
      "b", std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto call1 = reg.Create("tensor.add", {a1, b1}, sp);

  auto a2 = std::make_shared<Var>(
      "a", std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b2 = std::make_shared<Var>(
      "b", std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto call2 = reg.Create("tensor.add", {a2, b2}, sp);
  ASSERT_TRUE(structural_equal(call1, call2, true));
}

// ============================================================================
// Additional Statement Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestAssignStmtEqual) {
  Span sp = Span::unknown();
  auto var1 = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto val1 = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto stmt1 = std::make_shared<AssignStmt>(var1, val1, sp);
  auto var2 = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto val2 = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto stmt2 = std::make_shared<AssignStmt>(var2, val2, sp);
  ASSERT_TRUE(structural_equal(stmt1, stmt2));
}

TEST_F(StructuralEqualTest, TestReturnStmtEqual) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto ret1 = std::make_shared<ReturnStmt>(
      std::vector<ExprPtr>{v1}, sp);
  auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto ret2 = std::make_shared<ReturnStmt>(
      std::vector<ExprPtr>{v2}, sp);
  ASSERT_TRUE(structural_equal(ret1, ret2));
}

TEST_F(StructuralEqualTest, TestYieldStmtEqual) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto y1 = std::make_shared<YieldStmt>(
      std::vector<ExprPtr>{v1}, sp);
  auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto y2 = std::make_shared<YieldStmt>(
      std::vector<ExprPtr>{v2}, sp);
  ASSERT_TRUE(structural_equal(y1, y2));
}

TEST_F(StructuralEqualTest, TestIfStmtEqual) {
  Span sp = Span::unknown();
  auto cond1 = std::make_shared<ConstBool>(true, sp);
  auto then1 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
  auto else1 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
  auto if1 = std::make_shared<IfStmt>(
      cond1, then1, std::optional<StmtPtr>(else1),
      std::vector<VarPtr>{}, sp);

  auto cond2 = std::make_shared<ConstBool>(true, sp);
  auto then2 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
  auto else2 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
  auto if2 = std::make_shared<IfStmt>(
      cond2, then2, std::optional<StmtPtr>(else2),
      std::vector<VarPtr>{}, sp);
  ASSERT_TRUE(structural_equal(if1, if2));
}

TEST_F(StructuralEqualTest, TestIfStmtNoElseEqual) {
  Span sp = Span::unknown();
  auto cond1 = std::make_shared<ConstBool>(true, sp);
  auto then1 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
  auto if1 = std::make_shared<IfStmt>(
      cond1, then1, std::nullopt,
      std::vector<VarPtr>{}, sp);
  auto cond2 = std::make_shared<ConstBool>(true, sp);
  auto then2 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
  auto if2 = std::make_shared<IfStmt>(
      cond2, then2, std::nullopt,
      std::vector<VarPtr>{}, sp);
  ASSERT_TRUE(structural_equal(if1, if2));
}

TEST_F(StructuralEqualTest, TestForStmtEqual) {
  Span sp = Span::unknown();
  auto lv1 = std::make_shared<Var>(
      "i", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto start1 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto stop1 = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto step1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto body1 = std::make_shared<EvalStmt>(start1, sp);
  auto for1 = std::make_shared<ForStmt>(
      lv1, start1, stop1, step1,
      std::vector<IterArgPtr>{}, body1,
      std::vector<VarPtr>{}, sp);

  auto lv2 = std::make_shared<Var>(
      "i", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto start2 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto stop2 = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto step2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto body2 = std::make_shared<EvalStmt>(start2, sp);
  auto for2 = std::make_shared<ForStmt>(
      lv2, start2, stop2, step2,
      std::vector<IterArgPtr>{}, body2,
      std::vector<VarPtr>{}, sp);
  ASSERT_TRUE(structural_equal(for1, for2));
}

TEST_F(StructuralEqualTest, TestOpStmtsEqual) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto eval1 = std::make_shared<EvalStmt>(e1, sp);
  auto ops1 = std::make_shared<OpStmts>(
      std::vector<StmtPtr>{eval1}, sp);
  auto e2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto eval2 = std::make_shared<EvalStmt>(e2, sp);
  auto ops2 = std::make_shared<OpStmts>(
      std::vector<StmtPtr>{eval2}, sp);
  ASSERT_TRUE(structural_equal(ops1, ops2));
}

TEST_F(StructuralEqualTest, TestProgramEqual) {
  Span sp = Span::unknown();
  auto body1 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
  auto func1 = std::make_shared<Function>(
      "main", std::vector<VarPtr>{}, std::vector<TypePtr>{},
      body1, sp);
  auto prog1 = std::make_shared<Program>(
      std::vector<FunctionPtr>{func1}, "prog", sp);

  auto body2 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
  auto func2 = std::make_shared<Function>(
      "main", std::vector<VarPtr>{}, std::vector<TypePtr>{},
      body2, sp);
  auto prog2 = std::make_shared<Program>(
      std::vector<FunctionPtr>{func2}, "prog", sp);
  ASSERT_TRUE(structural_equal(prog1, prog2));
}

// ============================================================================
// assert_structural_equal with Types
// ============================================================================

TEST_F(StructuralEqualTest, TestAssertStructuralEqualTypePass) {
  auto t1 = std::make_shared<ScalarType>(DataType::FP32);
  auto t2 = std::make_shared<ScalarType>(DataType::FP32);
  ASSERT_NO_THROW(assert_structural_equal(t1, t2));
}

TEST_F(StructuralEqualTest, TestAssertStructuralEqualTypeFail) {
  auto t1 = std::make_shared<ScalarType>(DataType::FP32);
  auto t2 = std::make_shared<ScalarType>(DataType::INT32);
  ASSERT_THROW(assert_structural_equal(t1, t2), ValueError);
}

TEST_F(StructuralEqualTest, TestAssertStructuralEqualNodeTypeMismatch) {
  auto a = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto b = std::make_shared<ConstBool>(true, Span::unknown());
  ASSERT_THROW(assert_structural_equal(a, b), ValueError);
}

// ============================================================================
// IterArg Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestIterArgEqual) {
  Span sp = Span::unknown();
  auto init1 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto ia1 = std::make_shared<IterArg>(
      "acc", std::make_shared<ScalarType>(DataType::INT32),
      init1, sp);
  auto init2 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto ia2 = std::make_shared<IterArg>(
      "acc", std::make_shared<ScalarType>(DataType::INT32),
      init2, sp);
  ASSERT_TRUE(structural_equal(ia1, ia2, true));
}

TEST_F(StructuralEqualTest, TestIterArgNotEqual) {
  Span sp = Span::unknown();
  auto init1 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto ia1 = std::make_shared<IterArg>(
      "acc", std::make_shared<ScalarType>(DataType::INT32),
      init1, sp);
  auto init2 = std::make_shared<ConstInt>(99, DataType::INT32, sp);
  auto ia2 = std::make_shared<IterArg>(
      "acc", std::make_shared<ScalarType>(DataType::INT32),
      init2, sp);
  ASSERT_FALSE(structural_equal(ia1, ia2, true));
}

// ============================================================================
// Call with Kwargs Equality Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestCallWithKwargsEqual) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto tile1 = std::make_shared<Var>(
      "t", std::make_shared<TileType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp),
              std::make_shared<ConstInt>(8, DataType::INT64, sp)},
          DataType::FP32), sp);
  std::vector<std::pair<std::string, std::any>> kw1 = {
      {"axis", std::any(1)}, {"keepdim", std::any(true)}};
  auto call1 = reg.Create("block.sum", {tile1}, kw1, sp);

  auto tile2 = std::make_shared<Var>(
      "t", std::make_shared<TileType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp),
              std::make_shared<ConstInt>(8, DataType::INT64, sp)},
          DataType::FP32), sp);
  std::vector<std::pair<std::string, std::any>> kw2 = {
      {"axis", std::any(1)}, {"keepdim", std::any(true)}};
  auto call2 = reg.Create("block.sum", {tile2}, kw2, sp);
  ASSERT_TRUE(structural_equal(call1, call2, true));
}

TEST_F(StructuralEqualTest, TestCallWithKwargsNotEqual) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto tile1 = std::make_shared<Var>(
      "t", std::make_shared<TileType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp),
              std::make_shared<ConstInt>(8, DataType::INT64, sp)},
          DataType::FP32), sp);
  std::vector<std::pair<std::string, std::any>> kw1 = {
      {"axis", std::any(0)}, {"keepdim", std::any(false)}};
  auto call1 = reg.Create("block.sum", {tile1}, kw1, sp);

  auto tile2 = std::make_shared<Var>(
      "t", std::make_shared<TileType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp),
              std::make_shared<ConstInt>(8, DataType::INT64, sp)},
          DataType::FP32), sp);
  std::vector<std::pair<std::string, std::any>> kw2 = {
      {"axis", std::any(1)}, {"keepdim", std::any(true)}};
  auto call2 = reg.Create("block.sum", {tile2}, kw2, sp);
  ASSERT_FALSE(structural_equal(call1, call2, true));
}

// ============================================================================
// Vector Size Mismatch Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestSeqStmtsSizeMismatch) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto s1 = std::make_shared<EvalStmt>(e1, sp);
  auto seq1 = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{s1}, sp);

  auto e2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto s2 = std::make_shared<EvalStmt>(e2, sp);
  auto e3 = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto s3 = std::make_shared<EvalStmt>(e3, sp);
  auto seq2 = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{s2, s3}, sp);
  ASSERT_FALSE(structural_equal(seq1, seq2));
}

// ============================================================================
// Same Pointer Tests
// ============================================================================

TEST_F(StructuralEqualTest, TestSamePointer) {
  auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  ASSERT_TRUE(structural_equal(a, a));
}

// ============================================================================
// Function with Params Equality
// ============================================================================

TEST_F(StructuralEqualTest, TestFunctionWithParamsEqual) {
  Span sp = Span::unknown();
  auto p1 = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto body1 = std::make_shared<ReturnStmt>(
      std::vector<ExprPtr>{p1}, sp);
  auto ret_t1 = std::make_shared<ScalarType>(DataType::INT32);
  auto func1 = std::make_shared<Function>(
      "f", std::vector<VarPtr>{p1},
      std::vector<TypePtr>{ret_t1}, body1, sp);

  auto p2 = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto body2 = std::make_shared<ReturnStmt>(
      std::vector<ExprPtr>{p2}, sp);
  auto ret_t2 = std::make_shared<ScalarType>(DataType::INT32);
  auto func2 = std::make_shared<Function>(
      "f", std::vector<VarPtr>{p2},
      std::vector<TypePtr>{ret_t2}, body2, sp);
  ASSERT_TRUE(structural_equal(func1, func2));
}

}  // namespace ir
}  // namespace pypto
