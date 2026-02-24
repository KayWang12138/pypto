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
 * \file test_structural_hash.cpp
 * \brief Unit tests for IR structural hashing
 */

#include "gtest/gtest.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class StructuralHashTest : public testing::Test {};

// ============================================================================
// Hash Consistency Tests
// ============================================================================

TEST_F(StructuralHashTest, TestConstIntHashConsistent) {
  auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto b = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  ASSERT_EQ(structural_hash(a), structural_hash(b));
}

TEST_F(StructuralHashTest, TestConstIntHashDifferent) {
  auto a = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto b = std::make_shared<ConstInt>(99, DataType::INT32, Span::unknown());
  ASSERT_NE(structural_hash(a), structural_hash(b));
}

TEST_F(StructuralHashTest, TestConstFloatHashConsistent) {
  auto a = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  auto b = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  ASSERT_EQ(structural_hash(a), structural_hash(b));
}

TEST_F(StructuralHashTest, TestConstBoolHashConsistent) {
  auto a = std::make_shared<ConstBool>(true, Span::unknown());
  auto b = std::make_shared<ConstBool>(true, Span::unknown());
  ASSERT_EQ(structural_hash(a), structural_hash(b));
}

TEST_F(StructuralHashTest, TestConstBoolHashDifferent) {
  auto a = std::make_shared<ConstBool>(true, Span::unknown());
  auto b = std::make_shared<ConstBool>(false, Span::unknown());
  ASSERT_NE(structural_hash(a), structural_hash(b));
}

// ============================================================================
// Binary Expression Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestAddHashConsistent) {
  auto left1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right1 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto add1 = std::make_shared<Add>(left1, right1, DataType::INT32, Span::unknown());

  auto left2 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto add2 = std::make_shared<Add>(left2, right2, DataType::INT32, Span::unknown());

  ASSERT_EQ(structural_hash(add1), structural_hash(add2));
}

TEST_F(StructuralHashTest, TestDifferentOpsHashDifferent) {
  auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto add = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

  auto left2 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto sub = std::make_shared<Sub>(left2, right2, DataType::INT32, Span::unknown());

  ASSERT_NE(structural_hash(add), structural_hash(sub));
}

// ============================================================================
// Statement Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestEvalStmtHashConsistent) {
  auto expr1 = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto stmt1 = std::make_shared<EvalStmt>(expr1, Span::unknown());

  auto expr2 = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto stmt2 = std::make_shared<EvalStmt>(expr2, Span::unknown());

  ASSERT_EQ(structural_hash(stmt1), structural_hash(stmt2));
}

// ============================================================================
// Type Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestScalarTypeHashConsistent) {
  auto t1 = std::make_shared<ScalarType>(DataType::FP32);
  auto t2 = std::make_shared<ScalarType>(DataType::FP32);
  ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

TEST_F(StructuralHashTest, TestScalarTypeHashDifferent) {
  auto t1 = std::make_shared<ScalarType>(DataType::FP32);
  auto t2 = std::make_shared<ScalarType>(DataType::INT32);
  ASSERT_NE(structural_hash(t1), structural_hash(t2));
}

// ============================================================================
// Function Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestFunctionHashConsistent) {
  auto body1 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown()), Span::unknown());
  auto func1 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body1,
                                          Span::unknown());

  auto body2 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown()), Span::unknown());
  auto func2 = std::make_shared<Function>("f", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body2,
                                          Span::unknown());

  ASSERT_EQ(structural_hash(func1), structural_hash(func2));
}

// ============================================================================
// Unary Expression Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestNegHashConsistent) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<ConstInt>(5, DataType::INT32, sp);
  auto neg1 = std::make_shared<Neg>(v1, DataType::INT32, sp);
  auto v2 = std::make_shared<ConstInt>(5, DataType::INT32, sp);
  auto neg2 = std::make_shared<Neg>(v2, DataType::INT32, sp);
  ASSERT_EQ(structural_hash(neg1), structural_hash(neg2));
}

TEST_F(StructuralHashTest, TestCastHashConsistent) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<ConstInt>(5, DataType::INT32, sp);
  auto c1 = std::make_shared<Cast>(v1, DataType::FP32, sp);
  auto v2 = std::make_shared<ConstInt>(5, DataType::INT32, sp);
  auto c2 = std::make_shared<Cast>(v2, DataType::FP32, sp);
  ASSERT_EQ(structural_hash(c1), structural_hash(c2));
}

// ============================================================================
// Tuple Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestMakeTupleHashConsistent) {
  Span sp = Span::unknown();
  auto e1a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto e2a = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto t1 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1a, e2a}, sp);

  auto e1b = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto e2b = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto t2 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1b, e2b}, sp);
  ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

TEST_F(StructuralHashTest, TestTupleGetItemHashConsistent) {
  Span sp = Span::unknown();
  auto e1a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto t1 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1a}, sp);
  auto g1 = std::make_shared<TupleGetItemExpr>(t1, 0, sp);

  auto e1b = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto t2 = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1b}, sp);
  auto g2 = std::make_shared<TupleGetItemExpr>(t2, 0, sp);
  ASSERT_EQ(structural_hash(g1), structural_hash(g2));
}

// ============================================================================
// Type Hash Tests (TensorType, TileType, TupleType, UnknownType)
// ============================================================================

TEST_F(StructuralHashTest, TestTensorTypeHashConsistent) {
  Span sp = Span::unknown();
  auto t1 = std::make_shared<TensorType>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32);
  auto t2 = std::make_shared<TensorType>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32);
  ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

TEST_F(StructuralHashTest, TestTensorTypeHashDifferentDtype) {
  Span sp = Span::unknown();
  auto t1 = std::make_shared<TensorType>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32);
  auto t2 = std::make_shared<TensorType>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP16);
  ASSERT_NE(structural_hash(t1), structural_hash(t2));
}

TEST_F(StructuralHashTest, TestTileTypeHashConsistent) {
  Span sp = Span::unknown();
  auto t1 = std::make_shared<TileType>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32);
  auto t2 = std::make_shared<TileType>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32);
  ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

TEST_F(StructuralHashTest, TestTupleTypeHashConsistent) {
  auto s1 = std::make_shared<ScalarType>(DataType::INT32);
  auto s2 = std::make_shared<ScalarType>(DataType::FP32);
  auto tt1 = std::make_shared<TupleType>(std::vector<TypePtr>{s1, s2});
  auto tt2 = std::make_shared<TupleType>(
      std::vector<TypePtr>{std::make_shared<ScalarType>(DataType::INT32),
                           std::make_shared<ScalarType>(DataType::FP32)});
  ASSERT_EQ(structural_hash(tt1), structural_hash(tt2));
}

TEST_F(StructuralHashTest, TestUnknownTypeHash) {
  auto t1 = GetUnknownType();
  auto t2 = GetUnknownType();
  ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

// ============================================================================
// Variable Hash Tests (auto-mapping)
// ============================================================================

TEST_F(StructuralHashTest, TestVarHashAutoMapping) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto v2 = std::make_shared<Var>("y", std::make_shared<ScalarType>(DataType::INT32), sp);
  // With auto-mapping, structurally equivalent vars should hash the same
  ASSERT_EQ(structural_hash(v1, true), structural_hash(v2, true));
}

TEST_F(StructuralHashTest, TestVarHashNoAutoMapping) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto v2 = std::make_shared<Var>("y", std::make_shared<ScalarType>(DataType::INT32), sp);
  // Without auto-mapping, different pointers should hash differently
  ASSERT_NE(structural_hash(v1, false), structural_hash(v2, false));
}

// ============================================================================
// Statement Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestAssignStmtHashConsistent) {
  Span sp = Span::unknown();
  auto var1 = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto val1 = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto s1 = std::make_shared<AssignStmt>(var1, val1, sp);

  auto var2 = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto val2 = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto s2 = std::make_shared<AssignStmt>(var2, val2, sp);
  ASSERT_EQ(structural_hash(s1, true), structural_hash(s2, true));
}

TEST_F(StructuralHashTest, TestSeqStmtsHashConsistent) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto e2 = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto s1 = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{std::make_shared<EvalStmt>(e1, sp), std::make_shared<EvalStmt>(e2, sp)}, sp);

  auto e3 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto e4 = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto s2 = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{std::make_shared<EvalStmt>(e3, sp), std::make_shared<EvalStmt>(e4, sp)}, sp);
  ASSERT_EQ(structural_hash(s1), structural_hash(s2));
}

TEST_F(StructuralHashTest, TestReturnStmtHash) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto r1 = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{v1}, sp);
  auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto r2 = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{v2}, sp);
  ASSERT_EQ(structural_hash(r1), structural_hash(r2));
}

TEST_F(StructuralHashTest, TestYieldStmtHash) {
  Span sp = Span::unknown();
  auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto y1 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{v1}, sp);
  auto v2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto y2 = std::make_shared<YieldStmt>(std::vector<ExprPtr>{v2}, sp);
  ASSERT_EQ(structural_hash(y1), structural_hash(y2));
}

TEST_F(StructuralHashTest, TestIfStmtHash) {
  Span sp = Span::unknown();
  auto cond = std::make_shared<ConstBool>(true, sp);
  auto then_body = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
  auto if1 = std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>{},
                                       std::vector<VarPtr>{}, sp);

  auto cond2 = std::make_shared<ConstBool>(true, sp);
  auto then_body2 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
  auto if2 = std::make_shared<IfStmt>(cond2, then_body2, std::optional<StmtPtr>{},
                                       std::vector<VarPtr>{}, sp);
  ASSERT_EQ(structural_hash(if1), structural_hash(if2));
}

TEST_F(StructuralHashTest, TestForStmtHash) {
  Span sp = Span::unknown();
  auto lv1 = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto start1 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto stop1 = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto step1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto body1 = std::make_shared<EvalStmt>(start1, sp);
  auto for1 = std::make_shared<ForStmt>(lv1, start1, stop1, step1,
      std::vector<IterArgPtr>{}, body1, std::vector<VarPtr>{}, sp);

  auto lv2 = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto start2 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto stop2 = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto step2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto body2 = std::make_shared<EvalStmt>(start2, sp);
  auto for2 = std::make_shared<ForStmt>(lv2, start2, stop2, step2,
      std::vector<IterArgPtr>{}, body2, std::vector<VarPtr>{}, sp);
  ASSERT_EQ(structural_hash(for1, true), structural_hash(for2, true));
}

TEST_F(StructuralHashTest, TestOpStmtsHash) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto s1 = std::make_shared<OpStmts>(
      std::vector<StmtPtr>{std::make_shared<EvalStmt>(e1, sp)}, sp);
  auto e2 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto s2 = std::make_shared<OpStmts>(
      std::vector<StmtPtr>{std::make_shared<EvalStmt>(e2, sp)}, sp);
  ASSERT_EQ(structural_hash(s1), structural_hash(s2));
}

// ============================================================================
// Program Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestProgramHashConsistent) {
  Span sp = Span::unknown();
  auto body1 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
  auto func1 = std::make_shared<Function>("main", std::vector<VarPtr>{},
      std::vector<TypePtr>{}, body1, sp);
  auto prog1 = std::make_shared<Program>(
      std::vector<FunctionPtr>{func1}, "test", sp);

  auto body2 = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
  auto func2 = std::make_shared<Function>("main", std::vector<VarPtr>{},
      std::vector<TypePtr>{}, body2, sp);
  auto prog2 = std::make_shared<Program>(
      std::vector<FunctionPtr>{func2}, "test", sp);
  ASSERT_EQ(structural_hash(prog1, true), structural_hash(prog2, true));
}

// ============================================================================
// Call Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestCallHashConsistent) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto a1 = std::make_shared<Var>("a",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b1 = std::make_shared<Var>("b",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto call1 = reg.Create("tensor.add", {a1, b1}, sp);

  auto a2 = std::make_shared<Var>("a",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b2 = std::make_shared<Var>("b",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto call2 = reg.Create("tensor.add", {a2, b2}, sp);
  ASSERT_EQ(structural_hash(call1, true), structural_hash(call2, true));
}

TEST_F(StructuralHashTest, TestCallWithKwargsHash) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto tile1 = std::make_shared<Var>("t",
      std::make_shared<TileType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp),
              std::make_shared<ConstInt>(8, DataType::INT64, sp)},
          DataType::FP32), sp);
  std::vector<std::pair<std::string, std::any>> kwargs1 = {
      {"axis", std::any(1)}, {"keepdim", std::any(true)}};
  auto call1 = reg.Create("block.sum", {tile1}, kwargs1, sp);

  auto tile2 = std::make_shared<Var>("t",
      std::make_shared<TileType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp),
              std::make_shared<ConstInt>(8, DataType::INT64, sp)},
          DataType::FP32), sp);
  std::vector<std::pair<std::string, std::any>> kwargs2 = {
      {"axis", std::any(1)}, {"keepdim", std::any(true)}};
  auto call2 = reg.Create("block.sum", {tile2}, kwargs2, sp);
  ASSERT_EQ(structural_hash(call1, true), structural_hash(call2, true));
}

// ============================================================================
// IterArg Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestIterArgHash) {
  Span sp = Span::unknown();
  auto init1 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto ia1 = std::make_shared<IterArg>("acc",
      std::make_shared<ScalarType>(DataType::INT32), init1, sp);
  auto init2 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto ia2 = std::make_shared<IterArg>("acc",
      std::make_shared<ScalarType>(DataType::INT32), init2, sp);
  ASSERT_EQ(structural_hash(ia1, true), structural_hash(ia2, true));
}

// ============================================================================
// Kwargs Hash Tests (different value types)
// ============================================================================

TEST_F(StructuralHashTest, TestCallWithStringKwarg) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  // Create a base call to get a valid OpPtr
  auto a = std::make_shared<Var>("a",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b = std::make_shared<Var>("b",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto base_call = As<Call>(reg.Create("tensor.add", {a, b}, sp));
  ASSERT_NE(base_call, nullptr);

  // Create custom Calls with string kwargs
  std::vector<std::pair<std::string, std::any>> kwargs1 = {
      {"label", std::any(std::string("test"))}};
  auto call1 = std::make_shared<const Call>(
      base_call->op_, std::vector<ExprPtr>{a}, kwargs1,
      std::make_shared<ScalarType>(DataType::INT32), sp);

  std::vector<std::pair<std::string, std::any>> kwargs2 = {
      {"label", std::any(std::string("test"))}};
  auto call2 = std::make_shared<const Call>(
      base_call->op_, std::vector<ExprPtr>{a}, kwargs2,
      std::make_shared<ScalarType>(DataType::INT32), sp);

  ASSERT_EQ(structural_hash(ExprPtr(call1), true), structural_hash(ExprPtr(call2), true));
}

TEST_F(StructuralHashTest, TestCallWithDoubleKwarg) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto a = std::make_shared<Var>("a",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b = std::make_shared<Var>("b",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto base_call = As<Call>(reg.Create("tensor.add", {a, b}, sp));

  std::vector<std::pair<std::string, std::any>> kwargs1 = {
      {"scale", std::any(3.14)}};
  auto call1 = std::make_shared<const Call>(
      base_call->op_, std::vector<ExprPtr>{a}, kwargs1,
      std::make_shared<ScalarType>(DataType::FP32), sp);

  std::vector<std::pair<std::string, std::any>> kwargs2 = {
      {"scale", std::any(3.14)}};
  auto call2 = std::make_shared<const Call>(
      base_call->op_, std::vector<ExprPtr>{a}, kwargs2,
      std::make_shared<ScalarType>(DataType::FP32), sp);

  ASSERT_EQ(structural_hash(ExprPtr(call1), true), structural_hash(ExprPtr(call2), true));
}

TEST_F(StructuralHashTest, TestCallWithFloatKwarg) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto a = std::make_shared<Var>("a",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b = std::make_shared<Var>("b",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto base_call = As<Call>(reg.Create("tensor.add", {a, b}, sp));

  std::vector<std::pair<std::string, std::any>> kwargs1 = {
      {"epsilon", std::any(1.0f)}};
  auto call1 = std::make_shared<const Call>(
      base_call->op_, std::vector<ExprPtr>{a}, kwargs1,
      std::make_shared<ScalarType>(DataType::FP32), sp);

  std::vector<std::pair<std::string, std::any>> kwargs2 = {
      {"epsilon", std::any(1.0f)}};
  auto call2 = std::make_shared<const Call>(
      base_call->op_, std::vector<ExprPtr>{a}, kwargs2,
      std::make_shared<ScalarType>(DataType::FP32), sp);

  ASSERT_EQ(structural_hash(ExprPtr(call1), true), structural_hash(ExprPtr(call2), true));
}

TEST_F(StructuralHashTest, TestCallWithDataTypeKwarg) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto a = std::make_shared<Var>("a",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b = std::make_shared<Var>("b",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto base_call = As<Call>(reg.Create("tensor.add", {a, b}, sp));

  std::vector<std::pair<std::string, std::any>> kwargs1 = {
      {"dtype", std::any(DataType::FP16)}};
  auto call1 = std::make_shared<const Call>(
      base_call->op_, std::vector<ExprPtr>{a}, kwargs1,
      std::make_shared<ScalarType>(DataType::FP32), sp);

  std::vector<std::pair<std::string, std::any>> kwargs2 = {
      {"dtype", std::any(DataType::FP16)}};
  auto call2 = std::make_shared<const Call>(
      base_call->op_, std::vector<ExprPtr>{a}, kwargs2,
      std::make_shared<ScalarType>(DataType::FP32), sp);

  ASSERT_EQ(structural_hash(ExprPtr(call1), true), structural_hash(ExprPtr(call2), true));
}

// ============================================================================
// TileType with TileView Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestTileTypeWithTileViewHash) {
  Span sp = Span::unknown();
  auto shape1 = std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  TileView tv1(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT64, sp)},
      std::make_shared<ConstInt>(0, DataType::INT64, sp));
  auto t1 = std::make_shared<TileType>(shape1, DataType::FP32, std::nullopt, std::optional<TileView>(tv1));

  auto shape2 = std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  TileView tv2(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT64, sp)},
      std::make_shared<ConstInt>(0, DataType::INT64, sp));
  auto t2 = std::make_shared<TileType>(shape2, DataType::FP32, std::nullopt, std::optional<TileView>(tv2));

  ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

TEST_F(StructuralHashTest, TestTileTypeWithVsWithoutTileViewHash) {
  Span sp = Span::unknown();
  auto shape1 = std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  TileView tv(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)},
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INT64, sp)},
      std::make_shared<ConstInt>(0, DataType::INT64, sp));
  auto t1 = std::make_shared<TileType>(shape1, DataType::FP32, std::nullopt, std::optional<TileView>(tv));

  auto shape2 = std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)};
  auto t2 = std::make_shared<TileType>(shape2, DataType::FP32);

  // With TileView vs without should have different hashes
  ASSERT_NE(structural_hash(t1), structural_hash(t2));
}

// ============================================================================
// TypePtr overload Hash Tests
// ============================================================================

TEST_F(StructuralHashTest, TestTypeOverloadScalar) {
  TypePtr t1 = std::make_shared<ScalarType>(DataType::FP32);
  TypePtr t2 = std::make_shared<ScalarType>(DataType::FP32);
  ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

TEST_F(StructuralHashTest, TestTypeOverloadTensor) {
  Span sp = Span::unknown();
  TypePtr t1 = std::make_shared<TensorType>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32);
  TypePtr t2 = std::make_shared<TensorType>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32);
  ASSERT_EQ(structural_hash(t1), structural_hash(t2));
}

// ============================================================================
// IterArg Hash without Auto-mapping
// ============================================================================

TEST_F(StructuralHashTest, TestIterArgHashNoAutoMapping) {
  Span sp = Span::unknown();
  auto init1 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto ia1 = std::make_shared<IterArg>("acc",
      std::make_shared<ScalarType>(DataType::INT32), init1, sp);
  auto init2 = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto ia2 = std::make_shared<IterArg>("acc",
      std::make_shared<ScalarType>(DataType::INT32), init2, sp);
  // Without auto-mapping, different pointers should hash differently
  ASSERT_NE(structural_hash(ia1, false), structural_hash(ia2, false));
}

// ============================================================================
// Hash node caching test
// ============================================================================

TEST_F(StructuralHashTest, TestHashCachingWithSharedNode) {
  Span sp = Span::unknown();
  auto shared_val = std::make_shared<ConstInt>(42, DataType::INT32, sp);
  // Create a MakeTuple that references the same node twice
  auto tuple = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{shared_val, shared_val}, sp);
  auto h1 = structural_hash(tuple);
  auto h2 = structural_hash(tuple);
  ASSERT_EQ(h1, h2);
}

}  // namespace ir
}  // namespace pypto
