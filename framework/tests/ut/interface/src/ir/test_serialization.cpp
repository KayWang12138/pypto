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
 * \file test_serialization.cpp
 * \brief Unit tests for IR serialization and deserialization round-trip
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
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/serialization/deserializer.h"
#include "ir/serialization/serializer.h"
#include "ir/stmt.h"
#include "ir/transform/structural_comparison.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class SerializationTest : public testing::Test {};

// ============================================================================
// Scalar Expression Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestConstIntRoundTrip) {
  auto original = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());

  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);

  auto result = As<ConstInt>(deserialized);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->value_, 42);
}

TEST_F(SerializationTest, TestConstFloatRoundTrip) {
  auto original = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());

  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);

  auto result = As<ConstFloat>(deserialized);
  ASSERT_NE(result, nullptr);
}

TEST_F(SerializationTest, TestConstBoolRoundTrip) {
  auto original = std::make_shared<ConstBool>(true, Span::unknown());
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());

  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);

  auto result = As<ConstBool>(deserialized);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->value_, true);
}

// ============================================================================
// Binary Expression Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestAddRoundTrip) {
  auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto original = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());

  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);

  auto result = As<Add>(deserialized);
  ASSERT_NE(result, nullptr);
}

// ============================================================================
// Statement Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestEvalStmtRoundTrip) {
  auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto original = std::make_shared<EvalStmt>(expr, Span::unknown());

  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());

  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);

  auto result = As<EvalStmt>(deserialized);
  ASSERT_NE(result, nullptr);
}

// ============================================================================
// Function Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestFunctionRoundTrip) {
  auto bodyVal = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(bodyVal, Span::unknown());
  auto original = std::make_shared<Function>("testFunc", std::vector<VarPtr>{}, std::vector<TypePtr>{},
                                             body, Span::unknown());

  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());

  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);

  auto result = As<Function>(deserialized);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->name_, "testFunc");
}

// ============================================================================
// Program Round-Trip Tests
// ============================================================================
TEST_F(SerializationTest, TestProgramRoundTrip) {
  auto bodyVal = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(bodyVal, Span::unknown());
  auto func = std::make_shared<Function>("main", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                         Span::unknown());
  auto original = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "test_prog", Span::unknown());

  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());

  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);

  auto result = As<Program>(deserialized);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->name_, "test_prog");
}

// ============================================================================
// Structural Equality After Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestStructuralEqualAfterRoundTrip) {
  auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto original = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

  auto bytes = serialization::Serialize(original);
  auto deserialized = serialization::Deserialize(bytes);
  auto result = std::dynamic_pointer_cast<const Expr>(deserialized);

  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(structural_equal(original, result));
}

// ============================================================================
// Variable and Unary Expression Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestVarRoundTrip) {
  Span sp = Span::unknown();
  auto original = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  auto result = As<Var>(deserialized);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->name_, "x");
}

TEST_F(SerializationTest, TestNegRoundTrip) {
  Span sp = Span::unknown();
  auto v = std::make_shared<ConstInt>(5, DataType::INT32, sp);
  auto original = std::make_shared<Neg>(v, DataType::INT32, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
  ASSERT_TRUE(structural_equal(original, deserialized));
}

TEST_F(SerializationTest, TestCastRoundTrip) {
  Span sp = Span::unknown();
  auto v = std::make_shared<ConstInt>(5, DataType::INT32, sp);
  auto original = std::make_shared<Cast>(v, DataType::FP32, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

// ============================================================================
// Tuple Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestMakeTupleRoundTrip) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto e2 = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto original = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e1, e2}, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
  ASSERT_TRUE(structural_equal(original, deserialized));
}

TEST_F(SerializationTest, TestTupleGetItemRoundTrip) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto tup = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e1}, sp);
  auto original = std::make_shared<TupleGetItemExpr>(tup, 0, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

// ============================================================================
// Additional Statement Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestAssignStmtRoundTrip) {
  Span sp = Span::unknown();
  auto var = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto val = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto original = std::make_shared<AssignStmt>(var, val, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
  ASSERT_TRUE(structural_equal(original, deserialized));
}

TEST_F(SerializationTest, TestSeqStmtsRoundTrip) {
  Span sp = Span::unknown();
  auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto s1 = std::make_shared<EvalStmt>(e1, sp);
  auto e2 = std::make_shared<ConstInt>(2, DataType::INT32, sp);
  auto s2 = std::make_shared<EvalStmt>(e2, sp);
  auto original = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{s1, s2}, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
  ASSERT_TRUE(structural_equal(original, deserialized));
}

TEST_F(SerializationTest, TestReturnStmtRoundTrip) {
  Span sp = Span::unknown();
  auto v = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto original = std::make_shared<ReturnStmt>(
      std::vector<ExprPtr>{v}, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

TEST_F(SerializationTest, TestYieldStmtRoundTrip) {
  Span sp = Span::unknown();
  auto v = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto original = std::make_shared<YieldStmt>(
      std::vector<ExprPtr>{v}, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

TEST_F(SerializationTest, TestForStmtRoundTrip) {
  Span sp = Span::unknown();
  auto lv = std::make_shared<Var>(
      "i", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto start = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto stop = std::make_shared<ConstInt>(10, DataType::INT32, sp);
  auto step = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto body = std::make_shared<EvalStmt>(start, sp);
  auto original = std::make_shared<ForStmt>(
      lv, start, stop, step,
      std::vector<IterArgPtr>{}, body,
      std::vector<VarPtr>{}, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

TEST_F(SerializationTest, TestIfStmtRoundTrip) {
  Span sp = Span::unknown();
  auto cond = std::make_shared<ConstBool>(true, sp);
  auto thenBody = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
  auto elseBody = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
  auto original = std::make_shared<IfStmt>(
      cond, thenBody, std::optional<StmtPtr>(elseBody),
      std::vector<VarPtr>{}, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

TEST_F(SerializationTest, TestOpStmtsRoundTrip) {
  Span sp = Span::unknown();
  auto e = std::make_shared<ConstInt>(1, DataType::INT32, sp);
  auto eval = std::make_shared<EvalStmt>(e, sp);
  auto original = std::make_shared<OpStmts>(
      std::vector<StmtPtr>{eval}, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

// ============================================================================
// Call with Kwargs Round-Trip Tests
// ============================================================================

TEST_F(SerializationTest, TestCallRoundTrip) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto a = std::make_shared<Var>(
      "a", std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto b = std::make_shared<Var>(
      "b", std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto original = reg.Create("tensor.add", {a, b}, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

TEST_F(SerializationTest, TestCallWithKwargsRoundTrip) {
  Span sp = Span::unknown();
  auto& reg = OpRegistry::GetInstance();
  auto tile = std::make_shared<Var>(
      "t", std::make_shared<TileType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp),
              std::make_shared<ConstInt>(8, DataType::INT64, sp)},
          DataType::FP32), sp);
  std::vector<std::pair<std::string, std::any>> kwargs = {
      {"axis", std::any(1)}, {"keepdim", std::any(true)}};
  auto original = reg.Create("block.sum", {tile}, kwargs, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

// ============================================================================
// Type Serialization Tests
// ============================================================================

TEST_F(SerializationTest, TestTensorTypeVarRoundTrip) {
  Span sp = Span::unknown();
  auto var = std::make_shared<Var>(
      "x", std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp),
              std::make_shared<ConstInt>(8, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto bytes = serialization::Serialize(var);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

TEST_F(SerializationTest, TestTileTypeVarRoundTrip) {
  Span sp = Span::unknown();
  auto var = std::make_shared<Var>(
      "t", std::make_shared<TileType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(4, DataType::INT64, sp)},
          DataType::FP32), sp);
  auto bytes = serialization::Serialize(var);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

TEST_F(SerializationTest, TestFuncWithReturnTypes) {
  Span sp = Span::unknown();
  auto param = std::make_shared<Var>(
      "x", std::make_shared<ScalarType>(DataType::INT32), sp);
  auto body = std::make_shared<ReturnStmt>(
      std::vector<ExprPtr>{param}, sp);
  auto retType = std::make_shared<ScalarType>(DataType::INT32);
  auto original = std::make_shared<Function>(
      "f", std::vector<VarPtr>{param},
      std::vector<TypePtr>{retType}, body, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
  auto result = As<Function>(deserialized);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->name_, "f");
}

// ============================================================================
// Shared Pointer Deduplication Test
// ============================================================================

TEST_F(SerializationTest, TestSharedPointerDedup) {
  Span sp = Span::unknown();
  auto sharedVal = std::make_shared<ConstInt>(42, DataType::INT32, sp);
  auto add = std::make_shared<Add>(
      sharedVal, sharedVal, DataType::INT32, sp);
  auto bytes = serialization::Serialize(add);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

// ============================================================================
// IterArg Round-Trip Test
// ============================================================================

TEST_F(SerializationTest, TestIterArgRoundTrip) {
  Span sp = Span::unknown();
  auto init = std::make_shared<ConstInt>(0, DataType::INT32, sp);
  auto original = std::make_shared<IterArg>(
      "acc", std::make_shared<ScalarType>(DataType::INT32),
      init, sp);
  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());
  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);
}

}  // namespace ir
}  // namespace pypto
