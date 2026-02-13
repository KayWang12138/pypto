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

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/kind_traits.h"
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
  auto body_val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_val, Span::unknown());
  auto original = std::make_shared<Function>("test_func", std::vector<VarPtr>{}, std::vector<TypePtr>{},
                                             body, Span::unknown());

  auto bytes = serialization::Serialize(original);
  ASSERT_FALSE(bytes.empty());

  auto deserialized = serialization::Deserialize(bytes);
  ASSERT_NE(deserialized, nullptr);

  auto result = As<Function>(deserialized);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->name_, "test_func");
}

// ============================================================================
// Program Round-Trip Tests
// ============================================================================
TEST_F(SerializationTest, TestProgramRoundTrip) {
  auto body_val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_val, Span::unknown());
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

}  // namespace ir
}  // namespace pypto
