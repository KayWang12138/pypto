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
 * \file test_scalar_expr_basic.cpp
 * \brief Unit tests for basic scalar expressions (constants only)
 */

#include "gtest/gtest.h"

#include <memory>

#include "core/dtype.h"
#include "ir/scalar_expr.h"

namespace pypto {
namespace ir {

// ============================================================================
// ConstInt Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestConstIntInt32) {
  // Test ConstInt with INT32
  auto const_int = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  ASSERT_NE(const_int, nullptr);
  ASSERT_EQ(const_int->TypeName(), "ConstInt");
  ASSERT_EQ(const_int->value_, 42);
  ASSERT_EQ(const_int->dtype(), DataType::INT32);
}

TEST(ScalarExprBasicTest, TestConstIntZero) {
  // Test ConstInt with zero value
  auto const_int = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  ASSERT_NE(const_int, nullptr);
  ASSERT_EQ(const_int->value_, 0);
}

TEST(ScalarExprBasicTest, TestConstIntNegative) {
  // Test ConstInt with negative value
  auto const_int = std::make_shared<ConstInt>(-100, DataType::INT32, Span::unknown());
  ASSERT_NE(const_int, nullptr);
  ASSERT_EQ(const_int->value_, -100);
}

TEST(ScalarExprBasicTest, TestConstIntInt64) {
  // Test ConstInt with INT64
  auto const_int = std::make_shared<ConstInt>(1000000, DataType::INT64, Span::unknown());
  ASSERT_NE(const_int, nullptr);
  ASSERT_EQ(const_int->dtype(), DataType::INT64);
}

TEST(ScalarExprBasicTest, TestConstIntInt8) {
  // Test ConstInt with INT8
  auto const_int = std::make_shared<ConstInt>(127, DataType::INT8, Span::unknown());
  ASSERT_NE(const_int, nullptr);
  ASSERT_EQ(const_int->dtype(), DataType::INT8);
}

TEST(ScalarExprBasicTest, TestConstIntUInt32) {
  // Test ConstInt with UINT32
  auto const_int = std::make_shared<ConstInt>(255, DataType::UINT32, Span::unknown());
  ASSERT_NE(const_int, nullptr);
  ASSERT_EQ(const_int->dtype(), DataType::UINT32);
}

TEST(ScalarExprBasicTest, TestConstIntWithSpan) {
  // Test ConstInt with valid span
  Span span("test.py", 10, 5);
  auto const_int = std::make_shared<ConstInt>(42, DataType::INT32, span);
  ASSERT_NE(const_int, nullptr);
  ASSERT_EQ(const_int->span_.filename_, "test.py");
  ASSERT_EQ(const_int->span_.begin_line_, 10);
}

// ============================================================================
// ConstFloat Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestConstFloatFP32) {
  // Test ConstFloat with FP32
  auto const_float = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  ASSERT_NE(const_float, nullptr);
  ASSERT_EQ(const_float->TypeName(), "ConstFloat");
  ASSERT_DOUBLE_EQ(const_float->value_, 3.14);
  ASSERT_EQ(const_float->dtype(), DataType::FP32);
}

TEST(ScalarExprBasicTest, TestConstFloatZero) {
  // Test ConstFloat with zero value
  auto const_float = std::make_shared<ConstFloat>(0.0, DataType::FP32, Span::unknown());
  ASSERT_NE(const_float, nullptr);
  ASSERT_DOUBLE_EQ(const_float->value_, 0.0);
}

TEST(ScalarExprBasicTest, TestConstFloatNegative) {
  // Test ConstFloat with negative value
  auto const_float = std::make_shared<ConstFloat>(-2.5, DataType::FP32, Span::unknown());
  ASSERT_NE(const_float, nullptr);
  ASSERT_DOUBLE_EQ(const_float->value_, -2.5);
}

TEST(ScalarExprBasicTest, TestConstFloatFP16) {
  // Test ConstFloat with FP16
  auto const_float = std::make_shared<ConstFloat>(1.5, DataType::FP16, Span::unknown());
  ASSERT_NE(const_float, nullptr);
  ASSERT_EQ(const_float->dtype(), DataType::FP16);
}

TEST(ScalarExprBasicTest, TestConstFloatBF16) {
  // Test ConstFloat with BF16
  auto const_float = std::make_shared<ConstFloat>(2.5, DataType::BF16, Span::unknown());
  ASSERT_NE(const_float, nullptr);
  ASSERT_EQ(const_float->dtype(), DataType::BF16);
}

TEST(ScalarExprBasicTest, TestConstFloatLargeValue) {
  // Test ConstFloat with large value
  auto const_float = std::make_shared<ConstFloat>(1e10, DataType::FP32, Span::unknown());
  ASSERT_NE(const_float, nullptr);
  ASSERT_DOUBLE_EQ(const_float->value_, 1e10);
}

TEST(ScalarExprBasicTest, TestConstFloatSmallValue) {
  // Test ConstFloat with small value
  auto const_float = std::make_shared<ConstFloat>(1e-10, DataType::FP32, Span::unknown());
  ASSERT_NE(const_float, nullptr);
  ASSERT_DOUBLE_EQ(const_float->value_, 1e-10);
}

// ============================================================================
// ConstBool Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestConstBoolTrue) {
  // Test ConstBool with true value
  auto const_bool = std::make_shared<ConstBool>(true, Span::unknown());
  ASSERT_NE(const_bool, nullptr);
  ASSERT_EQ(const_bool->TypeName(), "ConstBool");
  ASSERT_TRUE(const_bool->value_);
  ASSERT_EQ(const_bool->dtype(), DataType::BOOL);
}

TEST(ScalarExprBasicTest, TestConstBoolFalse) {
  // Test ConstBool with false value
  auto const_bool = std::make_shared<ConstBool>(false, Span::unknown());
  ASSERT_NE(const_bool, nullptr);
  ASSERT_FALSE(const_bool->value_);
  ASSERT_EQ(const_bool->dtype(), DataType::BOOL);
}

TEST(ScalarExprBasicTest, TestConstBoolWithSpan) {
  // Test ConstBool with valid span
  Span span("test.py", 20, 10);
  auto const_bool = std::make_shared<ConstBool>(true, span);
  ASSERT_NE(const_bool, nullptr);
  ASSERT_EQ(const_bool->span_.filename_, "test.py");
  ASSERT_EQ(const_bool->span_.begin_line_, 20);
}

// ============================================================================
// Type Checking Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestConstIntGetType) {
  // Test ConstInt GetType method
  auto const_int = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto type = const_int->GetType();
  ASSERT_NE(type, nullptr);
  ASSERT_EQ(type->TypeName(), "ScalarType");

  auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(type);
  ASSERT_NE(scalar_type, nullptr);
  ASSERT_EQ(scalar_type->dtype_, DataType::INT32);
}

TEST(ScalarExprBasicTest, TestConstFloatGetType) {
  // Test ConstFloat GetType method
  auto const_float = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  auto type = const_float->GetType();
  ASSERT_NE(type, nullptr);
  ASSERT_EQ(type->TypeName(), "ScalarType");

  auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(type);
  ASSERT_NE(scalar_type, nullptr);
  ASSERT_EQ(scalar_type->dtype_, DataType::FP32);
}

TEST(ScalarExprBasicTest, TestConstBoolGetType) {
  // Test ConstBool GetType method
  auto const_bool = std::make_shared<ConstBool>(true, Span::unknown());
  auto type = const_bool->GetType();
  ASSERT_NE(type, nullptr);
  ASSERT_EQ(type->TypeName(), "ScalarType");

  auto scalar_type = std::dynamic_pointer_cast<const ScalarType>(type);
  ASSERT_NE(scalar_type, nullptr);
  ASSERT_EQ(scalar_type->dtype_, DataType::BOOL);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST(ScalarExprBasicTest, TestGetScalarDtype) {
  // Test GetScalarDtype helper function
  auto const_int = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  DataType dtype = GetScalarDtype(const_int);
  ASSERT_EQ(dtype, DataType::INT32);
}

TEST(ScalarExprBasicTest, TestIsBoolDtype) {
  // Test IsBoolDtype helper function
  ASSERT_TRUE(IsBoolDtype(DataType::BOOL));
  ASSERT_FALSE(IsBoolDtype(DataType::INT32));
  ASSERT_FALSE(IsBoolDtype(DataType::FP32));
}

TEST(ScalarExprBasicTest, TestGetNumericCategoryInt) {
  // Test GetNumericCategory for integer types
  auto category = GetNumericCategory(DataType::INT32, "test");
  ASSERT_EQ(category, ScalarCategory::kInt);
}

TEST(ScalarExprBasicTest, TestGetNumericCategoryFloat) {
  // Test GetNumericCategory for float types
  auto category = GetNumericCategory(DataType::FP32, "test");
  ASSERT_EQ(category, ScalarCategory::kFloat);
}

}  // namespace ir
}  // namespace pypto
