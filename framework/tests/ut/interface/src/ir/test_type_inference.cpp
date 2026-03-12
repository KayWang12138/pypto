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
 * \file test_type_inference.cpp
 * \brief Unit tests for type inference utilities
 */

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "ir/type_inference.h"

namespace pypto {
namespace ir {

class TypeInferenceTest : public testing::Test {};

// ============================================================================
// BroadcastShapes Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestBroadcastBothEmpty) {
    std::vector<ExprPtr> s1, s2;
    auto result = BroadcastShapes(s1, s2);
    ASSERT_TRUE(result.success);
    ASSERT_TRUE(result.shape.empty());
}

TEST_F(TypeInferenceTest, TestBroadcastFirstEmpty) {
    std::vector<ExprPtr> s1;
    std::vector<ExprPtr> s2 = {std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown())};
    auto result = BroadcastShapes(s1, s2);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.shape.size(), 1);
}

TEST_F(TypeInferenceTest, TestBroadcastSecondEmpty) {
    std::vector<ExprPtr> s1 = {std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown())};
    std::vector<ExprPtr> s2;
    auto result = BroadcastShapes(s1, s2);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.shape.size(), 1);
}

TEST_F(TypeInferenceTest, TestBroadcastSameShape) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d8 = std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown());
    std::vector<ExprPtr> s1 = {d4, d8};
    std::vector<ExprPtr> s2 = {std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown()),
        std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown())};
    auto result = BroadcastShapes(s1, s2);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.shape.size(), 2);
}

TEST_F(TypeInferenceTest, TestBroadcastDifferentRank) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d8 = std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown());
    std::vector<ExprPtr> s1 = {d4, d8};
    std::vector<ExprPtr> s2 = {std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown())};
    auto result = BroadcastShapes(s1, s2);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.shape.size(), 2);
}

TEST_F(TypeInferenceTest, TestBroadcastWithOne) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d1 = std::make_shared<ConstInt>(1, DataType::INT64, Span::Unknown());
    auto d8 = std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown());
    std::vector<ExprPtr> s1 = {d4, d1};
    std::vector<ExprPtr> s2 = {std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown())};
    auto result = BroadcastShapes(s1, s2);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.shape.size(), 2);
}

TEST_F(TypeInferenceTest, TestBroadcastIncompatible) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d5 = std::make_shared<ConstInt>(5, DataType::INT64, Span::Unknown());
    std::vector<ExprPtr> s1 = {d4};
    std::vector<ExprPtr> s2 = {d5};
    auto result = BroadcastShapes(s1, s2);
    ASSERT_FALSE(result.success);
    ASSERT_FALSE(result.errorMessage.empty());
}

// ============================================================================
// PromoteDataTypes Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestPromoteSameType) {
    auto result = PromoteDataTypes(DataType::INT32, DataType::INT32);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(*result, DataType::INT32);
}

TEST_F(TypeInferenceTest, TestPromoteFloatOverInt) {
    auto result = PromoteDataTypes(DataType::FP32, DataType::INT32);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(*result, DataType::FP32);
}

TEST_F(TypeInferenceTest, TestPromoteIntOverFloat) {
    auto result = PromoteDataTypes(DataType::INT32, DataType::FP32);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(*result, DataType::FP32);
}

TEST_F(TypeInferenceTest, TestPromoteLargerInt) {
    auto result = PromoteDataTypes(DataType::INT32, DataType::INT64);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(*result, DataType::INT64);
}

TEST_F(TypeInferenceTest, TestPromoteLargerFloat) {
    auto result = PromoteDataTypes(DataType::FP16, DataType::FP32);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(*result, DataType::FP32);
}

// ============================================================================
// CheckTypeCompatibility Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestCompatibleScalars) {
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto t2 = std::make_shared<ScalarType>(DataType::FP32);
    ASSERT_TRUE(CheckTypeCompatibility(t1, t2));
}

TEST_F(TypeInferenceTest, TestCompatibleTensors) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto t1 = std::make_shared<TensorType>(std::vector<ExprPtr>{d4}, DataType::FP32);
    auto t2 = std::make_shared<TensorType>(std::vector<ExprPtr>{d4}, DataType::FP32);
    ASSERT_TRUE(CheckTypeCompatibility(t1, t2));
}

TEST_F(TypeInferenceTest, TestIncompatibleScalarTensor) {
    auto t1 = std::make_shared<ScalarType>(DataType::INT32);
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto t2 = std::make_shared<TensorType>(std::vector<ExprPtr>{d4}, DataType::FP32);
    ASSERT_FALSE(CheckTypeCompatibility(t1, t2));
}

// ============================================================================
// ExtractDataType Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestExtractDataTypeScalar) {
    auto t = std::make_shared<ScalarType>(DataType::INT32);
    auto dt = ExtractDataType(t);
    ASSERT_TRUE(dt.has_value());
    ASSERT_EQ(*dt, DataType::INT32);
}

TEST_F(TypeInferenceTest, TestExtractDataTypeTensor) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto t = std::make_shared<TensorType>(std::vector<ExprPtr>{d4}, DataType::FP32);
    auto dt = ExtractDataType(t);
    ASSERT_TRUE(dt.has_value());
    ASSERT_EQ(*dt, DataType::FP32);
}

TEST_F(TypeInferenceTest, TestExtractDataTypeUnknown) {
    auto t = GetUnknownType();
    auto dt = ExtractDataType(t);
    ASSERT_FALSE(dt.has_value());
}

// ============================================================================
// ExtractShape Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestExtractShapeTensor) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d8 = std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown());
    auto t = std::make_shared<TensorType>(std::vector<ExprPtr>{d4, d8}, DataType::FP32);
    auto shape = ExtractShape(t);
    ASSERT_EQ(shape.size(), 2);
}

TEST_F(TypeInferenceTest, TestExtractShapeScalar) {
    auto t = std::make_shared<ScalarType>(DataType::INT32);
    auto shape = ExtractShape(t);
    ASSERT_TRUE(shape.empty());
}

// ============================================================================
// GetConstantDimension Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestGetConstantDimension) {
    auto dim = std::make_shared<ConstInt>(42, DataType::INT64, Span::Unknown());
    auto val = GetConstantDimension(dim);
    ASSERT_TRUE(val.has_value());
    ASSERT_EQ(*val, 42);
}

TEST_F(TypeInferenceTest, TestGetConstantDimensionNonConst) {
    auto intType = std::make_shared<ScalarType>(DataType::INT64);
    auto dim = std::make_shared<Var>("N", intType, Span::Unknown());
    auto val = GetConstantDimension(dim);
    ASSERT_FALSE(val.has_value());
}

// ============================================================================
// DimensionsEqual Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestDimensionsEqualSameObject) {
    auto dim = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    ASSERT_TRUE(DimensionsEqual(dim, dim));
}

TEST_F(TypeInferenceTest, TestDimensionsEqualSameValue) {
    auto d1 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d2 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    ASSERT_TRUE(DimensionsEqual(d1, d2));
}

TEST_F(TypeInferenceTest, TestDimensionsEqualDifferentValue) {
    auto d1 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d2 = std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown());
    ASSERT_FALSE(DimensionsEqual(d1, d2));
}

TEST_F(TypeInferenceTest, TestDimensionsEqualSymbolic) {
    auto intType = std::make_shared<ScalarType>(DataType::INT64);
    auto d1 = std::make_shared<Var>("N", intType, Span::Unknown());
    auto d2 = std::make_shared<Var>("M", intType, Span::Unknown());
    ASSERT_FALSE(DimensionsEqual(d1, d2));
}

// ============================================================================
// IsBroadcastable Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestIsBroadcastableEqual) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    ASSERT_TRUE(IsBroadcastable(d4, d4));
}

TEST_F(TypeInferenceTest, TestIsBroadcastableSourceOne) {
    auto d1 = std::make_shared<ConstInt>(1, DataType::INT64, Span::Unknown());
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    ASSERT_TRUE(IsBroadcastable(d1, d4));
}

TEST_F(TypeInferenceTest, TestIsBroadcastableTargetOne) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d1 = std::make_shared<ConstInt>(1, DataType::INT64, Span::Unknown());
    ASSERT_TRUE(IsBroadcastable(d4, d1));
}

TEST_F(TypeInferenceTest, TestIsBroadcastableIncompatible) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d5 = std::make_shared<ConstInt>(5, DataType::INT64, Span::Unknown());
    ASSERT_FALSE(IsBroadcastable(d4, d5));
}

// ============================================================================
// FormatShape Tests
// ============================================================================

TEST_F(TypeInferenceTest, TestFormatShapeEmpty) {
    std::vector<ExprPtr> shape;
    ASSERT_EQ(FormatShape(shape), "[]");
}

TEST_F(TypeInferenceTest, TestFormatShapeConstants) {
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto d8 = std::make_shared<ConstInt>(8, DataType::INT64, Span::Unknown());
    std::string result = FormatShape({d4, d8});
    ASSERT_EQ(result, "[4, 8]");
}

TEST_F(TypeInferenceTest, TestFormatShapeSymbolic) {
    auto intType = std::make_shared<ScalarType>(DataType::INT64);
    auto d4 = std::make_shared<ConstInt>(4, DataType::INT64, Span::Unknown());
    auto dN = std::make_shared<Var>("N", intType, Span::Unknown());
    std::string result = FormatShape({d4, dN});
    ASSERT_EQ(result, "[4, ?]");
}

} // namespace ir
} // namespace pypto
