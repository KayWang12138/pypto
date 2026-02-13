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

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
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

}  // namespace ir
}  // namespace pypto
