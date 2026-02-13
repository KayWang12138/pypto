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

#include <memory>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
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

}  // namespace ir
}  // namespace pypto
