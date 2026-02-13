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
 * \file test_printer.cpp
 * \brief Unit tests for IR Python printer
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/printer.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class IRPrinterTest : public testing::Test {};

// ============================================================================
// Scalar Expression Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintConstInt) {
  auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  std::string result = PythonPrint(expr);
  ASSERT_NE(result.find("42"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintConstFloat) {
  auto expr = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
  std::string result = PythonPrint(expr);
  ASSERT_NE(result.find("3.14"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintConstBoolTrue) {
  auto expr = std::make_shared<ConstBool>(true, Span::unknown());
  std::string result = PythonPrint(expr);
  ASSERT_NE(result.find("True"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintConstBoolFalse) {
  auto expr = std::make_shared<ConstBool>(false, Span::unknown());
  std::string result = PythonPrint(expr);
  ASSERT_NE(result.find("False"), std::string::npos);
}

// ============================================================================
// Binary Expression Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintAdd) {
  auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto add = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

  std::string result = PythonPrint(add);
  ASSERT_NE(result.find("+"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintSub) {
  auto left = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
  auto sub = std::make_shared<Sub>(left, right, DataType::INT32, Span::unknown());

  std::string result = PythonPrint(sub);
  ASSERT_NE(result.find("-"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintMul) {
  auto left = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
  auto right = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
  auto mul = std::make_shared<Mul>(left, right, DataType::INT32, Span::unknown());

  std::string result = PythonPrint(mul);
  ASSERT_NE(result.find("*"), std::string::npos);
}

// ============================================================================
// Variable Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintVar) {
  auto var = std::make_shared<Var>("my_var", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  std::string result = PythonPrint(var);
  ASSERT_NE(result.find("my_var"), std::string::npos);
}

// ============================================================================
// Statement Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintEvalStmt) {
  auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
  auto stmt = std::make_shared<EvalStmt>(expr, Span::unknown());
  std::string result = PythonPrint(stmt);
  ASSERT_NE(result.find("42"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintAssignStmt) {
  auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
  auto val = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
  auto stmt = std::make_shared<AssignStmt>(var, val, Span::unknown());
  std::string result = PythonPrint(stmt);
  ASSERT_NE(result.find("x"), std::string::npos);
  ASSERT_NE(result.find("10"), std::string::npos);
}

// ============================================================================
// Function Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintFunction) {
  auto body_val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_val, Span::unknown());
  auto func = std::make_shared<Function>("my_func", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                         Span::unknown());
  std::string result = PythonPrint(func);
  ASSERT_NE(result.find("my_func"), std::string::npos);
  ASSERT_NE(result.find("def"), std::string::npos);
}

// ============================================================================
// Program Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintProgram) {
  auto body_val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_val, Span::unknown());
  auto func = std::make_shared<Function>("main", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body,
                                         Span::unknown());
  auto program = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "test_prog", Span::unknown());

  std::string result = PythonPrint(program);
  ASSERT_NE(result.find("main"), std::string::npos);
}

// ============================================================================
// Type Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintScalarType) {
  auto type = std::make_shared<ScalarType>(DataType::FP32);
  std::string result = PythonPrint(type);
  ASSERT_FALSE(result.empty());
}

TEST_F(IRPrinterTest, TestPrintTensorType) {
  std::vector<ExprPtr> shape = {
      std::make_shared<ConstInt>(10, DataType::INT64, Span::unknown()),
      std::make_shared<ConstInt>(20, DataType::INT64, Span::unknown())};
  auto type = std::make_shared<TensorType>(shape, DataType::FP32);
  std::string result = PythonPrint(type);
  ASSERT_FALSE(result.empty());
}

}  // namespace ir
}  // namespace pypto
