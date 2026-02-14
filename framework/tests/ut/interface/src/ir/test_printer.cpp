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
#include <optional>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
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

// ============================================================================
// Additional Binary Expression Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintFloorDiv) {
  auto l = std::make_shared<ConstInt>(
      10, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      3, DataType::INT32, Span::unknown());
  auto e = std::make_shared<FloorDiv>(
      l, r, DataType::INT32, Span::unknown());
  std::string result = PythonPrint(e);
  ASSERT_NE(result.find("//"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintFloorMod) {
  auto l = std::make_shared<ConstInt>(
      10, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      3, DataType::INT32, Span::unknown());
  auto e = std::make_shared<FloorMod>(
      l, r, DataType::INT32, Span::unknown());
  std::string result = PythonPrint(e);
  ASSERT_NE(result.find("%"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintFloatDiv) {
  auto l = std::make_shared<ConstFloat>(
      1.0, DataType::FP32, Span::unknown());
  auto r = std::make_shared<ConstFloat>(
      2.0, DataType::FP32, Span::unknown());
  auto e = std::make_shared<FloatDiv>(
      l, r, DataType::FP32, Span::unknown());
  std::string result = PythonPrint(e);
  ASSERT_FALSE(result.empty());
}

TEST_F(IRPrinterTest, TestPrintPow) {
  auto l = std::make_shared<ConstInt>(
      2, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      3, DataType::INT32, Span::unknown());
  auto e = std::make_shared<Pow>(
      l, r, DataType::INT32, Span::unknown());
  std::string result = PythonPrint(e);
  ASSERT_FALSE(result.empty());
}

TEST_F(IRPrinterTest, TestPrintMinMax) {
  auto l = std::make_shared<ConstInt>(
      1, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      2, DataType::INT32, Span::unknown());
  auto mn = std::make_shared<Min>(
      l, r, DataType::INT32, Span::unknown());
  auto mx = std::make_shared<Max>(
      l, r, DataType::INT32, Span::unknown());
  ASSERT_FALSE(PythonPrint(mn).empty());
  ASSERT_FALSE(PythonPrint(mx).empty());
}

TEST_F(IRPrinterTest, TestPrintComparisonOps) {
  auto l = std::make_shared<ConstInt>(
      1, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      2, DataType::INT32, Span::unknown());
  ASSERT_FALSE(PythonPrint(std::make_shared<Eq>(
      l, r, DataType::BOOL, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<Ne>(
      l, r, DataType::BOOL, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<Lt>(
      l, r, DataType::BOOL, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<Le>(
      l, r, DataType::BOOL, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<Gt>(
      l, r, DataType::BOOL, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<Ge>(
      l, r, DataType::BOOL, Span::unknown())).empty());
}

TEST_F(IRPrinterTest, TestPrintLogicalOps) {
  auto l = std::make_shared<ConstBool>(
      true, Span::unknown());
  auto r = std::make_shared<ConstBool>(
      false, Span::unknown());
  ASSERT_FALSE(PythonPrint(std::make_shared<And>(
      l, r, DataType::BOOL, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<Or>(
      l, r, DataType::BOOL, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<Xor>(
      l, r, DataType::BOOL, Span::unknown())).empty());
}

TEST_F(IRPrinterTest, TestPrintBitwiseOps) {
  auto l = std::make_shared<ConstInt>(
      0xFF, DataType::INT32, Span::unknown());
  auto r = std::make_shared<ConstInt>(
      0x0F, DataType::INT32, Span::unknown());
  ASSERT_FALSE(PythonPrint(std::make_shared<BitAnd>(
      l, r, DataType::INT32, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<BitOr>(
      l, r, DataType::INT32, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(std::make_shared<BitXor>(
      l, r, DataType::INT32, Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(
      std::make_shared<BitShiftLeft>(
          l, r, DataType::INT32,
          Span::unknown())).empty());
  ASSERT_FALSE(PythonPrint(
      std::make_shared<BitShiftRight>(
          l, r, DataType::INT32,
          Span::unknown())).empty());
}

// ============================================================================
// Unary Expression Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintNeg) {
  auto v = std::make_shared<ConstInt>(
      5, DataType::INT32, Span::unknown());
  auto e = std::make_shared<Neg>(
      v, DataType::INT32, Span::unknown());
  std::string result = PythonPrint(e);
  ASSERT_NE(result.find("-"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintAbs) {
  auto v = std::make_shared<ConstInt>(
      5, DataType::INT32, Span::unknown());
  auto e = std::make_shared<Abs>(
      v, DataType::INT32, Span::unknown());
  ASSERT_FALSE(PythonPrint(e).empty());
}

TEST_F(IRPrinterTest, TestPrintNot) {
  auto v = std::make_shared<ConstBool>(
      true, Span::unknown());
  auto e = std::make_shared<Not>(
      v, DataType::BOOL, Span::unknown());
  ASSERT_FALSE(PythonPrint(e).empty());
}

TEST_F(IRPrinterTest, TestPrintBitNot) {
  auto v = std::make_shared<ConstInt>(
      0xFF, DataType::INT32, Span::unknown());
  auto e = std::make_shared<BitNot>(
      v, DataType::INT32, Span::unknown());
  ASSERT_FALSE(PythonPrint(e).empty());
}

TEST_F(IRPrinterTest, TestPrintCast) {
  auto v = std::make_shared<ConstInt>(
      5, DataType::INT32, Span::unknown());
  auto e = std::make_shared<Cast>(
      v, DataType::FP32, Span::unknown());
  ASSERT_FALSE(PythonPrint(e).empty());
}

// ============================================================================
// Additional Node Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintMakeTuple) {
  auto e1 = std::make_shared<ConstInt>(
      1, DataType::INT32, Span::unknown());
  auto e2 = std::make_shared<ConstInt>(
      2, DataType::INT32, Span::unknown());
  auto tup = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e1, e2}, Span::unknown());
  ASSERT_FALSE(PythonPrint(tup).empty());
}

TEST_F(IRPrinterTest, TestPrintTupleGetItem) {
  auto e1 = std::make_shared<ConstInt>(
      1, DataType::INT32, Span::unknown());
  auto tup = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{e1}, Span::unknown());
  auto get = std::make_shared<TupleGetItemExpr>(
      tup, 0,
      Span::unknown());
  ASSERT_FALSE(PythonPrint(get).empty());
}

TEST_F(IRPrinterTest, TestPrintCall) {
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
  ASSERT_FALSE(PythonPrint(call).empty());
}

TEST_F(IRPrinterTest, TestPrintReturnStmt) {
  Span sp = Span::unknown();
  auto val = std::make_shared<ConstInt>(
      1, DataType::INT32, sp);
  auto ret = std::make_shared<ReturnStmt>(
      std::vector<ExprPtr>{val}, sp);
  ASSERT_FALSE(PythonPrint(ret).empty());
}

TEST_F(IRPrinterTest, TestPrintForStmt) {
  Span sp = Span::unknown();
  auto loop_var = std::make_shared<Var>(
      "i",
      std::make_shared<ScalarType>(DataType::INT32),
      sp);
  auto start = std::make_shared<ConstInt>(
      0, DataType::INT32, sp);
  auto stop = std::make_shared<ConstInt>(
      10, DataType::INT32, sp);
  auto step = std::make_shared<ConstInt>(
      1, DataType::INT32, sp);
  auto body = std::make_shared<EvalStmt>(start, sp);
  auto for_stmt = std::make_shared<ForStmt>(
      loop_var, start, stop, step,
      std::vector<IterArgPtr>{}, body,
      std::vector<VarPtr>{}, sp);
  std::string result = PythonPrint(for_stmt);
  ASSERT_NE(result.find("for"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintIfStmt) {
  Span sp = Span::unknown();
  auto cond = std::make_shared<ConstBool>(true, sp);
  auto then_body = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(
          1, DataType::INT32, sp), sp);
  auto else_body = std::make_shared<EvalStmt>(
      std::make_shared<ConstInt>(
          2, DataType::INT32, sp), sp);
  auto if_stmt = std::make_shared<IfStmt>(
      cond, then_body,
      std::optional<StmtPtr>(else_body),
      std::vector<VarPtr>{}, sp);
  std::string result = PythonPrint(if_stmt);
  ASSERT_NE(result.find("if"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintOpStmts) {
  Span sp = Span::unknown();
  auto expr = std::make_shared<ConstInt>(
      1, DataType::INT32, sp);
  auto eval = std::make_shared<EvalStmt>(expr, sp);
  auto ops = std::make_shared<OpStmts>(
      std::vector<StmtPtr>{eval}, sp);
  ASSERT_FALSE(PythonPrint(ops).empty());
}

TEST_F(IRPrinterTest, TestPrintYieldStmt) {
  Span sp = Span::unknown();
  auto val = std::make_shared<ConstInt>(
      1, DataType::INT32, sp);
  auto yield = std::make_shared<YieldStmt>(
      std::vector<ExprPtr>{val}, sp);
  ASSERT_FALSE(PythonPrint(yield).empty());
}

TEST_F(IRPrinterTest, TestPrintTileType) {
  std::vector<ExprPtr> shape = {
      std::make_shared<ConstInt>(
          4, DataType::INT64, Span::unknown()),
      std::make_shared<ConstInt>(
          8, DataType::INT64, Span::unknown())};
  auto type = std::make_shared<TileType>(
      shape, DataType::FP32);
  ASSERT_FALSE(PythonPrint(type).empty());
}

TEST_F(IRPrinterTest, TestPrintMemRef) {
  auto addr = std::make_shared<ConstInt>(
      0, DataType::INT64, Span::unknown());
  auto mr = std::make_shared<MemRef>(
      MemorySpace::UB, addr, 1024, 0);
  ASSERT_FALSE(PythonPrint(mr).empty());
}

TEST_F(IRPrinterTest, TestPrintDataTypes) {
  Span sp = Span::unknown();
  auto fp16 = std::make_shared<ScalarType>(
      DataType::FP16);
  auto int64 = std::make_shared<ScalarType>(
      DataType::INT64);
  auto int32 = std::make_shared<ScalarType>(
      DataType::INT32);
  auto bool_t = std::make_shared<ScalarType>(
      DataType::BOOL);
  ASSERT_FALSE(PythonPrint(fp16).empty());
  ASSERT_FALSE(PythonPrint(int64).empty());
  ASSERT_FALSE(PythonPrint(int32).empty());
  ASSERT_FALSE(PythonPrint(bool_t).empty());
}

TEST_F(IRPrinterTest, TestPrintFuncWithParams) {
  Span sp = Span::unknown();
  auto param = std::make_shared<Var>(
      "x",
      std::make_shared<TensorType>(
          std::vector<ExprPtr>{
              std::make_shared<ConstInt>(
                  4, DataType::INT64, sp)},
          DataType::FP32),
      sp);
  auto body = std::make_shared<ReturnStmt>(
      std::vector<ExprPtr>{param}, sp);
  auto ret_type = std::make_shared<TensorType>(
      std::vector<ExprPtr>{
          std::make_shared<ConstInt>(
              4, DataType::INT64, sp)},
      DataType::FP32);
  auto func = std::make_shared<Function>(
      "identity",
      std::vector<VarPtr>{param},
      std::vector<TypePtr>{ret_type},
      body, sp);
  std::string result = PythonPrint(func);
  ASSERT_NE(result.find("identity"), std::string::npos);
  ASSERT_NE(result.find("x"), std::string::npos);
}

}  // namespace ir
}  // namespace pypto
