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
    auto bodyVal = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
    auto body = std::make_shared<EvalStmt>(bodyVal, Span::unknown());
    auto func =
        std::make_shared<Function>("my_func", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Span::unknown());
    std::string result = PythonPrint(func);
    ASSERT_NE(result.find("my_func"), std::string::npos);
    ASSERT_NE(result.find("def"), std::string::npos);
}

// ============================================================================
// Program Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintProgram) {
    auto bodyVal = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
    auto body = std::make_shared<EvalStmt>(bodyVal, Span::unknown());
    auto func =
        std::make_shared<Function>("main", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, Span::unknown());
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
    std::vector<ExprPtr> shape = {std::make_shared<ConstInt>(10, DataType::INT64, Span::unknown()),
        std::make_shared<ConstInt>(20, DataType::INT64, Span::unknown())};
    auto type = std::make_shared<TensorType>(shape, DataType::FP32);
    std::string result = PythonPrint(type);
    ASSERT_FALSE(result.empty());
}

// ============================================================================
// Additional Binary Expression Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintFloorDiv) {
    auto l = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
    auto e = std::make_shared<FloorDiv>(l, r, DataType::INT32, Span::unknown());
    std::string result = PythonPrint(e);
    ASSERT_NE(result.find("//"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintFloorMod) {
    auto l = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
    auto e = std::make_shared<FloorMod>(l, r, DataType::INT32, Span::unknown());
    std::string result = PythonPrint(e);
    ASSERT_NE(result.find("%"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintFloatDiv) {
    auto l = std::make_shared<ConstFloat>(1.0, DataType::FP32, Span::unknown());
    auto r = std::make_shared<ConstFloat>(2.0, DataType::FP32, Span::unknown());
    auto e = std::make_shared<FloatDiv>(l, r, DataType::FP32, Span::unknown());
    std::string result = PythonPrint(e);
    ASSERT_FALSE(result.empty());
}

TEST_F(IRPrinterTest, TestPrintPow) {
    auto l = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
    auto e = std::make_shared<Pow>(l, r, DataType::INT32, Span::unknown());
    std::string result = PythonPrint(e);
    ASSERT_FALSE(result.empty());
}

TEST_F(IRPrinterTest, TestPrintMinMax) {
    auto l = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto mn = std::make_shared<Min>(l, r, DataType::INT32, Span::unknown());
    auto mx = std::make_shared<Max>(l, r, DataType::INT32, Span::unknown());
    ASSERT_FALSE(PythonPrint(mn).empty());
    ASSERT_FALSE(PythonPrint(mx).empty());
}

TEST_F(IRPrinterTest, TestPrintComparisonOps) {
    auto l = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    ASSERT_FALSE(PythonPrint(std::make_shared<Eq>(l, r, DataType::BOOL, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<Ne>(l, r, DataType::BOOL, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<Lt>(l, r, DataType::BOOL, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<Le>(l, r, DataType::BOOL, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<Gt>(l, r, DataType::BOOL, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<Ge>(l, r, DataType::BOOL, Span::unknown())).empty());
}

TEST_F(IRPrinterTest, TestPrintLogicalOps) {
    auto l = std::make_shared<ConstBool>(true, Span::unknown());
    auto r = std::make_shared<ConstBool>(false, Span::unknown());
    ASSERT_FALSE(PythonPrint(std::make_shared<And>(l, r, DataType::BOOL, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<Or>(l, r, DataType::BOOL, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<Xor>(l, r, DataType::BOOL, Span::unknown())).empty());
}

TEST_F(IRPrinterTest, TestPrintBitwiseOps) {
    auto l = std::make_shared<ConstInt>(0xFF, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(0x0F, DataType::INT32, Span::unknown());
    ASSERT_FALSE(PythonPrint(std::make_shared<BitAnd>(l, r, DataType::INT32, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<BitOr>(l, r, DataType::INT32, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<BitXor>(l, r, DataType::INT32, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<BitShiftLeft>(l, r, DataType::INT32, Span::unknown())).empty());
    ASSERT_FALSE(PythonPrint(std::make_shared<BitShiftRight>(l, r, DataType::INT32, Span::unknown())).empty());
}

// ============================================================================
// Unary Expression Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintNeg) {
    auto v = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
    auto e = std::make_shared<Neg>(v, DataType::INT32, Span::unknown());
    std::string result = PythonPrint(e);
    ASSERT_NE(result.find("-"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintAbs) {
    auto v = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
    auto e = std::make_shared<Abs>(v, DataType::INT32, Span::unknown());
    ASSERT_FALSE(PythonPrint(e).empty());
}

TEST_F(IRPrinterTest, TestPrintNot) {
    auto v = std::make_shared<ConstBool>(true, Span::unknown());
    auto e = std::make_shared<Not>(v, DataType::BOOL, Span::unknown());
    ASSERT_FALSE(PythonPrint(e).empty());
}

TEST_F(IRPrinterTest, TestPrintBitNot) {
    auto v = std::make_shared<ConstInt>(0xFF, DataType::INT32, Span::unknown());
    auto e = std::make_shared<BitNot>(v, DataType::INT32, Span::unknown());
    ASSERT_FALSE(PythonPrint(e).empty());
}

TEST_F(IRPrinterTest, TestPrintCast) {
    auto v = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
    auto e = std::make_shared<Cast>(v, DataType::FP32, Span::unknown());
    ASSERT_FALSE(PythonPrint(e).empty());
}

// ============================================================================
// Additional Node Printing Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintMakeTuple) {
    auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto e2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1, e2}, Span::unknown());
    ASSERT_FALSE(PythonPrint(tup).empty());
}

TEST_F(IRPrinterTest, TestPrintTupleGetItem) {
    auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1}, Span::unknown());
    auto get = std::make_shared<TupleGetItemExpr>(tup, 0, Span::unknown());
    ASSERT_FALSE(PythonPrint(get).empty());
}

TEST_F(IRPrinterTest, TestPrintCall) {
    Span sp = Span::unknown();
    auto &reg = OpRegistry::GetInstance();
    auto a = std::make_shared<Var>("a",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto b = std::make_shared<Var>("b",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto call = reg.Create("tensor.add", {a, b}, sp);
    ASSERT_FALSE(PythonPrint(call).empty());
}

TEST_F(IRPrinterTest, TestPrintReturnStmt) {
    Span sp = Span::unknown();
    auto val = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{val}, sp);
    ASSERT_FALSE(PythonPrint(ret).empty());
}

TEST_F(IRPrinterTest, TestPrintForStmt) {
    Span sp = Span::unknown();
    auto loopVar = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, sp);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto body = std::make_shared<EvalStmt>(start, sp);
    auto forStmt = std::make_shared<ForStmt>(
        loopVar, start, stop, step, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{}, sp);
    std::string result = PythonPrint(forStmt);
    ASSERT_NE(result.find("for"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintIfStmt) {
    Span sp = Span::unknown();
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto thenBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto elseBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseBody), std::vector<VarPtr>{}, sp);
    std::string result = PythonPrint(ifStmt);
    ASSERT_NE(result.find("if"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintOpStmts) {
    Span sp = Span::unknown();
    auto expr = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto eval = std::make_shared<EvalStmt>(expr, sp);
    auto ops = std::make_shared<OpStmts>(std::vector<StmtPtr>{eval}, sp);
    ASSERT_FALSE(PythonPrint(ops).empty());
}

TEST_F(IRPrinterTest, TestPrintYieldStmt) {
    Span sp = Span::unknown();
    auto val = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, sp);
    ASSERT_FALSE(PythonPrint(yield).empty());
}

TEST_F(IRPrinterTest, TestPrintTileType) {
    std::vector<ExprPtr> shape = {std::make_shared<ConstInt>(4, DataType::INT64, Span::unknown()),
        std::make_shared<ConstInt>(8, DataType::INT64, Span::unknown())};
    auto type = std::make_shared<TileType>(shape, DataType::FP32);
    ASSERT_FALSE(PythonPrint(type).empty());
}

TEST_F(IRPrinterTest, TestPrintMemRef) {
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());
    auto mr = std::make_shared<MemRef>(MemorySpace::UB, addr, 1024, 0);
    ASSERT_FALSE(PythonPrint(mr).empty());
}

TEST_F(IRPrinterTest, TestPrintDataTypes) {
    Span sp = Span::unknown();
    auto fp16 = std::make_shared<ScalarType>(DataType::FP16);
    auto int64 = std::make_shared<ScalarType>(DataType::INT64);
    auto int32 = std::make_shared<ScalarType>(DataType::INT32);
    auto bool_t = std::make_shared<ScalarType>(DataType::BOOL);
    ASSERT_FALSE(PythonPrint(fp16).empty());
    ASSERT_FALSE(PythonPrint(int64).empty());
    ASSERT_FALSE(PythonPrint(int32).empty());
    ASSERT_FALSE(PythonPrint(bool_t).empty());
}

TEST_F(IRPrinterTest, TestPrintFuncWithParams) {
    Span sp = Span::unknown();
    auto param = std::make_shared<Var>("x",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto body = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{param}, sp);
    auto retType = std::make_shared<TensorType>(
        std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32);
    auto func =
        std::make_shared<Function>("identity", std::vector<VarPtr>{param}, std::vector<TypePtr>{retType}, body, sp);
    std::string result = PythonPrint(func);
    ASSERT_NE(result.find("identity"), std::string::npos);
    ASSERT_NE(result.find("x"), std::string::npos);
}

// ============================================================================
// Custom Prefix Tests
// ============================================================================

TEST_F(IRPrinterTest, TestCustomPrefix) {
    auto type = std::make_shared<ScalarType>(DataType::FP32);
    std::string result = PythonPrint(type, "ir");
    ASSERT_NE(result.find("ir.FP32"), std::string::npos);
}

// ============================================================================
// FormatFloatLiteral Edge Case: integer-valued float
// ============================================================================

TEST_F(IRPrinterTest, TestPrintConstFloatIntegerValue) {
    auto expr = std::make_shared<ConstFloat>(5.0, DataType::FP32, Span::unknown());
    std::string result = PythonPrint(expr);
    ASSERT_NE(result.find("5.0"), std::string::npos);
}

// ============================================================================
// Unsupported Node Type
// ============================================================================

TEST_F(IRPrinterTest, TestPrintUnsupportedNode) {
    // MemRef is an IRNode but not handled as top-level in Print(IRNodePtr)
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());
    auto memref = std::make_shared<MemRef>(MemorySpace::UB, addr, 1024, 0);
    // MemRef is dispatched via VisitExpr_ so it's actually supported as an expression.
    // Use a type as an IRNode (TypePtr can't be cast to Expr/Stmt/Function/Program)
    // Since types aren't IRNodes, test the default path with a raw IRNode
    std::string result = PythonPrint(memref);
    ASSERT_FALSE(result.empty());
}

// ============================================================================
// Type Printing - TileType with MemRef and TileView
// ============================================================================

TEST_F(IRPrinterTest, TestPrintTileTypeWithMemRef) {
    Span sp = Span::unknown();
    std::vector<ExprPtr> shape = {
        std::make_shared<ConstInt>(16, DataType::INT64, sp), std::make_shared<ConstInt>(16, DataType::INT64, sp)};
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, sp);
    auto memref = std::make_shared<MemRef>(MemorySpace::L0A, addr, 2048, 0);
    auto type = std::make_shared<TileType>(shape, DataType::FP16, std::optional<MemRefPtr>(memref));
    std::string result = PythonPrint(type);
    ASSERT_NE(result.find("Tile"), std::string::npos);
    ASSERT_NE(result.find("memref="), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintTileTypeWithTileView) {
    Span sp = Span::unknown();
    std::vector<ExprPtr> shape = {
        std::make_shared<ConstInt>(16, DataType::INT64, sp), std::make_shared<ConstInt>(16, DataType::INT64, sp)};
    std::vector<ExprPtr> vs = {
        std::make_shared<ConstInt>(8, DataType::INT64, sp), std::make_shared<ConstInt>(8, DataType::INT64, sp)};
    std::vector<ExprPtr> stride = {
        std::make_shared<ConstInt>(1, DataType::INT64, sp), std::make_shared<ConstInt>(16, DataType::INT64, sp)};
    auto offset = std::make_shared<ConstInt>(0, DataType::INT64, sp);
    TileView tv(vs, stride, offset);
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, sp);
    auto memref = std::make_shared<MemRef>(MemorySpace::L0A, addr, 2048, 0);
    auto type = std::make_shared<TileType>(
        shape, DataType::FP16, std::optional<MemRefPtr>(memref), std::optional<TileView>(tv));
    std::string result = PythonPrint(type);
    ASSERT_NE(result.find("tile_view="), std::string::npos);
    ASSERT_NE(result.find("TileView"), std::string::npos);
    ASSERT_NE(result.find("valid_shape"), std::string::npos);
    ASSERT_NE(result.find("stride"), std::string::npos);
    ASSERT_NE(result.find("start_offset"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintTensorTypeWithMemRef) {
    Span sp = Span::unknown();
    std::vector<ExprPtr> shape = {
        std::make_shared<ConstInt>(4, DataType::INT64, sp), std::make_shared<ConstInt>(8, DataType::INT64, sp)};
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, sp);
    auto memref = std::make_shared<MemRef>(MemorySpace::DDR, addr, 4096, 0);
    auto type = std::make_shared<TensorType>(shape, DataType::FP32, std::optional<MemRefPtr>(memref));
    std::string result = PythonPrint(type);
    ASSERT_NE(result.find("Tensor"), std::string::npos);
    ASSERT_NE(result.find("memref="), std::string::npos);
}

// ============================================================================
// Type Printing - TupleType, MemRefType, UnknownType
// ============================================================================

TEST_F(IRPrinterTest, TestPrintTupleType) {
    auto s1 = std::make_shared<ScalarType>(DataType::INT32);
    auto s2 = std::make_shared<ScalarType>(DataType::FP32);
    auto type = std::make_shared<TupleType>(std::vector<TypePtr>{s1, s2});
    std::string result = PythonPrint(type);
    ASSERT_NE(result.find("Tuple"), std::string::npos);
    ASSERT_NE(result.find("INT32"), std::string::npos);
    ASSERT_NE(result.find("FP32"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintMemRefType) {
    auto type = std::make_shared<MemRefType>();
    std::string result = PythonPrint(type);
    ASSERT_NE(result.find("MemRefType"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintUnknownType) {
    auto type = std::make_shared<UnknownType>();
    std::string result = PythonPrint(type);
    ASSERT_NE(result.find("UnknownType"), std::string::npos);
}

// ============================================================================
// DataType Coverage - Additional Types
// ============================================================================

TEST_F(IRPrinterTest, TestPrintAllDataTypes) {
    auto checkDtype = [](DataType dt, const std::string &expected) {
        auto type = std::make_shared<ScalarType>(dt);
        std::string result = PythonPrint(type);
        ASSERT_NE(result.find(expected), std::string::npos) << "Failed for " << expected;
    };
    checkDtype(DataType::INT4, "INT4");
    checkDtype(DataType::INT8, "INT8");
    checkDtype(DataType::INT16, "INT16");
    checkDtype(DataType::UINT4, "UINT4");
    checkDtype(DataType::UINT8, "UINT8");
    checkDtype(DataType::UINT16, "UINT16");
    checkDtype(DataType::UINT32, "UINT32");
    checkDtype(DataType::UINT64, "UINT64");
    checkDtype(DataType::FP4, "FP4");
    checkDtype(DataType::FP8E4M3FN, "FP8E4M3FN");
    checkDtype(DataType::FP8E5M2, "FP8E5M2");
    checkDtype(DataType::BF16, "BFLOAT16");
    checkDtype(DataType::HF4, "HF4");
    checkDtype(DataType::HF8, "HF8");
}

// ============================================================================
// ForStmt with IterArgs
// ============================================================================

TEST_F(IRPrinterTest, TestPrintForStmtWithIterArgs) {
    Span sp = Span::unknown();
    auto loopVar = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, sp);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, sp);

    auto initVal = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto iterArg = std::make_shared<IterArg>("acc", std::make_shared<ScalarType>(DataType::INT32), initVal, sp);

    auto rv = std::make_shared<Var>("result", std::make_shared<ScalarType>(DataType::INT32), sp);

    auto yieldVal = std::make_shared<Add>(iterArg, loopVar, DataType::INT32, sp);
    auto body = std::make_shared<YieldStmt>(std::vector<ExprPtr>{yieldVal}, sp);

    auto forStmt = std::make_shared<ForStmt>(
        loopVar, start, stop, step, std::vector<IterArgPtr>{iterArg}, body, std::vector<VarPtr>{rv}, sp);

    std::string result = PythonPrint(forStmt);
    ASSERT_NE(result.find("pl.range"), std::string::npos);
    ASSERT_NE(result.find("init_values"), std::string::npos);
    ASSERT_NE(result.find("acc"), std::string::npos);
}

// ============================================================================
// IfStmt with returnVars and yield body
// ============================================================================

TEST_F(IRPrinterTest, TestPrintIfStmtWithReturnVarsYield) {
    Span sp = Span::unknown();
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto rv = std::make_shared<Var>("out", std::make_shared<ScalarType>(DataType::INT32), sp);

    auto thenVal = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto thenYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{thenVal}, sp);

    auto elseVal = std::make_shared<ConstInt>(2, DataType::INT32, sp);
    auto elseYield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{elseVal}, sp);

    auto ifStmt =
        std::make_shared<IfStmt>(cond, thenYield, std::optional<StmtPtr>(elseYield), std::vector<VarPtr>{rv}, sp);

    std::string result = PythonPrint(ifStmt);
    ASSERT_NE(result.find("out"), std::string::npos);
    ASSERT_NE(result.find("pl.yield_"), std::string::npos);
}

// ============================================================================
// SeqStmts with YieldStmt at end and returnVars
// ============================================================================

TEST_F(IRPrinterTest, TestPrintSeqStmtsWithYieldAndReturnVars) {
    Span sp = Span::unknown();
    auto rv = std::make_shared<Var>("out", std::make_shared<ScalarType>(DataType::INT32), sp);

    auto assignVar = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto assignVal = std::make_shared<ConstInt>(10, DataType::INT32, sp);
    auto assignStmt = std::make_shared<AssignStmt>(assignVar, assignVal, sp);

    auto yieldVal = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto yieldStmt = std::make_shared<YieldStmt>(std::vector<ExprPtr>{yieldVal}, sp);

    auto seq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assignStmt, yieldStmt}, sp);

    auto cond = std::make_shared<ConstBool>(true, sp);
    auto ifStmt = std::make_shared<IfStmt>(cond, seq, std::nullopt, std::vector<VarPtr>{rv}, sp);

    std::string result = PythonPrint(ifStmt);
    ASSERT_NE(result.find("out"), std::string::npos);
    ASSERT_NE(result.find("pl.yield_"), std::string::npos);
}

// ============================================================================
// Function with non-Opaque FunctionType
// ============================================================================

TEST_F(IRPrinterTest, TestPrintFunctionOrchestration) {
    Span sp = Span::unknown();
    auto body = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto func = std::make_shared<Function>(
        "my_orch", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, sp, FunctionType::ORCHESTRATION);
    std::string result = PythonPrint(func);
    ASSERT_NE(result.find("type=pl.FunctionType.Orchestration"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintFunctionInCore) {
    Span sp = Span::unknown();
    auto body = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto func = std::make_shared<Function>(
        "my_incore", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, sp, FunctionType::IN_CORE);
    std::string result = PythonPrint(func);
    ASSERT_NE(result.find("type=pl.FunctionType.InCore"), std::string::npos);
}

// ============================================================================
// Function with multiple return types
// ============================================================================

TEST_F(IRPrinterTest, TestPrintFunctionMultipleReturnTypes) {
    Span sp = Span::unknown();
    auto v1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto v2 = std::make_shared<ConstFloat>(2.0, DataType::FP32, sp);
    auto body = std::make_shared<YieldStmt>(std::vector<ExprPtr>{v1, v2}, sp);
    auto retT1 = std::make_shared<ScalarType>(DataType::INT32);
    auto retT2 = std::make_shared<ScalarType>(DataType::FP32);
    auto func =
        std::make_shared<Function>("multi_ret", std::vector<VarPtr>{}, std::vector<TypePtr>{retT1, retT2}, body, sp);
    std::string result = PythonPrint(func);
    ASSERT_NE(result.find("tuple["), std::string::npos);
    ASSERT_NE(result.find("return"), std::string::npos);
}

// ============================================================================
// Function with yield-to-return conversion (SeqStmts body with yield)
// ============================================================================

TEST_F(IRPrinterTest, TestPrintFunctionYieldToReturn) {
    Span sp = Span::unknown();
    auto assignVar = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto assignVal = std::make_shared<ConstInt>(10, DataType::INT32, sp);
    auto assignStmt = std::make_shared<AssignStmt>(assignVar, assignVal, sp);

    auto yieldStmt = std::make_shared<YieldStmt>(std::vector<ExprPtr>{assignVar}, sp);

    auto body = std::make_shared<SeqStmts>(std::vector<StmtPtr>{assignStmt, yieldStmt}, sp);

    auto retType = std::make_shared<ScalarType>(DataType::INT32);
    auto func =
        std::make_shared<Function>("func_with_yield", std::vector<VarPtr>{}, std::vector<TypePtr>{retType}, body, sp);

    std::string result = PythonPrint(func);
    ASSERT_NE(result.find("return"), std::string::npos);
    ASSERT_NE(result.find("x"), std::string::npos);
}

// ============================================================================
// Program with multiple functions and cross-function calls
// ============================================================================

TEST_F(IRPrinterTest, TestPrintProgramWithCrossFunctionCall) {
    Span sp = Span::unknown();

    // Helper function
    auto helperParam = std::make_shared<Var>("a", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto helperBody = std::make_shared<YieldStmt>(std::vector<ExprPtr>{helperParam}, sp);
    auto helperRet = std::make_shared<ScalarType>(DataType::INT32);
    auto helper = std::make_shared<Function>(
        "helper", std::vector<VarPtr>{helperParam}, std::vector<TypePtr>{helperRet}, helperBody, sp);

    // Main function calls helper via GlobalVar
    auto mainParam = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto gvar = std::make_shared<GlobalVar>("helper");
    auto call = std::make_shared<Call>(gvar, std::vector<ExprPtr>{mainParam},
        std::vector<std::pair<std::string, std::any>>{}, std::make_shared<ScalarType>(DataType::INT32), sp);
    auto mainBody = std::make_shared<YieldStmt>(std::vector<ExprPtr>{call}, sp);
    auto mainRet = std::make_shared<ScalarType>(DataType::INT32);
    auto mainFunc =
        std::make_shared<Function>("main", std::vector<VarPtr>{mainParam}, std::vector<TypePtr>{mainRet}, mainBody, sp);

    auto program = std::make_shared<Program>(std::vector<FunctionPtr>{helper, mainFunc}, "test_prog", sp);

    std::string result = PythonPrint(program);
    ASSERT_NE(result.find("# pypto.program: test_prog"), std::string::npos);
    ASSERT_NE(result.find("import pypto.language as pl"), std::string::npos);
    ASSERT_NE(result.find("@pl.program"), std::string::npos);
    ASSERT_NE(result.find("class test_prog"), std::string::npos);
    ASSERT_NE(result.find("self.helper"), std::string::npos);
    ASSERT_NE(result.find("self"), std::string::npos);
}

TEST_F(IRPrinterTest, TestPrintProgramCustomPrefix) {
    Span sp = Span::unknown();
    auto body = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto func = std::make_shared<Function>("main", std::vector<VarPtr>{}, std::vector<TypePtr>{}, body, sp);
    auto program = std::make_shared<Program>(std::vector<FunctionPtr>{func}, "my_prog", sp);
    std::string result = PythonPrint(program, "ir");
    ASSERT_NE(result.find("from pypto import language as ir"), std::string::npos);
}

// ============================================================================
// Call with kwargs of various types
// ============================================================================

TEST_F(IRPrinterTest, TestPrintCallWithKwargsVariousTypes) {
    Span sp = Span::unknown();
    auto &reg = OpRegistry::GetInstance();
    auto tile = std::make_shared<Var>("t",
        std::make_shared<TileType>(std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp),
                                       std::make_shared<ConstInt>(8, DataType::INT64, sp)},
            DataType::FP32),
        sp);
    std::vector<std::pair<std::string, std::any>> kwargs = {
        {   "axis",    std::any(1)},
        {"keepdim", std::any(true)},
    };
    auto call = reg.Create("block.sum", {tile}, kwargs, sp);
    std::string result = PythonPrint(call);
    ASSERT_NE(result.find("axis=1"), std::string::npos);
    ASSERT_NE(result.find("keepdim=True"), std::string::npos);
}

// ============================================================================
// MakeTuple single element (trailing comma)
// ============================================================================

TEST_F(IRPrinterTest, TestPrintMakeTupleSingleElement) {
    Span sp = Span::unknown();
    auto e1 = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1}, sp);
    std::string result = PythonPrint(tup);
    ASSERT_NE(result.find("42,)"), std::string::npos);
}

// ============================================================================
// ReturnStmt with empty values
// ============================================================================

TEST_F(IRPrinterTest, TestPrintReturnStmtEmpty) {
    Span sp = Span::unknown();
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{}, sp);
    std::string result = PythonPrint(ret);
    ASSERT_EQ(result, "return");
}

// ============================================================================
// Precedence / Parenthesization Tests
// ============================================================================

TEST_F(IRPrinterTest, TestPrintNestedExprPrecedence) {
    Span sp = Span::unknown();
    // (1 + 2) * 3 should print as "(1 + 2) * 3"
    auto a = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto b = std::make_shared<ConstInt>(2, DataType::INT32, sp);
    auto c = std::make_shared<ConstInt>(3, DataType::INT32, sp);
    auto add = std::make_shared<Add>(a, b, DataType::INT32, sp);
    auto mul = std::make_shared<Mul>(add, c, DataType::INT32, sp);
    std::string result = PythonPrint(mul);
    ASSERT_NE(result.find("(1 + 2)"), std::string::npos);
    ASSERT_NE(result.find("* 3"), std::string::npos);
}

// ============================================================================
// IterArg printing
// ============================================================================

TEST_F(IRPrinterTest, TestPrintIterArg) {
    Span sp = Span::unknown();
    auto init = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto iterArg = std::make_shared<IterArg>("acc", std::make_shared<ScalarType>(DataType::INT32), init, sp);
    std::string result = PythonPrint(iterArg);
    ASSERT_EQ(result, "acc");
}

// ============================================================================
// SeqStmts printing (multiple stmts)
// ============================================================================

TEST_F(IRPrinterTest, TestPrintSeqStmts) {
    Span sp = Span::unknown();
    auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto e2 = std::make_shared<ConstInt>(2, DataType::INT32, sp);
    auto s1 = std::make_shared<EvalStmt>(e1, sp);
    auto s2 = std::make_shared<EvalStmt>(e2, sp);
    auto seq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{s1, s2}, sp);
    std::string result = PythonPrint(seq);
    ASSERT_NE(result.find("1"), std::string::npos);
    ASSERT_NE(result.find("2"), std::string::npos);
}

} // namespace ir
} // namespace pypto
