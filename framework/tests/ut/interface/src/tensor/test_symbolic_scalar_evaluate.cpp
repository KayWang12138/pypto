/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_symbolic_scalar_evaluate.cpp
 * \brief Unit test for symbolic scalar evaluate.
 */

#include <vector>
#include <string>
#include <memory>
#include "gtest/gtest.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "interface/tensor/symbolic_scalar.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

namespace npu {
namespace tile_fwk {

class TestSymbolicScalarEvaluate : public testing::Test {
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarImmediate) {
    EvaluateSymbol evaluator;
    auto immediate = RawSymbolicImmediate::Create(42);
    auto result = evaluator.EvaluateSymbolicScalar(immediate);
    EXPECT_EQ(result, 42);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarImmediateNegative) {
    EvaluateSymbol evaluator;
    auto immediate = RawSymbolicImmediate::Create(-100);
    auto result = evaluator.EvaluateSymbolicScalar(immediate);
    EXPECT_EQ(result, -100);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarImmediateZero) {
    EvaluateSymbol evaluator;
    auto immediate = RawSymbolicImmediate::Create(0);
    auto result = evaluator.EvaluateSymbolicScalar(immediate);
    EXPECT_EQ(result, 0);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarSymbol) {
    EvaluateSymbol evaluator;
    auto symbol = RawSymbolicSymbol::Create("test_symbol");
    evaluator.UpdateSymbolDict("test_symbol", 123);
    auto result = evaluator.EvaluateSymbolicScalar(symbol);
    EXPECT_EQ(result, 123);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarSymbolNotFound) {
    EvaluateSymbol evaluator;
    auto symbol = RawSymbolicSymbol::Create("nonexistent_symbol");
    EXPECT_THROW(evaluator.EvaluateSymbolicScalar(symbol), std::exception);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionUnaryPos) {
    EvaluateSymbol evaluator;
    auto operand = RawSymbolicImmediate::Create(5);
    auto expr = RawSymbolicExpression::CreateUopPos(operand);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 5);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionUnaryNeg) {
    EvaluateSymbol evaluator;
    auto operand = RawSymbolicImmediate::Create(10);
    auto expr = RawSymbolicExpression::CreateUopNeg(operand);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, -10);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionUnaryNot) {
    EvaluateSymbol evaluator;
    auto operand = RawSymbolicImmediate::Create(0);
    auto expr = RawSymbolicExpression::CreateUopNot(operand);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 1);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryAdd) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(3);
    auto operand2 = RawSymbolicImmediate::Create(4);
    auto expr = RawSymbolicExpression::CreateBopAdd(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 7);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinarySub) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(10);
    auto operand2 = RawSymbolicImmediate::Create(3);
    auto expr = RawSymbolicExpression::CreateBopSub(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 7);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryMul) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(6);
    auto operand2 = RawSymbolicImmediate::Create(7);
    auto expr = RawSymbolicExpression::CreateBopMul(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 42);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryDiv) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(20);
    auto operand2 = RawSymbolicImmediate::Create(4);
    auto expr = RawSymbolicExpression::CreateBopDiv(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 5);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryMod) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(17);
    auto operand2 = RawSymbolicImmediate::Create(5);
    auto expr = RawSymbolicExpression::CreateBopMod(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 2);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryEq) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(5);
    auto operand2 = RawSymbolicImmediate::Create(5);
    auto expr = RawSymbolicExpression::CreateBopEq(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 1);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryNe) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(5);
    auto operand2 = RawSymbolicImmediate::Create(10);
    auto expr = RawSymbolicExpression::CreateBopNe(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 1);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryLt) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(3);
    auto operand2 = RawSymbolicImmediate::Create(5);
    auto expr = RawSymbolicExpression::CreateBopLt(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 1);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryLe) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(5);
    auto operand2 = RawSymbolicImmediate::Create(5);
    auto expr = RawSymbolicExpression::CreateBopLe(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 1);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryGt) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(10);
    auto operand2 = RawSymbolicImmediate::Create(5);
    auto expr = RawSymbolicExpression::CreateBopGt(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 1);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionBinaryGe) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(10);
    auto operand2 = RawSymbolicImmediate::Create(10);
    auto expr = RawSymbolicExpression::CreateBopGe(operand1, operand2);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 1);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionMultipleMin) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(5);
    auto operand2 = RawSymbolicImmediate::Create(3);
    auto operand3 = RawSymbolicImmediate::Create(8);
    std::vector<RawSymbolicScalarPtr> operands = {operand1, operand2, operand3};
    auto expr = RawSymbolicExpression::CreateMopMin(operands);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 3);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionMultipleMax) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(5);
    auto operand2 = RawSymbolicImmediate::Create(3);
    auto operand3 = RawSymbolicImmediate::Create(8);
    std::vector<RawSymbolicScalarPtr> operands = {operand1, operand2, operand3};
    auto expr = RawSymbolicExpression::CreateMopMax(operands);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 8);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionNested) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(2);
    auto operand2 = RawSymbolicImmediate::Create(3);
    auto operand3 = RawSymbolicImmediate::Create(4);
    auto addExpr = RawSymbolicExpression::CreateBopAdd(operand1, operand2);
    auto mulExpr = RawSymbolicExpression::CreateBopMul(addExpr, operand3);
    auto result = evaluator.EvaluateSymbolicScalar(mulExpr);
    EXPECT_EQ(result, 20);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarExpressionWithSymbol) {
    EvaluateSymbol evaluator;
    auto symbol = RawSymbolicSymbol::Create("x");
    auto operand = RawSymbolicImmediate::Create(10);
    evaluator.UpdateSymbolDict("x", 5);
    auto expr = RawSymbolicExpression::CreateBopAdd(symbol, operand);
    auto result = evaluator.EvaluateSymbolicScalar(expr);
    EXPECT_EQ(result, 15);
}

TEST_F(TestSymbolicScalarEvaluate, TestUpdateSymbolDict) {
    EvaluateSymbol evaluator;
    evaluator.UpdateSymbolDict("key1", 100);
    evaluator.UpdateSymbolDict("key2", 200);
    auto dict = evaluator.GetSymbolDict();
    EXPECT_EQ(dict.size(), 2);
    EXPECT_EQ(dict["key1"], 100);
    EXPECT_EQ(dict["key2"], 200);
}

TEST_F(TestSymbolicScalarEvaluate, TestSetSymbolDict) {
    EvaluateSymbol evaluator;
    std::unordered_map<std::string, ScalarImmediateType> newDict = {
        {"a", 10},
        {"b", 20},
        {"c", 30}
    };
    evaluator.SetSymbolDict(newDict);
    auto dict = evaluator.GetSymbolDict();
    EXPECT_EQ(dict.size(), 3);
    EXPECT_EQ(dict["a"], 10);
    EXPECT_EQ(dict["b"], 20);
    EXPECT_EQ(dict["c"], 30);
}

TEST_F(TestSymbolicScalarEvaluate, TestRuntimeIsLoopBegin) {
    EvaluateSymbol evaluator;
    EXPECT_TRUE(evaluator.RuntimeIsLoopBegin(0, 0));
    EXPECT_FALSE(evaluator.RuntimeIsLoopBegin(1, 0));
    EXPECT_FALSE(evaluator.RuntimeIsLoopBegin(5, 0));
}

TEST_F(TestSymbolicScalarEvaluate, TestRuntimeIsLoopEnd) {
    EvaluateSymbol evaluator;
    EXPECT_TRUE(evaluator.RuntimeIsLoopEnd(10, 10));
    EXPECT_TRUE(evaluator.RuntimeIsLoopEnd(11, 10));
    EXPECT_FALSE(evaluator.RuntimeIsLoopEnd(9, 10));
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateValidShapeEmpty) {
    EvaluateSymbol evaluator;
    std::vector<SymbolicScalar> dynValidShape;
    auto result = evaluator.EvaluateValidShape(dynValidShape);
    EXPECT_TRUE(result.empty());
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateValidShapeWithImmediate) {
    EvaluateSymbol evaluator;
    std::vector<SymbolicScalar> dynValidShape = {SymbolicScalar(10), SymbolicScalar(20), SymbolicScalar(30)};
    auto result = evaluator.EvaluateValidShape(dynValidShape);
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[1], 20);
    EXPECT_EQ(result[2], 30);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateValidShapeWithExpression) {
    EvaluateSymbol evaluator;
    auto operand1 = RawSymbolicImmediate::Create(5);
    auto operand2 = RawSymbolicImmediate::Create(3);
    auto expr = RawSymbolicExpression::CreateBopAdd(operand1, operand2);
    std::vector<SymbolicScalar> dynValidShape = {SymbolicScalar(expr)};
    auto result = evaluator.EvaluateValidShape(dynValidShape);
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 8);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateOffsetStatic) {
    EvaluateSymbol evaluator;
    std::vector<int64_t> offset = {1, 2, 3};
    std::vector<SymbolicScalar> dynOffset;
    auto result = evaluator.EvaluateOffset(offset, dynOffset);
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 3);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateOffsetDynamic) {
    EvaluateSymbol evaluator;
    std::vector<int64_t> offset;
    std::vector<SymbolicScalar> dynOffset = {SymbolicScalar(10), SymbolicScalar(20), SymbolicScalar(30)};
    auto result = evaluator.EvaluateOffset(offset, dynOffset);
    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 10);
    EXPECT_EQ(result[1], 20);
    EXPECT_EQ(result[2], 30);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateOffsetDynamicWithExpression) {
    EvaluateSymbol evaluator;
    std::vector<int64_t> offset;
    auto operand1 = RawSymbolicImmediate::Create(100);
    auto operand2 = RawSymbolicImmediate::Create(50);
    auto expr = RawSymbolicExpression::CreateBopSub(operand1, operand2);
    std::vector<SymbolicScalar> dynOffset = {SymbolicScalar(expr)};
    auto result = evaluator.EvaluateOffset(offset, dynOffset);
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 50);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicCallRuntimeGetViewValidShapeDim) {
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {100, 10, 50};
    auto result = evaluator.EvaluateSymbolicCall("RUNTIME_GetViewValidShapeDim", dataList, {});
    EXPECT_EQ(result, 50);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicCallRuntimeGetViewValidShapeDimLessThanViewOffset) {
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {5, 10, 50};
    auto result = evaluator.EvaluateSymbolicCall("RUNTIME_GetViewValidShapeDim", dataList, {});
    EXPECT_EQ(result, 0);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicCallRuntimeGetViewValidShapeDimGreaterThanViewShape) {
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList = {100, 10, 20};
    auto result = evaluator.EvaluateSymbolicCall("RUNTIME_GetViewValidShapeDim", dataList, {});
    EXPECT_EQ(result, 20);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicCallUnknown) {
    EvaluateSymbol evaluator;
    std::vector<ScalarImmediateType> dataList;
    EXPECT_THROW(evaluator.EvaluateSymbolicCall("UNKNOWN_FUNCTION", dataList, {}), std::exception);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarFromSymbolicScalar) {
    EvaluateSymbol evaluator;
    SymbolicScalar ss(42);
    auto result = evaluator.EvaluateSymbolicScalar(ss);
    EXPECT_EQ(result, 42);
}

TEST_F(TestSymbolicScalarEvaluate, TestEvaluateSymbolicScalarFromSymbolicScalarWithLinearArgList) {
    EvaluateSymbol evaluator;
    auto operand = RawSymbolicImmediate::Create(10);
    SymbolicScalar ss(operand);
    std::vector<SymbolicScalar> linearArgList;
    auto result = evaluator.EvaluateSymbolicScalar(ss, linearArgList);
    EXPECT_EQ(result, 10);
}

} // namespace tile_fwk
} // namespace npu
