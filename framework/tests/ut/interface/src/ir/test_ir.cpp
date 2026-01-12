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
 * \file test_ir.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "ir/utils_defop.h"
#include "ir/opcode.h"
#include "ir/builder/ir_builder.h"
#include "ir/builder/ir_context.h"
#include "ir/verify.h"

using namespace pto;

class IRTest : public testing::Test {
public:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

static_assert(MAP_SIZE(a, b, c) == 3, "Invalid MAP_SIZE 3");
static_assert(MAP_SIZE(a, b, c, d, e, f, g, h, a, b, c, d, e, f, g, h,
                       a, b, c, d, e, f, g, h, a, b, c, d, e, f, g, h) == 32, "Invalid MAP_SIZE 32");

TEST_F(IRTest, TestUtils) {
    EXPECT_EQ(std::vector<int>({2, 3, 4}), std::vector<int>({
#define ADD_1(n) (n) + 1,
        MAP(ADD_1, 1, 2, 3)
    }));
}

TEST_F(IRTest, TestOpcode) {
    EXPECT_EQ("OP_SCALAR_NEG", GetOpcodeName(Opcode::OP_SCALAR_NEG));
    EXPECT_EQ("", GetOpcodeName(Opcode::OP_INVALID));
}

TEST_F(IRTest, TestClass) {
    ScalarValuePtr lhs = std::make_shared<ScalarValue>(int64_t{2});
    ScalarValuePtr rhs = std::make_shared<ScalarValue>(int64_t{4});
    ScalarValuePtr out = std::make_shared<ScalarValue>(DataType::INT64, "aaa");
    BinaryScalarOpPtr op = std::make_shared<BinaryScalarOp>(Opcode::OP_SCALAR_ADD, rhs, lhs, out);
    EXPECT_EQ(2, op->GetNumInputOperand());
    EXPECT_EQ(1, op->GetNumOutputOperand());
}

TEST_F(IRTest, TestIRBuilder) {
    auto module = std::make_shared<ProgramModule>("main");
    IRBuilder builder;
    IRBuilderContext ctx;

    auto func = builder.CreateFunction("bbb", FunctionKind::Kernel, FunctionSignature());
    module->AddFunction(func);
    module->SetProgramEntry(func);
    builder.EnterFunctionBody(ctx, func);

    ScalarValuePtr lhs = builder.CreateConst(ctx, int64_t{2});
    ScalarValuePtr rhs = builder.CreateConst(ctx, int64_t{4});
    ScalarValuePtr out = builder.CreateScalar(ctx, DataType::INT64, "aaa");

    std::vector<ScalarValuePtr> dataList;
    BinaryScalarOpPtr op = builder.CreateBinaryScalarOp(Opcode::OP_SCALAR_ADD, lhs, rhs, out);
    builder.Emit(ctx, op);
    EXPECT_EQ(2, op->GetNumInputOperand());
    EXPECT_EQ(1, op->GetNumOutputOperand());

    ctx.PopScope();
}

TEST_F(IRTest, TestTileValueVerify) {
    TileValueVerify verifier;

    // Register all verification rules
    verifier.RegisterRule("VerifyShapeNotEmpty", RuleVerifyShapeNotEmpty);
    verifier.RegisterRule("VerifyShapeDimensions", RuleVerifyShapeDimensions);
    verifier.RegisterRule("VerifyValidShapesSize", RuleVerifyValidShapesSize);
    verifier.RegisterRule("VerifyValidShapesValues", RuleVerifyValidShapesValues);
    verifier.RegisterRule("VerifySSAId", RuleVerifySSAId);
    verifier.RegisterRule("VerifySSAName", RuleVerifySSAName);
    verifier.RegisterRule("VerifyStridesSize", RuleVerifyStridesSize);

    // Test 1: Valid TileValue - all rules should pass
    {
        std::vector<int64_t> shape = {128, 64};
        auto validTile = std::make_shared<TileValue>(shape, DataType::FP32, "test_tile");
        VerifyResult result = verifier.VerifyAllRules(*validTile);
        EXPECT_TRUE(result.passed) << "Valid tile should pass all rules";
        EXPECT_TRUE(result.errorMsg.empty() || result.errorMsg.find("failed") == std::string::npos);
    }

    // Test 2: Empty shape - VerifyShapeNotEmpty should fail
    {
        std::vector<int64_t> emptyShape = {};
        auto emptyTile = std::make_shared<TileValue>(emptyShape, DataType::FP32, "empty_tile");
        VerifyResult result = verifier.VerifyRule("VerifyShapeNotEmpty", *emptyTile);
        EXPECT_FALSE(result.passed);
        EXPECT_FALSE(result.errorMsg.empty());
        EXPECT_EQ(result.errorMsg, "Tile shape is empty");
    }

    // Test 3: Invalid dimensions (<= 0) - VerifyShapeDimensions should fail
    {
        std::vector<int64_t> invalidShape = {128, 0, 64};
        auto invalidTile = std::make_shared<TileValue>(invalidShape, DataType::FP32, "invalid_tile");
        VerifyResult result = verifier.VerifyRule("VerifyShapeDimensions", *invalidTile);
        EXPECT_FALSE(result.passed);
        EXPECT_FALSE(result.errorMsg.empty());
        EXPECT_EQ(result.errorMsg, "Tile shape contains invalid dimensions (<= 0)");
    }

    // Test 4: ValidShapes size mismatch - VerifyValidShapesSize should fail
    {
        std::vector<int64_t> shape = {128, 64};
        std::vector<ScalarValuePtr> validShapes;
        validShapes.push_back(std::make_shared<ScalarValue>(128));
        // Only one validShape but shape has 2 dimensions
        auto mismatchTile = std::make_shared<TileValue>(shape, DataType::FP32, validShapes, "mismatch_tile");
        VerifyResult result = verifier.VerifyRule("VerifyValidShapesSize", *mismatchTile);
        EXPECT_FALSE(result.passed);
        EXPECT_FALSE(result.errorMsg.empty());
        EXPECT_EQ(result.errorMsg, "ValidShapes size (1) does not match shape size (2)");
    }

    // Test 5: Invalid ValidShapes values - VerifyValidShapesValues should fail
    {
        std::vector<int64_t> shape = {128, 64};
        std::vector<ScalarValuePtr> invalidValidShapes;
        invalidValidShapes.push_back(std::make_shared<ScalarValue>(-1)); // Invalid: negative value
        invalidValidShapes.push_back(std::make_shared<ScalarValue>(64));
        auto invalidValidTile =
            std::make_shared<TileValue>(shape, DataType::FP32, invalidValidShapes, "invalid_valid_tile");
        VerifyResult result = verifier.VerifyRule("VerifyValidShapesValues", *invalidValidTile);
        EXPECT_FALSE(result.passed);
        EXPECT_FALSE(result.errorMsg.empty());
        EXPECT_EQ(result.errorMsg, "ValidShape[0] has invalid value (<= 0): -1");
    }

    // Test 6: Strides size mismatch - VerifyStridesSize should fail
    {
        std::vector<int64_t> shape = {128, 64};
        std::vector<int64_t> invalidStrides = {128}; // Only one stride but shape has 2 dimensions
        std::vector<ScalarValuePtr> validShapes;
        validShapes.push_back(std::make_shared<ScalarValue>(128));
        validShapes.push_back(std::make_shared<ScalarValue>(64));
        auto tile = std::make_shared<TileValue>("test", validShapes, shape, invalidStrides, nullptr, DataType::FP32);
        VerifyResult result = verifier.VerifyRule("VerifyStridesSize", *tile);
        EXPECT_FALSE(result.passed);
        EXPECT_FALSE(result.errorMsg.empty());
        EXPECT_EQ(result.errorMsg, "Strides size (1) does not match shape size (2)");
    }

    // Test 7: Unknown rule name
    {
        std::vector<int64_t> shape = {128, 64};
        auto tile = std::make_shared<TileValue>(shape, DataType::FP32, "test_tile");
        VerifyResult result = verifier.VerifyRule("UnknownRule", *tile);
        EXPECT_FALSE(result.passed);
        EXPECT_FALSE(result.errorMsg.empty());
        EXPECT_EQ(result.errorMsg, "Unknown rule: UnknownRule");
    }

    // Test 8: GetRuleNames
    {
        auto ruleNames = verifier.GetRuleNames();
        EXPECT_EQ(ruleNames.size(), 7) << "Should have 7 registered rules";
        // Check that some expected rules are present
        bool foundShapeNotEmpty = false;
        bool foundSSAId = false;
        for (const auto &name : ruleNames) {
            if (name == "VerifyShapeNotEmpty")
                foundShapeNotEmpty = true;
            if (name == "VerifySSAId")
                foundSSAId = true;
        }
        EXPECT_TRUE(foundShapeNotEmpty);
        EXPECT_TRUE(foundSSAId);
    }

    // Test 9: Register custom rule
    {
        auto customRule = [](const TileValue &tile) -> VerifyResult {
            if (tile.GetShape().size() > 10) {
                return {false, "Shape has too many dimensions"};
            }
            return {true, ""};
        };
        verifier.RegisterRule("CustomRule", customRule);

        std::vector<int64_t> largeShape(11, 64); // 11 dimensions
        auto largeTile = std::make_shared<TileValue>(largeShape, DataType::FP32, "large_tile");
        VerifyResult result = verifier.VerifyRule("CustomRule", *largeTile);
        EXPECT_FALSE(result.passed);
        EXPECT_EQ(result.errorMsg, "Shape has too many dimensions");
    }

    // Test 10: VerifyAllRules with invalid tile
    {
        std::vector<int64_t> invalidShape = {128, -1}; // Invalid dimension
        auto invalidTile = std::make_shared<TileValue>(invalidShape, DataType::FP32, "invalid_tile");
        VerifyResult result = verifier.VerifyAllRules(*invalidTile);
        EXPECT_FALSE(result.passed);
        EXPECT_EQ(
            result.errorMsg, "Rule 'VerifyShapeDimensions' failed: Tile shape contains invalid dimensions (<= 0); Rule "
                             "'VerifyValidShapesValues' failed: ValidShape[1] has invalid value (<= 0): -1");
    }
}