/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_expected_value.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/expected_value.h"

using namespace npu::tile_fwk;

class ExpectedValueTest : public testing::Test {
protected:
    void SetUp() override {
        // 初始化代码
    }

    void TearDown() override {
        // 清理代码
    }
};

// 测试 RawExpectedOperator
TEST_F(ExpectedValueTest, RawExpectedOperator_Test) {
    RawExpectedOperator operator1(Opcode::OP_ADD, {1, 2, 3});
    EXPECT_EQ(operator1.GetOpcode(), Opcode::OP_ADD);
    EXPECT_EQ(operator1.GetAttrs(), std::vector<int64_t>({1, 2, 3}));
    EXPECT_NE(operator1.GetHash(), 0);
}

// 测试 ExpectedOperator
TEST_F(ExpectedValueTest, ExpectedOperator_Test) {
    ExpectedOperator operator1(Opcode::OP_ADD, {1, 2, 3});
    EXPECT_EQ(operator1.Get()->GetOpcode(), Opcode::OP_ADD);
    EXPECT_EQ(operator1.Get()->GetAttrs(), std::vector<int64_t>({1, 2, 3}));
}

// 测试 RawExpectedValue
TEST_F(ExpectedValueTest, RawExpectedValue_Test) {
    // 使用具体子类进行测试
    RawExpectedInputValue inputValue(std::vector<int64_t>({1, 2, 3}), DataType::DT_FP32, "input");
    EXPECT_EQ(inputValue.Kind(), RawExpectedValue::ValueKind::T_EXPECTED_INPUT);
    EXPECT_NE(inputValue.GetHash(), 0);
}

// 测试 ExpectedValue
TEST_F(ExpectedValueTest, ExpectedValue_Test) {
    ExpectedValue value1(std::vector<int64_t>({1, 2, 3}), DataType::DT_FP32, "input");
    EXPECT_FALSE(value1.IsNull());
    EXPECT_TRUE(value1.IsInputValue());
    EXPECT_EQ(value1.CastInputValue()->GetShape(), std::vector<int64_t>({1, 2, 3}));
}

// 测试 RawExpectedInputValue
TEST_F(ExpectedValueTest, RawExpectedInputValue_Test) {
    RawExpectedInputValue inputValue(std::vector<int64_t>({1, 2, 3}), DataType::DT_FP32, "input");
    EXPECT_EQ(inputValue.GetShape(), std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(inputValue.GetDataType(), DataType::DT_FP32);
    EXPECT_EQ(inputValue.GetName(), "input");
}

// 测试 RawExpectedOperationValue
TEST_F(ExpectedValueTest, RawExpectedOperationValue_Test) {
    ExpectedOperator operator1(Opcode::OP_ADD, {1, 2, 3});
    std::vector<ExpectedValue> operands = {ExpectedValue(std::vector<int64_t>({1, 2, 3}), DataType::DT_FP32, "input")};
    RawExpectedOperationValue operationValue(operator1, operands);
    EXPECT_EQ(operationValue.GetOperator().Get()->GetOpcode(), Opcode::OP_ADD);
    EXPECT_EQ(operationValue.GetOperands().size(), 1);
}

// 测试 RawExpectedExtractValue
TEST_F(ExpectedValueTest, RawExpectedExtractValue_Test) {
    ExpectedValue source(std::vector<int64_t>({1, 2, 3}), DataType::DT_FP32, "input");
    RawExpectedExtractValue extractValue(source, std::vector<int64_t>({0, 0, 0}), std::vector<int64_t>({0, 0, 0}), std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(extractValue.GetSource().CastInputValue()->GetShape(), std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(extractValue.GetSourceShape(), std::vector<int64_t>({0, 0, 0}));
}

// 测试 RawExpectedInsertValue
TEST_F(ExpectedValueTest, RawExpectedInsertValue_Test) {
    std::vector<RawExpectedInsertValueElement> elements = {
        RawExpectedInsertValueElement(std::vector<int64_t>({0, 0, 0}), std::vector<int64_t>({1, 2, 3}), ExpectedValue(std::vector<int64_t>({1, 2, 3}), DataType::DT_FP32, "input"))
    };
    RawExpectedInsertValue insertValue(std::vector<int64_t>({1, 2, 3}), elements);
    EXPECT_EQ(insertValue.GetShape(), std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(insertValue.GetElements().size(), 1);
}

// 测试 RawExpectedResultofValue
TEST_F(ExpectedValueTest, RawExpectedResultofValue_Test) {
    ExpectedValue resultof(std::vector<int64_t>({1, 2, 3}), DataType::DT_FP32, "input");
    RawExpectedResultofValue resultofValue(resultof, 0);
    EXPECT_EQ(resultofValue.GetResultof().CastInputValue()->GetShape(), std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(resultofValue.GetIndex(), 0);
}

// 测试 ListExpectedValue
TEST_F(ExpectedValueTest, ListExpectedValue_Test) {
    std::vector<ExpectedValue> elements = {ExpectedValue(std::vector<int64_t>({1, 2, 3}), DataType::DT_FP32, "input")};
    ListExpectedValue listValue(elements);
    EXPECT_EQ(listValue.GetElements().size(), 1);
    EXPECT_NE(listValue.GetHash(), 0);
}

// 测试 ExpectedValueBuilder
TEST_F(ExpectedValueTest, ExpectedValueBuilder_Test) {
    ExpectedValueBuilder builder;
    // 创建一个 Function 对象
    Function func(Program::GetInstance(), "funcMagicName", "funcRawName", nullptr);
    Operation operation(func, Opcode::OP_ADD);
    ExpectedOperator operator1 = builder.CreateOperator(operation);
    EXPECT_EQ(operator1.Get()->GetOpcode(), Opcode::OP_ADD);
    EXPECT_EQ(operator1.Get()->GetAttrs(), std::vector<int64_t>());
}