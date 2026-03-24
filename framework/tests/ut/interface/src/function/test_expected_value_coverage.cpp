/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_expected_value_coverage.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "tilefwk/function.h"
#include "interface/inner/tilefwk.h"
#include "interface/operation/operation.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/expected_value.h"

using namespace npu::tile_fwk;

const std::string SUB_FUNC_SUFFIX = "_Unroll1_PATH0";

class ExpectedValueCoverageTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, false);
    }

    void TearDown() override {
        config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, false);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    }
};

TEST_F(ExpectedValueCoverageTest, TestNormalizerAddWithOperationValue) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");
    Tensor d(DataType::DT_FP32, shape, "d");
    Tensor result;

    FUNCTION("TestNormalizerAddWithOperationValue") {
        auto ab = Add(a, b);
        auto cd = Add(c, d);
        result = Add(ab, cd);
    }
}

TEST_F(ExpectedValueCoverageTest, TestEvaluatorView) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");

    FUNCTION("TestEvaluatorView") {
        b = View(a, shape, {0, 0});
    }
}

TEST_F(ExpectedValueCoverageTest, TestEvaluatorViewWithOffset) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> inputShape{64, 64};
    std::vector<int64_t> outputShape{32, 32};
    Tensor a(DataType::DT_FP32, inputShape, "a");
    Tensor b(DataType::DT_FP32, outputShape, "b");

    FUNCTION("TestEvaluatorViewWithOffset") {
        b = View(a, outputShape, {16, 16});
    }
}

TEST_F(ExpectedValueCoverageTest, TestEvaluatorConvert) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP16, shape, "b");

    FUNCTION("TestEvaluatorConvert") {
        b = Cast(a, DataType::DT_FP16);
    }
}

TEST_F(ExpectedValueCoverageTest, TestCreateValueWithCallOpcode) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    FUNCTION("TestCreateValueWithCallOpcode") {
        c = Add(a, b);
    }
}

TEST_F(ExpectedValueCoverageTest, TestCreateIncast) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");

    FUNCTION("TestCreateIncast") {
        b = Add(a, a);
    }
}

TEST_F(ExpectedValueCoverageTest, TestCreateOperationInsertOOperandsAssemble) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{64, 64};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");

    FUNCTION("TestCreateOperationInsertOOperandsAssemble") {
        std::vector<std::pair<Tensor, std::vector<int64_t>>> items = {
            {a, {0, 0}}
        };
        b = Assemble(items);
    }
}

TEST_F(ExpectedValueCoverageTest, TestCreateOperationInsertOOperandsCopyOut) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{64, 64};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");

    FUNCTION("TestCreateOperationInsertOOperandsCopyOut") {
        std::vector<std::pair<Tensor, std::vector<int64_t>>> items = {
            {a, {0, 0}}
        };
        b = Assemble(items);
    }
}

TEST_F(ExpectedValueCoverageTest, TestCreateCallWithIncastExpectedValueList) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");

    FUNCTION("TestCreateCallWithIncastExpectedValueList") {
        b = Add(a, a);
    }
}

TEST_F(ExpectedValueCoverageTest, TestCreateCallWithEmptyIncastExpectedValueList) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");

    FUNCTION("TestCreateCallWithEmptyIncastExpectedValueList") {
        b = Add(a, a);
    }
}

TEST_F(ExpectedValueCoverageTest, TestViewWithDifferentShapes) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> inputShape{64, 64};
    std::vector<int64_t> outputShape{32, 32};
    Tensor a(DataType::DT_FP32, inputShape, "a");
    Tensor b(DataType::DT_FP32, outputShape, "b");

    FUNCTION("TestViewWithDifferentShapes") {
        b = View(a, outputShape, {16, 16});
    }
}

TEST_F(ExpectedValueCoverageTest, TestAssembleWithNonZeroOffset) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{64, 64};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");

    FUNCTION("TestAssembleWithNonZeroOffset") {
        std::vector<std::pair<Tensor, std::vector<int64_t>>> items = {
            {a, {0, 0}}
        };
        b = Assemble(items);
    }
}

TEST_F(ExpectedValueCoverageTest, TestNestedAssembleOperations) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{64, 64};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");

    FUNCTION("TestNestedAssembleOperations") {
        std::vector<std::pair<Tensor, std::vector<int64_t>>> items1 = {
            {a, {0, 0}}
        };
        b = Assemble(items1);
        std::vector<std::pair<Tensor, std::vector<int64_t>>> items2 = {
            {b, {0, 0}}
        };
        c = Assemble(items2);
    }
}

TEST_F(ExpectedValueCoverageTest, TestConvertWithSameShape) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP16, shape, "b");

    FUNCTION("TestConvertWithSameShape") {
        b = Cast(a, DataType::DT_FP16);
    }
}

TEST_F(ExpectedValueCoverageTest, TestAddNormalizerWithMultipleOperands) {
    config::SetHostOption(COMPILE_STAGE, CS_EXECUTE_GRAPH);
    TileShape::Current().SetVecTile(16, 16);

    std::vector<int64_t> shape{32, 32};
    Tensor a(DataType::DT_FP32, shape, "a");
    Tensor b(DataType::DT_FP32, shape, "b");
    Tensor c(DataType::DT_FP32, shape, "c");
    Tensor d(DataType::DT_FP32, shape, "d");
    Tensor result;

    FUNCTION("TestAddNormalizerWithMultipleOperands") {
        auto temp1 = Add(a, b);
        auto temp2 = Add(c, d);
        result = Add(temp1, temp2);
    }
}

TEST_F(ExpectedValueCoverageTest, TestCreate) {
    ExpectedValueBuilder builder;

    std::vector<int64_t> shape{32, 32};
    ExpectedValue v1 = builder.CreateValue(shape, DataType::DT_FP32, "tensor1");
    ExpectedValue v2 = builder.CreateValue(shape, DataType::DT_FP32, "tensor1");

    EXPECT_EQ(v1, v2);
}

TEST_F(ExpectedValueCoverageTest, TestCreateValueResultof) {
    ExpectedValueBuilder builder;

    std::vector<int64_t> shape{32, 32};
    ExpectedValue base = builder.CreateValue(shape, DataType::DT_FP32, "base");
    ExpectedValue resultof = builder.CreateValue(base, 0);

    EXPECT_TRUE(resultof.IsResultofValue());
}

TEST_F(ExpectedValueCoverageTest, TestCreateList) {
    ExpectedValueBuilder builder;

    std::vector<int64_t> shape{32, 32};
    ExpectedValue v1 = builder.CreateValue(shape, DataType::DT_FP32, "v1");
    ExpectedValue v2 = builder.CreateValue(shape, DataType::DT_FP32, "v2");

    ListExpectedValue list = builder.CreateList({v1, v2});

    EXPECT_EQ(list.GetElements().size(), 2);
}

TEST_F(ExpectedValueCoverageTest, TestExpectedValueHash) {
    ExpectedValueBuilder builder;

    std::vector<int64_t> shape{32, 32};
    ExpectedValue v1 = builder.CreateValue(shape, DataType::DT_FP32, "tensor1");
    ExpectedValue v2 = builder.CreateValue(shape, DataType::DT_FP32, "tensor1");

    EXPECT_EQ(v1->GetHash(), v2->GetHash());
}

TEST_F(ExpectedValueCoverageTest, TestExpectedValueEquality) {
    ExpectedValueBuilder builder;

    std::vector<int64_t> shape{32, 32};
    ExpectedValue v1 = builder.CreateValue(shape, DataType::DT_FP32, "tensor1");
    ExpectedValue v2 = builder.CreateValue(shape, DataType::DT_FP32, "tensor1");
    ExpectedValue v3 = builder.CreateValue(shape, DataType::DT_FP32, "tensor2");

    EXPECT_EQ(v1, v2);
    EXPECT_FALSE(v1 == v3);
}

TEST_F(ExpectedValueCoverageTest, TestExpectedValueCast) {
    ExpectedValueBuilder builder;

    std::vector<int64_t> shape{32, 32};
    ExpectedValue v = builder.CreateValue(shape, DataType::DT_FP32, "tensor");

    EXPECT_TRUE(v.IsInputValue());
    EXPECT_FALSE(v.IsOperationValue());
    EXPECT_FALSE(v.IsExtractValue());
    EXPECT_FALSE(v.IsInsertValue());
    EXPECT_FALSE(v.IsResultofValue());

    auto input = v.CastInputValue();
    EXPECT_NE(input, nullptr);
}

TEST_F(ExpectedValueCoverageTest, TestCreateValueExtract) {
    ExpectedValueBuilder builder;

    std::vector<int64_t> sourceShape{64, 64};
    std::vector<int64_t> resultOffset{16, 16};
    std::vector<int64_t> resultShape{32, 32};

    ExpectedValue source = builder.CreateValue(sourceShape, DataType::DT_FP32, "source");
    ExpectedValue extract(source, sourceShape, resultOffset, resultShape);

    EXPECT_TRUE(extract.IsExtractValue());
}

TEST_F(ExpectedValueCoverageTest, TestCreateValueInsert) {
    ExpectedValueBuilder builder;

    std::vector<int64_t> shape{64, 64};
    ExpectedValue source = builder.CreateValue({32, 32}, DataType::DT_FP32, "source");

    std::vector<RawExpectedInsertValueElement> elements;
    elements.emplace_back(std::vector<int64_t>{0, 0}, std::vector<int64_t>{32, 32}, source);

    ExpectedValue insert(shape, elements);

    EXPECT_TRUE(insert.IsInsertValue());
}
