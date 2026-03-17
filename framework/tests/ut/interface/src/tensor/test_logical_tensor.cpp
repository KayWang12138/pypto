/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_logical_tensor.cpp
 * \brief Test cases for LogicalTensor class with error codes
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/tensormap.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class TestLogicalTensor : public testing::Test {
public:
    static void SetUpTestCase() { config::SetRunDataOption(KEY_RUNTYPE, "npu"); }
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestLogicalTensor, ViewDimensionMismatch) {
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {8, 8};
    std::vector<int64_t> newOffset = {0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, ViewOffsetMismatch) {
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {8, 8, 8};
    std::vector<int64_t> newOffset = {0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, ViewShapeOutOfBounds) {
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor input(DT_FP32, shape, "input");

    auto storage = input.GetStorage(false);
    std::vector<int64_t> newShape = {20, 8, 8};
    std::vector<int64_t> newOffset = {0, 0, 0};

    EXPECT_THROW(storage->View(*Program::GetInstance().GetCurrentFunction(), newShape, newOffset), std::exception);
}

TEST_F(TestLogicalTensor, GetViewValidShapeDimensionMismatch) {
    std::vector<int64_t> validShape = {16, 16};
    std::vector<int64_t> viewShape = {16, 16, 16};

    EXPECT_THROW(GetViewValidShape(validShape, viewShape), std::exception);
}

TEST_F(TestLogicalTensor, TensorExtractDimensionMismatch) {
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor src(DT_FP32, shape, "src");

    std::vector<int64_t> offset = {0, 0};

    EXPECT_THROW(TensorExtract(src, offset), std::exception);
}

TEST_F(TestLogicalTensor, TensorInsertDimensionMismatch) {
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor src(DT_FP32, shape, "src");
    Tensor dst(DT_FP32, shape, "dst");

    std::vector<int64_t> offset = {0, 0};

    EXPECT_THROW(TensorInsert(src, offset, dst), std::exception);
}

TEST_F(TestLogicalTensor, TensorInsertShapeMismatch) {
    std::vector<int64_t> shape = {16, 16, 16};
    Tensor src(DT_FP32, shape, "src");
    Tensor dst(DT_FP32, shape, "dst");

    std::vector<int64_t> offset = {0, 0, 0};

    EXPECT_THROW(TensorInsert(src, offset, dst), std::exception);
}
