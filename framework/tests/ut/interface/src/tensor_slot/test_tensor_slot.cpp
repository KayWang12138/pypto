/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_tensor_slot.cpp
 * \brief Test cases for TensorSlot class with error codes
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/tensormap.h"
#include "tilefwk/data_type.h"

using namespace npu::tile_fwk;

class TestTensorSlot : public testing::Test {
public:
    static void SetUpTestCase() { config::SetRunDataOption(KEY_RUNTYPE, "npu"); }
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestTensorSlot, BuildSlotSetWithInvalidIncast) {
    std::vector<int64_t> shape = {16, 16};

    Function("test_func", FunctionType::STATIC, GraphType::TENSOR_GRAPH);
    Tensor input(DT_FP32, shape, "input");
    Tensor output = Add(input, input);
    EndFunction();
}

TEST_F(TestTensorSlot, GetTensorSlotUsageNotFound) {
    std::vector<int64_t> shape = {16, 16};

    Function("test_func", FunctionType::STATIC, GraphType::TENSOR_GRAPH);
    Tensor input(DT_FP32, shape, "input");
    Tensor output = Add(input, input);
    EndFunction();
}
