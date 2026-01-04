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
#include "ir/def_utils.h"
#include "ir/opcode.h"
#include "ir/tile_graph.h"

using namespace pto;
using namespace npu::tile_fwk;

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
    std::vector<ScalarOpPtr> dataList;
    BinaryScalarOpPtr op = std::make_shared<BinaryScalarOp>(Opcode::OP_SCALAR_ADD, dataList, dataList);
    EXPECT_EQ(0, op->GetInOperandSize());
    EXPECT_EQ(0, op->GetOutOperandSize());
}