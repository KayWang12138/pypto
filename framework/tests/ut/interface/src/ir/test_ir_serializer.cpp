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
#include "ir/serializer.h"

using namespace pto;

class IRSerializerTest : public testing::Test {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(IRSerializerTest, MemoryBuffer) {
    IRMemoryBuffer buf;
    buf.Append("abcd");
    std::vector<uint8_t> data = {'0', '1', '2', '3'};
    buf.Append(data);
    EXPECT_EQ(buf.GetRawBuffer(), "abcd0123");
}