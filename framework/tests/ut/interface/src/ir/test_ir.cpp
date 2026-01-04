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
#include "ir/utils.h"

using namespace npu::tile_fwk;

class IRTest : public testing::Test {
public:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

static_assert(MAP_SIZE(a, b, c) == 3, "Invalid MAP_SIZE");

TEST_F(IRTest, TestUtils) {
}