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
 * \file test_op_registry.cpp
 * \brief UT for op_registry
 */

#include <gtest/gtest.h>

#include "tilefwk/op_registry.h"

using namespace npu::tile_fwk;

static void NoOpImpl(uint64_t) {}

TEST(OpRegistryTest, CreateOrGetAndAddImplFunc) {
    auto& registry = OpImplRegistry::GetInstance();
    auto reg = registry.CreateOrGetOpRegister("OpRegistryTestOp");
    ASSERT_NE(reg, nullptr);
    // configKey uses low 52 bits only (SUB_KEY_MASK high 12 bits)
    reg->AddImplFunc(0, NoOpImpl);
    auto func = reg->GetOpImplFunc(0);
    EXPECT_NE(func, nullptr);
    auto keys = reg->GetAllConfigKeys();
    EXPECT_EQ(keys.size(), 1u);
    EXPECT_EQ(keys[0], 0u);
}

TEST(OpRegistryTest, RegistryGetOpImplFuncAndGetAllConfigKeys) {
    auto& registry = OpImplRegistry::GetInstance();
    auto reg = registry.CreateOrGetOpRegister("OpRegistryTestOp2");
    reg->AddImplFunc(1, NoOpImpl);
    auto func = registry.GetOpImplFunc("OpRegistryTestOp2", 1);
    EXPECT_NE(func, nullptr);
    EXPECT_EQ(registry.GetOpImplFunc("NotExist", 0), nullptr);
    auto keys = registry.GetAllConfigKeys("OpRegistryTestOp2");
    EXPECT_EQ(keys.size(), 1u);
    EXPECT_EQ(keys[0], 1u);
    auto emptyKeys = registry.GetAllConfigKeys("NotExist");
    EXPECT_TRUE(emptyKeys.empty());
}

TEST(OpRegistryTest, AddImplFuncMap) {
    auto reg = OpImplRegistry::GetInstance().CreateOrGetOpRegister("OpRegistryTestOpMap");
    std::map<uint64_t, OpImplFunc> m;
    m[0] = NoOpImpl;
    m[2] = NoOpImpl;
    reg->AddImplFunc(m);
    EXPECT_NE(reg->GetOpImplFunc(0), nullptr);
    EXPECT_NE(reg->GetOpImplFunc(2), nullptr);
}
