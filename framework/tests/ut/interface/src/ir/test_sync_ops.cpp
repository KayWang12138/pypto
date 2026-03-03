/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sync_ops.cpp
 * \brief Unit tests for sync operations registration
 */

#include "gtest/gtest.h"

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "ir/expr.h"
#include "ir/op_registry.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class SyncOpsTest : public testing::Test {};

TEST_F(SyncOpsTest, TestSyncSrcRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("system.sync_src"));
}

TEST_F(SyncOpsTest, TestSyncDstRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("system.sync_dst"));
}

TEST_F(SyncOpsTest, TestBarVRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("system.bar_v"));
}

TEST_F(SyncOpsTest, TestBarMRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("system.bar_m"));
}

TEST_F(SyncOpsTest, TestBarAllRegistered) {
    auto &registry = OpRegistry::GetInstance();
    ASSERT_TRUE(registry.IsRegistered("system.bar_all"));
}

TEST_F(SyncOpsTest, TestSyncSrcGetEntry) {
    auto &registry = OpRegistry::GetInstance();
    auto &entry = registry.GetEntry("system.sync_src");
    ASSERT_EQ(entry.GetName(), "system.sync_src");
    ASSERT_EQ(entry.GetDescription(), "Send a synchronization signal (Set Flag)");
    ASSERT_EQ(entry.GetOpCategory(), "SyncOp");
}

TEST_F(SyncOpsTest, TestSyncDstGetEntry) {
    auto &registry = OpRegistry::GetInstance();
    auto &entry = registry.GetEntry("system.sync_dst");
    ASSERT_EQ(entry.GetDescription(), "Wait for a synchronization signal (Wait Flag)");
}

TEST_F(SyncOpsTest, TestBarVGetEntry) {
    auto &registry = OpRegistry::GetInstance();
    auto &entry = registry.GetEntry("system.bar_v");
    ASSERT_EQ(entry.GetDescription(), "Vector unit barrier");
}

TEST_F(SyncOpsTest, TestBarMGetEntry) {
    auto &registry = OpRegistry::GetInstance();
    auto &entry = registry.GetEntry("system.bar_m");
    ASSERT_EQ(entry.GetDescription(), "Matrix unit barrier");
}

TEST_F(SyncOpsTest, TestBarAllGetEntry) {
    auto &registry = OpRegistry::GetInstance();
    auto &entry = registry.GetEntry("system.bar_all");
    ASSERT_EQ(entry.GetDescription(), "Global barrier synchronization");
}

TEST_F(SyncOpsTest, TestSyncSrcCreate) {
    auto &registry = OpRegistry::GetInstance();
    std::vector<ExprPtr> args;
    std::vector<std::pair<std::string, std::any>> kwargs;
    kwargs.emplace_back("set_pipe", std::any(1));
    kwargs.emplace_back("wait_pipe", std::any(2));
    kwargs.emplace_back("event_id", std::any(0));
    auto call = registry.Create("system.sync_src", args, kwargs, Span::Unknown());
    ASSERT_NE(call, nullptr);
}

TEST_F(SyncOpsTest, TestBarVCreate) {
    auto &registry = OpRegistry::GetInstance();
    std::vector<ExprPtr> args;
    auto call = registry.Create("system.bar_v", args, Span::Unknown());
    ASSERT_NE(call, nullptr);
}

TEST_F(SyncOpsTest, TestBarMCreate) {
    auto &registry = OpRegistry::GetInstance();
    std::vector<ExprPtr> args;
    auto call = registry.Create("system.bar_m", args, Span::Unknown());
    ASSERT_NE(call, nullptr);
}

TEST_F(SyncOpsTest, TestBarAllCreate) {
    auto &registry = OpRegistry::GetInstance();
    std::vector<ExprPtr> args;
    auto call = registry.Create("system.bar_all", args, Span::Unknown());
    ASSERT_NE(call, nullptr);
}

} // namespace ir
} // namespace pypto
