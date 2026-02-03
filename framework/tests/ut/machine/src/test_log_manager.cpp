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
 * \file test_log_manager.cpp
 * \brief
 */

#include <gtest/gtest.h>
#define private public
#include "utils/log_manager.h"
#undef private

namespace tile::fwk {
class TestLogManager : public testing::Test {
public:
    void SetUp() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        unsetenv("ASCEND_MODULE_LOG_LEVEL");
        unsetenv("ASCEND_PROCESS_LOG_PATH");
    }
    void TearDown() override {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        unsetenv("ASCEND_MODULE_LOG_LEVEL");
        unsetenv("ASCEND_PROCESS_LOG_PATH");
    }
};

TEST_F(TestLogManager, test_log_level_case0) {
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), false);
    log_manager.Record(LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
    log_manager.Record(LogLevel::INFO, "And I aiming it right at you, right at you");
    log_manager.Record(LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
    log_manager.Record(LogLevel::INFO, "And I'm so lost without you, without you");
}

TEST_F(TestLogManager, test_log_level_case1) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    log_manager.Record(LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
    log_manager.Record(LogLevel::INFO, "And I aiming it right at you, right at you");
    log_manager.Record(LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
    log_manager.Record(LogLevel::INFO, "And I'm so lost without you, without you");
}

TEST_F(TestLogManager, test_log_level_case2) {
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
    setenv("ASCEND_MODULE_LOG_LEVEL", "PYPTO=2", 1);
    LogManager log_manager;
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::ERROR), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::INFO), true);
    EXPECT_EQ(log_manager.CheckLevel(LogLevel::DEBUG), false);
    log_manager.Record(LogLevel::INFO, "I'm a space-bound %s and your heart's the moon", "rocketship");
    log_manager.Record(LogLevel::INFO, "And I aiming it right at you, right at you");
    log_manager.Record(LogLevel::INFO, "%d miles on a clear night in %s", 250000, "June");
    log_manager.Record(LogLevel::INFO, "And I'm so lost without you, without you");
}
}
