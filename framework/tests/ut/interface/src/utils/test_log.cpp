/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_log.cpp
 * \brief
 */

#include "gtest/gtest.h"

#include <thread>
#include "interface/utils/log.h"

using namespace npu::tile_fwk;

class LogTest : public testing::Test {
public:
    static void TearDownTestCase() {}

    static void SetUpTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(LogTest, TestLog) {
    LoggerManager::StdLoggerEnable(true);
    auto line = LoggerManager::LineLoggerRegister("line");

    ALOG_INFO(2, TTY_RED("aaaa"));
    ALOG_DEBUG(2, "aaaa");
    EXPECT_LE(line->size(), 2);
    EXPECT_EQ("2aaaa\n", line->at(0).substr(line->at(0).size() - 6));
    LoggerManager::LineLoggerUnregister("line");

    auto time = LoggerManager::LineLoggerRegister("time");
    auto ms = std::chrono::milliseconds(10);
    for (int i = 0; i < 10; i++) {
        ALOG_INFO(i);
        std::this_thread::sleep_for(ms);
    }
    EXPECT_EQ(10, time->size());

    std::vector<int> timeList;
    for (auto &time_line : *time) {
        size_t min = time_line.find_last_of(":");
        EXPECT_NE(min, std::string::npos);
        size_t mil = time_line.find_first_of(" ", min);
        EXPECT_NE(mil, std::string::npos);
        std::string secmsec = time_line.substr(min + 1, mil - min - 1);
        double time_d = std::stod(secmsec);
        timeList.emplace_back(time_d * 1000 + 0.5);
    }
    for (size_t i = 0; i < timeList.size() - 1; i++) {
        EXPECT_LE(timeList[i] + 10, timeList[i + 1]);
        EXPECT_LT(timeList[i + 1], timeList[i] + 20);
    }
    LoggerManager::LineLoggerUnregister("time");
}
