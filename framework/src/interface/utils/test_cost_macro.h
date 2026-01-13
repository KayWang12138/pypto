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
 * \file test_cost_macro.h
 * \brief
 */

#ifndef GTEST_COST_MINIMAL_H
#define GTEST_COST_MINIMAL_H

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <mutex>

// 声明全局变量
extern std::unordered_map<std::string, double> g_test_cost_map;
extern std::mutex g_test_cost_map_mtx;


// 辅助函数：拼接测试用例全名
inline std::string GetTestFullName(const std::string& suite_name, const std::string& test_name) {
    return suite_name + "." + test_name;
}

// 核心宏
#define TEST_WITH_COST(TestCaseName, TestName, CostSeconds)                 \
static bool g_##TestCaseName##_##TestName##_cost_registered = []() {        \
    std::lock_guard<std::mutex> lock(g_test_cost_map_mtx);                 \
    std::string full_name = GetTestFullName(#TestCaseName, #TestName);     \
    g_test_cost_map[full_name] = CostSeconds;                              \
    return true;                                                           \
}();                                                                        \
TEST(TestCaseName, TestName)

#define TEST_F_WITH_COST(TestFixtureClass, TestName, CostSeconds)          \
static bool g_##TestFixtureClass##_##TestName##_cost_registered = []() {    \
    std::lock_guard<std::mutex> lock(g_test_cost_map_mtx);                 \
    std::string full_name = GetTestFullName(#TestFixtureClass, #TestName); \
    g_test_cost_map[full_name] = CostSeconds;                              \
    return true;                                                           \
}();                                                                        \
TEST_F(TestFixtureClass, TestName)

#endif // GTEST_COST_MINIMAL_H