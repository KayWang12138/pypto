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
 * \file main.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <iostream>


// 1. 声明全局耗时映射
std::unordered_map<std::string, double> g_test_cost_map;
std::mutex g_test_cost_map_mtx;

// 2. 辅助函数：拼接测试用例全名（TestCase.TestName）
inline std::string GetTestFullName(const std::string& suite_name, const std::string& test_name) {
    return suite_name + "." + test_name;
}

// 3. 自定义函数：遍历GTest注册表，输出带耗时的测试列表
void ListTestsWithMetadata() {
    const testing::UnitTest& unit_test = *testing::UnitTest::GetInstance();

    // 遍历所有测试套件
    for (int suite_idx = 0; suite_idx < unit_test.total_test_suite_count(); ++suite_idx) {
        const testing::TestSuite* test_suite = unit_test.GetTestSuite(suite_idx);
        if (!test_suite) continue;

        std::string suite_name = test_suite->name();
        if (suite_name == "GoogleTestVerification") continue;

        // 遍历测试用例
        for (int test_idx = 0; test_idx < test_suite->total_test_count(); ++test_idx) {
            const testing::TestInfo* test_info = test_suite->GetTestInfo(test_idx);
            if (!test_info) continue;

            std::string test_name = test_info->name();
            std::string full_name = GetTestFullName(suite_name, test_name);

            // ========== 仅处理有耗时注册的用例 ==========
            std::lock_guard<std::mutex> lock(g_test_cost_map_mtx);
            auto it = g_test_cost_map.find(full_name);
            if (it != g_test_cost_map.end()) { // 仅输出注册过耗时的用例
                std::cout << full_name << "|" << it->second << std::endl;
            }
        }
    }
}
class TestExecutionCounter : public testing::EmptyTestEventListener {
public:
    uint64_t executed_count = 0;

    void OnTestStart(const testing::TestInfo&) override {
        executed_count++;
    }
};

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    auto is_meta_param = [](const std::string& arg) {
        return arg == "--gtest_list_tests_with_meta";
    };
    if (std::find_if(argv + 1, argv + argc, is_meta_param) != argv + argc) {
        ListTestsWithMetadata();
        return 0;
    }
    // 创建并注册监听器
    TestExecutionCounter counter;
    testing::UnitTest::GetInstance()->listeners().Append(&counter);

    auto ret = RUN_ALL_TESTS();

    // 移除监听器（避免析构时访问已释放内存）
    testing::UnitTest::GetInstance()->listeners().Release(&counter);
    if (counter.executed_count == 0) {
        std::cout << "Error: Can't get any case to run when using " << testing::GTEST_FLAG(filter)
                  << " to filter." << std::endl;
        ret = ret == 0 ? 1 : ret;
    }

    return ret;
}
