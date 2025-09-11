/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_config.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"

#include <climits>

using namespace npu::tile_fwk;

class TestConfig : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {
        Program::GetInstance().Reset();
    }
    void TearDown() override {}
};

TEST_F(TestConfig, InitialGet) {
    EXPECT_EQ(GetOption<int64_t>(NBUFFER_NUM, 0), 1);
    EXPECT_EQ(GetOption<bool>("global_configs.platform_configs.USE_SSA", false), true);
    EXPECT_EQ(GetOption<std::string>("global_configs.platform_configs.DEVICE_PLATFORM", ""), "ASCEND_910B2");
}

TEST_F(TestConfig, SetOptionTest) {
    std::vector<std::string> key = {NBUFFER_NUM, L1_REUSE_NUM, CUBE_NBUFFER_NUM,
        SG_CYCLE_UPPER_BOUND, SG_CYCLE_LOWER_BOUND, SG_PARALLEL_NUM, COPYIN_THRESHOLD, DB_TYPE};

    for (auto &it : key) {
        Config::SetOption(it, static_cast<int64_t>(INT64_MAX));
        EXPECT_EQ(GetOption<int64_t>(it, 0), INT64_MAX);
    }

    for (auto &it : key) {
        Config::SetOption(it, static_cast<int64_t>(INT64_MIN));
        EXPECT_EQ(GetOption<int64_t>(it, 0), INT64_MIN);
    }

    std::vector<int64_t> lowerBoundValue = {1, 0, 1, 0, 0, 0, 1, 0};
    for (size_t index = 0 ; index < key.size() && index < lowerBoundValue.size(); ++index) {
        Config::SetOption(key[index], static_cast<int64_t>(lowerBoundValue[index]));
        EXPECT_EQ(GetOption<int64_t>(key[index], 999), lowerBoundValue[index]);
    }

    Config::SetOption("global_configs.simulation_configs.EXECUTE_CYCLE_THRESHOLD", static_cast<int64_t>(1));
    EXPECT_EQ(GetOption<int64_t>("global_configs.simulation_configs.EXECUTE_CYCLE_THRESHOLD", 0), 1);
    Config::SetOption("global_configs.platform_configs.USE_SSA", false);
    EXPECT_EQ(GetOption<bool>("global_configs.platform_configs.USE_SSA", true), false);
    Config::SetOption("global_configs.platform_configs.DEVICE_PLATFORM", "ASCEND_TEST");
    EXPECT_EQ(GetOption<std::string>("global_configs.platform_configs.DEVICE_PLATFORM", ""), "ASCEND_TEST");
}

TEST_F(TestConfig, SetBuildStaticTest) {
    Config::SetBuildStatic(true);
    EXPECT_EQ(GetFunctionType(), FunctionType::DYNAMIC);
}