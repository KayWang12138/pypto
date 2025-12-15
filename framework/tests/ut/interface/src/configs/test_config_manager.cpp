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
 * \file test_config_manager.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/configs/config_manager_ng.h"

using namespace npu::tile_fwk;

class TestConfigManager : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestConfigManager, GlobalConfig) {
    auto ret = config::GetPlatformConfig(npu::tile_fwk::KEY_ENABLE_COST_MODEL, true);
    EXPECT_EQ(ret, false);
    config::SetPlatformConfig(npu::tile_fwk::KEY_ENABLE_COST_MODEL, true);
    ret = config::GetPlatformConfig(npu::tile_fwk::KEY_ENABLE_COST_MODEL, false);
    EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, HostConfigs) {
    {
        auto ret = config::GetRuntimeOption<int>(WORKSPACE_RECYCLE_PERIOD);
        EXPECT_EQ(ret, 10);
        config::SetRuntimeOption<int>(WORKSPACE_RECYCLE_PERIOD, 20);
        ret = config::GetRuntimeOption<int>(WORKSPACE_RECYCLE_PERIOD);
        EXPECT_EQ(ret, 20);
    }

    {
        auto ret = config::GetHostConfig(npu::tile_fwk::KEY_STRATEGY, " ");
        EXPECT_EQ(ret, "PVC2_OOO");
        config::SetHostConfig(npu::tile_fwk::KEY_STRATEGY, "test_strategy");
        ret = config::GetHostConfig(npu::tile_fwk::KEY_STRATEGY, " ");
        EXPECT_EQ(ret, "test_strategy");
    }
}

TEST_F(TestConfigManager, SimulationConfigs) {
        auto ret = config::GetSimConfig("DEBUG_SINGLE_FUNC", true);
        EXPECT_EQ(ret, false);
        config::SetSimConfig("DEBUG_SINGLE_FUNC", true);
        ret = config::GetSimConfig("DEBUG_SINGLE_FUNC", false);
        EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, PassGloablConfig) {
    {
        auto ret = config::GetPassGlobalConfig("pass_thread_num", 0);
        EXPECT_EQ(ret, 1);
        config::SetPassGlobalConfig("pass_thread_num", 0);
        ret = config::GetPassGlobalConfig("pass_thread_num", 1);
        EXPECT_EQ(ret, 0);
    }

    {
        auto ret = config::GetPassGlobalConfig("enable_cv_fuse", true);
        EXPECT_EQ(ret, false);
        config::SetPassGlobalConfig("enable_cv_fuse", true);
        ret = config::GetPassGlobalConfig("enable_cv_fuse", false);
        EXPECT_EQ(ret, true);
    }
}

TEST_F(TestConfigManager, PassDefaultConfig) {
        auto ret = config::GetPassDefaultConfig(KEY_PRINT_FUNCTION, true);
        EXPECT_EQ(ret, false);
        config::SetPassDefaultConfig(KEY_PRINT_FUNCTION, true);
        ret = config::GetPassDefaultConfig(KEY_PRINT_FUNCTION, false);
        EXPECT_EQ(ret, true);
}

TEST_F(TestConfigManager, PassStrategies1) {
    {
        auto ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
        EXPECT_EQ(ret.expectedValueCheck, false);

        config::SetPassConfig("PVC2_OOO", "RemoveRedundantReshape", KEY_EXPECTED_VALUE_CHECK, true);
        ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
        EXPECT_EQ(ret.expectedValueCheck, true);
    }
}

TEST_F(TestConfigManager, PassStrategies2) {
    {
        auto ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
        EXPECT_EQ(ret.dumpFunctionGraphBeforePass, false);

        // set default config useful
        config::SetPassDefaultConfig(npu::tile_fwk::KEY_DUMP_FUNCTION_GRAPH_BEFORE_PASS, true);
        ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
        EXPECT_EQ(ret.dumpFunctionGraphBeforePass, true);
    }
}

TEST_F(TestConfigManager, PassStrategies3) {
    // set default config useless
    auto ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
    EXPECT_EQ(ret.expectedValueCheck, false);

    config::SetPassDefaultConfig(KEY_EXPECTED_VALUE_CHECK, true);
    ret = ConfigManager::Instance().GetPassConfigs("PVC2_OOO", "RemoveRedundantReshape");
    EXPECT_EQ(ret.expectedValueCheck, true);
}

TEST_F(TestConfigManager, Dump) {
    auto &cm = ConfigManagerNg::GetInstance();

    cm.BeginScope("scope1", {{"debug.print.edgeitems", 10L}});
    auto scope1 = cm.CurrentScope();
    cm.EndScope();

    cm.BeginScope("scope2", {{"debug.print.edgeitems", 20L}});
    {
        cm.BeginScope("scope2.1", {{"debug.print.linewidth", 120L}});
        auto scope2 = cm.CurrentScope();
        auto linewidth = AnyCast<int64_t>(scope2->GetConfig("debug.print.linewidth"));
        EXPECT_EQ(linewidth, 120);
        auto edgeitems = AnyCast<int64_t>(scope2->GetConfig("debug.print.edgeitems"));
        EXPECT_EQ(edgeitems, 20);
        cm.EndScope();
    }

    auto scope = cm.CurrentScope();
    auto linewidth = AnyCast<int64_t>(scope->GetConfig("debug.print.linewidth"));
    EXPECT_EQ(linewidth, 80);
    auto edgeitems = AnyCast<int64_t>(scope->GetConfig("debug.print.edgeitems"));
    EXPECT_EQ(edgeitems, 20);
    cm.EndScope();

    cm.BeginScope("scope3", {{"debug.print.edgeitems", 30L}});
    auto scope3 = cm.CurrentScope();
    cm.SetScope({{"debug.print.edgeitems", 35L}});
    auto scope4 = cm.CurrentScope();
    cm.EndScope();

    std::cout << cm.GetOptionsTree() << std::endl;
    std::cout << "-- scope3 -- " << std::endl;
    std::cout << scope3->ToString() << std::endl;
}
