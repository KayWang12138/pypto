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
 * \file test_tensor.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"

using namespace npu::tile_fwk;

class TestConfigStorage : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestConfigStorage, InitialGet) {
    auto config = Program::GetInstance().GetConfig();
    std::map<int,int> nullMap;
    EXPECT_EQ(config.Get<int>(PARALLEL_THRESHOLD), 20);
    EXPECT_EQ(config.Get<int>(CYCLE_UPPER_BOUND), 10000);
    EXPECT_EQ(config.Get<bool>(USE_NODE_HASH), false);
    EXPECT_EQ(config.Get<int>(CYCLES_THRESHOLD), 512);
    EXPECT_EQ(config.Get<int>(DB_TYPE), 0);
    EXPECT_EQ(config.Get<int>(NBUFFER_NUM), 1);
    EXPECT_EQ(config.Get<int>(L1_REUSE), 0);
    EXPECT_EQ((config.Get<std::map<int,int>>(L1_REUSE_MAP)), nullMap);
    EXPECT_EQ(config.Get<int>(CUBE_NBUFFER), 1);
    EXPECT_EQ((config.Get<std::map<int,int>>(CUBE_NBUFFER_MAP)), nullMap);
    EXPECT_EQ(config.Get<bool>(LOAD_BALANCE), false);
    EXPECT_EQ(config.Get<int>(COPYIN_THRESHOLD), 1024 * 1024);
    EXPECT_EQ(config.Get<uint8_t>(MACHINE_CONFIG), 0);
}

TEST_F(TestConfigStorage, HasConfig) {
    auto config = Program::GetInstance().GetConfig();
    EXPECT_EQ(config.Has("test"), false);
    EXPECT_EQ(config.Has(MACHINE_CONFIG), true);

    config.Set<int>("test", 10);
    EXPECT_EQ(config.Has("test"), true);
}

TEST_F(TestConfigStorage, ConfigSet) {
    auto config = Program::GetInstance().GetConfig();
    config.Set<int>(npu::tile_fwk::CYCLE_UPPER_BOUND, 1);
    EXPECT_EQ(config.Get<int>(CYCLE_UPPER_BOUND), 1);

    config.Set<bool>(npu::tile_fwk::USE_NODE_HASH, true);
    EXPECT_EQ(config.Get<bool>(USE_NODE_HASH), true);

    std::map<int, int> expect = {{3,4}};
    config.Set<std::map<int,int>>(CUBE_NBUFFER_MAP, expect);
    auto cubeNbuffer = config.Get<std::map<int,int>>(CUBE_NBUFFER_MAP);
    EXPECT_EQ(cubeNbuffer, expect);
}

TEST_F(TestConfigStorage, TunerInterface) {
    const int defaultCyclesThreshold = 100;
    const int defaultCyclesUpperBound = 1000;
    const int defaultParalleThreshold = 10;

    Config::GetInstance()
        .SetCyclesThreshold(defaultCyclesThreshold)
        .SetCycleUpperBound(defaultCyclesUpperBound)
        .SetMachineSchMode({MachineScheduleConfig::L2CACHE_AFFINITY_SCH})
        .SetParallelThreshold(defaultParalleThreshold)
        .SetUseNodeHash(true);

    EXPECT_EQ(Program::GetInstance().GetConfig().Get<int>(CYCLES_THRESHOLD), defaultCyclesThreshold);
    EXPECT_EQ(Program::GetInstance().GetConfig().Get<int>(CYCLE_UPPER_BOUND), defaultCyclesUpperBound);
    EXPECT_EQ(Program::GetInstance().GetConfig().Get<uint8_t>(MACHINE_CONFIG), (static_cast<uint8_t>(MachineScheduleConfig::L2CACHE_AFFINITY_SCH)));
    EXPECT_EQ(Program::GetInstance().GetConfig().Get<int>(PARALLEL_THRESHOLD), defaultParalleThreshold);
    EXPECT_EQ(Program::GetInstance().GetConfig().Get<bool>(USE_NODE_HASH), true);
}