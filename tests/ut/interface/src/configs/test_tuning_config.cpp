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
    const int parallel_num = 20;
    const int cycle_upper_bound = 10000;
    const int cycle_lower_bound = 512;
    const int copyin_threshold = 1024 * 1024;
    const int sg_cube_parallel_num = 24;
    const int sg_vec_parallel_num = 48;
    EXPECT_EQ(config.Get<int>(SG_PARALLEL_NUM), parallel_num);
    EXPECT_EQ(config.Get<int>(SG_CYCLE_UPPER_BOUND), cycle_upper_bound);
    EXPECT_EQ(config.Get<int>(SG_CYCLE_LOWER_BOUND), cycle_lower_bound);
    EXPECT_EQ(config.Get<int>(L1_REUSE), 0);
    EXPECT_EQ((config.Get<std::map<int,int>>(L1_REUSE_MAP)), nullMap);
    EXPECT_EQ(config.Get<int>(CUBE_NBUFFER), 1);
    EXPECT_EQ((config.Get<std::map<int,int>>(CUBE_NBUFFER_MAP)), nullMap);
    EXPECT_EQ(config.Get<int>(COPYIN_THRESHOLD), copyin_threshold);
    EXPECT_EQ(config.Get<uint8_t>(MACHINE_CONFIG), 0);
    EXPECT_EQ(config.Get<int>(NBUFFER_MERGE_MODE), 1);
    EXPECT_EQ(config.Get<int>(SG_CUBE_PARALLEL_NUM), sg_cube_parallel_num);
    EXPECT_EQ(config.Get<int>(SG_VEC_PARALLEL_NUM), sg_vec_parallel_num);
    EXPECT_EQ((config.Get<std::map<int, int>>(VEC_NBUFFER_MAP)), nullMap);
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
    config.Set<int>(npu::tile_fwk::SG_CYCLE_UPPER_BOUND, 1);
    EXPECT_EQ(config.Get<int>(SG_CYCLE_UPPER_BOUND), 1);

    std::map<int, int> expect = {{3,4}};
    config.Set<std::map<int,int>>(CUBE_NBUFFER_MAP, expect);
    auto cubeNbuffer = config.Get<std::map<int,int>>(CUBE_NBUFFER_MAP);
    EXPECT_EQ(cubeNbuffer, expect);
}