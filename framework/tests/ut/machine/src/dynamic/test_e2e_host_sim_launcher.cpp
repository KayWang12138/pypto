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
 * \file test_e2e_host_sim_launcher.cpp
 * \brief unit tests for E2E host sim launch routing and memory abstraction
 */

#include "gtest/gtest.h"
#include "interface/configs/config_manager.h"
#include "machine/runtime/e2e_host_sim_launcher.h"
#include "machine/runtime/launcher_router.h"
#include "machine/runtime/memory/memory_manager.h"

using namespace npu::tile_fwk;
using namespace npu::tile_fwk::dynamic;

class TestE2EHostSimLauncher : public testing::Test {
public:
    void SetUp() override { config::Reset(); }
    void TearDown() override { config::Reset(); }
};

TEST_F(TestE2EHostSimLauncher, ResolveByDebugMode)
{
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_DEBUG_NONE), LaunchMode::DEVICE_RT);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_DEBUG_ALL), LaunchMode::EMULATION);
    EXPECT_EQ(LauncherRouter::ResolveByDebugMode(CFG_DEBUG_E2E_HOST_SIM), LaunchMode::E2E_HOST_SIM);
}

TEST_F(TestE2EHostSimLauncher, BuildExecutionProfileByBlockDim)
{
    auto profile = E2EHostSimLauncher::BuildExecutionProfileByBlockDim(24, 3);
    EXPECT_EQ(profile.blockdim, 24);
    EXPECT_EQ(profile.scheCpuNum, 3);
    EXPECT_EQ(profile.aicoreLogicalNum, 72);
    EXPECT_EQ(profile.aicpuThreadNum, 4);
}

TEST_F(TestE2EHostSimLauncher, HostMemoryManagerBasic)
{
    HostMemoryManager mem;
    constexpr size_t kSize = 64;
    uint8_t src[kSize] = {0};
    uint8_t dst[kSize] = {0};
    src[0] = 42;
    auto* ptr = static_cast<uint8_t*>(mem.Alloc(kSize));
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(mem.Memcpy(ptr, kSize, src, kSize, 0), 0);
    EXPECT_EQ(mem.Memcpy(dst, kSize, ptr, kSize, 0), 0);
    EXPECT_EQ(dst[0], 42);
    EXPECT_EQ(mem.Free(ptr), 0);
}

