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
#include "machine/runtime/e2e_host_sim/host_aicore_entry_adapter.h"
#include "machine/runtime/e2e_host_sim/host_reg_bus.h"
#include "machine/runtime/e2e_host_sim/host_sim_clock.h"
#include "machine/runtime/e2e_host_sim_launcher.h"
#include "machine/runtime/launcher_router.h"
#include "machine/runtime/memory/memory_manager.h"
#include "machine/runtime/rt_api/machine_rt_api.h"

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

TEST_F(TestE2EHostSimLauncher, BindCoreAndMapping)
{
    EXPECT_EQ(E2EHostSimLauncher::ResolveBindCoreNum(72), 36);
    EXPECT_EQ(E2EHostSimLauncher::ResolveBindCoreNum(24), 18);
    EXPECT_EQ(E2EHostSimLauncher::ResolveBindCoreNum(0), 0);

    auto mapping = E2EHostSimLauncher::BuildLogicalToPhysicalMap(40, 18);
    ASSERT_EQ(mapping.size(), 40);
    EXPECT_EQ(mapping[0], 0);
    EXPECT_EQ(mapping[17], 17);
    EXPECT_EQ(mapping[18], 0);
    EXPECT_EQ(mapping[39], 3);

    auto noBindMapping = E2EHostSimLauncher::BuildLogicalToPhysicalMap(5, 0);
    ASSERT_EQ(noBindMapping.size(), 5);
    EXPECT_EQ(noBindMapping[0], 0);
    EXPECT_EQ(noBindMapping[4], 4);
}

TEST_F(TestE2EHostSimLauncher, PgmaskAndProtocol)
{
    EXPECT_EQ(E2EHostSimLauncher::BuildDefaultPgMask(5), 0x1FULL);
    EXPECT_EQ(E2EHostSimLauncher::BuildDefaultPgMask(64), ~0ULL);

    auto profile = E2EHostSimLauncher::BuildExecutionProfileByBlockDim(8, 2);
    auto snapshot = E2EHostSimLauncher::SimulateProtocolOnce(profile, E2EHostSimLauncher::BuildDefaultPgMask(5), 18);
    EXPECT_EQ(snapshot.bindCoreNum, 18);
    EXPECT_EQ(snapshot.pgmask, 0x1FULL);
    EXPECT_EQ(snapshot.logicalClockNs, 5000);
    EXPECT_FALSE(snapshot.forcedRecycle);
    ASSERT_GE(snapshot.events.size(), 6);
    EXPECT_EQ(snapshot.events[0], "HELLO");
    EXPECT_EQ(snapshot.events[1], "ACK");
    EXPECT_EQ(snapshot.events[2], "FIN");
    EXPECT_EQ(snapshot.events[3], "FUNC_STOP");
    EXPECT_EQ(snapshot.events[4], "TASK_STOP");
    EXPECT_EQ(snapshot.events[5], "GOODBYE");
    ASSERT_EQ(snapshot.logicalToPhysical.size(), static_cast<size_t>(profile.aicoreLogicalNum));
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

TEST_F(TestE2EHostSimLauncher, HostSimClockAndRegBusBasic)
{
    HostSimClock::Reset(0);
    EXPECT_EQ(HostSimClock::NowNs(), 0);
    HostSimClock::AdvanceNs(5000);
    EXPECT_EQ(HostSimClock::NowNs(), 5000);

    HostRegBus::Global().Reset(4);
    HostRegBus::Global().WriteMainBase(1, 0x1234);
    HostRegBus::Global().WriteCond(1, 0x5678);
    EXPECT_EQ(HostRegBus::Global().ReadMainBase(1), 0x1234);
    EXPECT_EQ(HostRegBus::Global().ReadCond(1), 0x5678);
}

TEST_F(TestE2EHostSimLauncher, CallSubFuncTaskFixed5us)
{
    HostSimClock::Reset(0);
    CallSubFuncTask(0, nullptr, 0, nullptr);
    EXPECT_EQ(HostSimClock::NowNs(), 5000);
}

TEST_F(TestE2EHostSimLauncher, HostRtApiFallbackCount)
{
    HostRtApi api(
        [](void*, bool, bool, Function*) { return 0; },
        [](void*, void*, void*, void*, bool, uint32_t, uint64_t) { return 0; },
        [](void*, void*, void*) { return 0; });

    EXPECT_EQ(api.LaunchAicpu(nullptr, true, false, nullptr), 0);
    EXPECT_EQ(api.LaunchAicore(nullptr, nullptr, nullptr, nullptr, false, 24, 0), 0);
    EXPECT_EQ(api.StreamSync(nullptr, nullptr, nullptr), 0);
    EXPECT_EQ(api.RealRtCallCount(), 0);
}

TEST_F(TestE2EHostSimLauncher, HostRtApiFallbackFailureCount)
{
    HostRtApi api({}, {}, {}, true);

    EXPECT_NE(api.LaunchAicpu(nullptr, true, false, nullptr), 0);
    EXPECT_NE(api.LaunchAicore(nullptr, nullptr, nullptr, nullptr, false, 24, 0), 0);
    EXPECT_NE(api.StreamSync(nullptr, nullptr, nullptr), 0);
    EXPECT_EQ(api.RealRtCallCount(), 3);
}
