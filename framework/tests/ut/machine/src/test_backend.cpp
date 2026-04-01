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
 * \file test_backend.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/cache/function_cache.h"
#include "machine/host/backend.h"
#include "tilefwk/platform.h"

using namespace npu::tile_fwk;

class TestSuite_Backend : public testing::Test {};

extern "C" int32_t Initialize();
extern "C" bool MatchCache(const std::string& cacheKey);
extern "C" int32_t Execute(MachineTask* task, FunctionCache& cache);

TEST_F(TestSuite_Backend, AihacBackend_Err1)
{
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
    try {
        config::Reset();
    } catch (std::runtime_error&) {
    }
}

TEST_F(TestSuite_Backend, SimulationBackend_Err1)
{
    config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, false);
    config::SetPlatformConfig(KEY_ENABLE_COST_MODEL, true);
    try {
        config::Reset();
    } catch (std::runtime_error&) {
    }
}

TEST_F(TestSuite_Backend, Execute_NullTask_ReturnsZero)
{
    FunctionCache cache;
    EXPECT_EQ(Execute(nullptr, cache), 0);
}

TEST_F(TestSuite_Backend, InitializeAndMatchCache_Smoke)
{
    EXPECT_EQ(Initialize(), 0);
    EXPECT_FALSE(MatchCache("ut_non_exist_cache_key"));
}

enum ParallelMode {
    DEFAULT = 0,
    PARALLEL,
    CHILD,
};

static bool GetFunctionParallelMode(Function* func)
{
    if (func->GetDynloopAttribute()->parallel) {
        return true;
    }

    if (func->HasParent() && func->Parent().HasParent() && func->Parent().Parent().GetDynloopAttribute() &&
        func->Parent().Parent().GetDynloopAttribute()->parallel) {
        return false;
    }
    return false;
}

static bool NeedCrossDie(Function* func, bool isLoop = false)
{
    if ((Platform::Instance().GetSoc().GetNPUArch() == NPUArch::DAV_3510) &&
        (!isLoop || (GetFunctionParallelMode(func) == ParallelMode::PARALLEL))) {
        return true;
    }
    return false;
}

class TestNeedCrossDie : public testing::Test {
protected:
    void SetUp() override
    {
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
        Program::GetInstance().Reset();
        config::Reset();
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
        TileShape::Current().SetVecTile(32, 32);
        TileShape::Current().SetCubeTile({32, 32}, {32, 32}, {32, 32});
    }

    void TearDown() override
    {
        Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
        Program::GetInstance().Reset();
        config::Reset();
    }
};

TEST_F(TestNeedCrossDie, test_need_cross_die_dav3510_non_loop_returns_true)
{
    int s = 8;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");
    FUNCTION("ut_need_cross_die_1", {t0, t1}, {out})
    {
        auto x = Add(t0, t1);
        Assemble(x, {0, 0}, out);
    }
    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    bool result = NeedCrossDie(func, false);
    EXPECT_TRUE(result);
}

TEST_F(TestNeedCrossDie, test_need_cross_die_dav3510_loop_parallel_returns_true)
{
    int s = 8;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");
    FUNCTION("ut_need_cross_die_2", {t0, t1}, {out})
    {
        auto x = Add(t0, t1);
        Assemble(x, {0, 0}, out);
    }
    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    func->GetDynloopAttribute()->parallel = true;

    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    bool result = NeedCrossDie(func, true);
    EXPECT_TRUE(result);
}

TEST_F(TestNeedCrossDie, test_need_cross_die_dav3510_loop_non_parallel_returns_false)
{
    int s = 8;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");
    FUNCTION("ut_need_cross_die_3", {t0, t1}, {out})
    {
        auto x = Add(t0, t1);
        Assemble(x, {0, 0}, out);
    }
    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    func->GetDynloopAttribute()->parallel = false;

    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    bool result = NeedCrossDie(func, true);
    EXPECT_FALSE(result);
}

TEST_F(TestNeedCrossDie, test_need_cross_die_non_dav3510_returns_false)
{
    int s = 8;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");
    FUNCTION("ut_need_cross_die_4", {t0, t1}, {out})
    {
        auto x = Add(t0, t1);
        Assemble(x, {0, 0}, out);
    }
    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);

    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_UNKNOWN);
    bool result = NeedCrossDie(func, false);
    EXPECT_FALSE(result);
}

TEST_F(TestNeedCrossDie, test_need_cross_die_dav3510_loop_child_returns_false)
{
    int s = 8;
    Tensor t0(DT_FP32, {s, s}, "t0");
    Tensor t1(DT_FP32, {s, s}, "t1");
    Tensor out(DT_FP32, {s, s}, "out");
    FUNCTION("ut_need_cross_die_5", {t0, t1}, {out})
    {
        auto x = Add(t0, t1);
        Assemble(x, {0, 0}, out);
    }
    auto* func = Program::GetInstance().GetLastFunction();
    ASSERT_NE(func, nullptr);
    func->GetDynloopAttribute()->parallel = false;

    Platform::Instance().GetSoc().SetNPUArch(NPUArch::DAV_3510);
    bool result = NeedCrossDie(func, true);
    EXPECT_FALSE(result);
}
