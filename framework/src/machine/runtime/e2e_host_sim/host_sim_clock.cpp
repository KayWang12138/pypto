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
 * \file host_sim_clock.cpp
 * \brief logical clock for host e2e simulation
 */

#include "machine/runtime/e2e_host_sim/host_sim_clock.h"

#include <atomic>

namespace npu::tile_fwk::dynamic {

namespace {
std::atomic<uint64_t> g_nowNs{0};
} // namespace

uint64_t HostSimClock::NowNs() { return g_nowNs.load(std::memory_order_relaxed); }

void HostSimClock::Reset(uint64_t nowNs) { g_nowNs.store(nowNs, std::memory_order_relaxed); }

void HostSimClock::AdvanceNs(uint64_t deltaNs)
{
    g_nowNs.fetch_add(deltaNs, std::memory_order_relaxed);
}

} // namespace npu::tile_fwk::dynamic
