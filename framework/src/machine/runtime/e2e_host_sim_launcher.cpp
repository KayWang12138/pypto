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
 * \file e2e_host_sim_launcher.cpp
 * \brief DebugMode E2E host simulation launcher
 */

#include "machine/runtime/e2e_host_sim_launcher.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/e2e_host_sim/host_core_context.h"
#include "machine/runtime/e2e_host_sim/host_reg_bus.h"
#include "machine/runtime/e2e_host_sim/host_sim_clock.h"
#include "machine/runtime/emulation_launcher.h"
#include "machine/runtime/rt_api/machine_rt_api.h"
#include "machine/utils/machine_error.h"
#include "tilefwk/aicpu_common.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dynamic {

namespace {
constexpr int64_t kAivPerAic = 2;
constexpr int64_t kBindCoreNum36 = 36;
constexpr int64_t kBindCoreNum18 = 18;
constexpr uint64_t kCallSubFuncTaskNs = 5000;
constexpr uint64_t kProtocolTimeoutNs = 10 * 1000 * 1000;
constexpr uint64_t kProtocolPollStepNs = 100;
constexpr uint64_t kLow32Mask = 0xFFFFFFFFULL;

inline uint64_t Low32(uint64_t v) { return v & kLow32Mask; }

class E2EHostSimSession {
public:
    E2EHostSimSession(Function* function, const DeviceLauncherConfig& config)
        : function_(function), launchConfig_(config)
    {}

    int Prepare()
    {
        if (function_ == nullptr) {
            MACHINE_LOGE(DevCommonErr::NULLPTR, "E2EHostSim session invalid function.");
            return -1;
        }

        DeviceLauncher::DeviceLauncherConfigFillDeviceInfo(launchConfig_);
        auto archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
        auto scheCpuNum =
            static_cast<int64_t>(CalcSchAicpuNumByBlockDim(launchConfig_.blockdim, launchConfig_.aicpuNum, archInfo));
        profile_ = E2EHostSimLauncher::BuildExecutionProfileByBlockDim(launchConfig_.blockdim, scheCpuNum);

        protocol_ = E2EHostSimLauncher::SimulateProtocolOnce(
            profile_, E2EHostSimLauncher::BuildDefaultPgMask(profile_.blockdim),
            E2EHostSimLauncher::ResolveBindCoreNum(profile_.aicoreLogicalNum));
        protocol_.events.clear();
        protocol_.realRtCallCount = 0;
        protocol_.forcedRecycle = false;
        protocol_.logicalClockNs = 0;
        blockIdToPhyCoreId_ = protocol_.logicalToPhysical;

        HostRegBus::Global().Reset(static_cast<size_t>(std::max<int64_t>(0, profile_.aicoreLogicalNum)));
        HostSimClock::Reset(0);

        hostRtApi_ = std::make_unique<HostRtApi>(
            [this](void*, bool, bool, Function*) {
                AppendEvent("AICPU_LAUNCH");
                return 0;
            },
            [this](void*, void*, void*, void*, bool, uint32_t blockDim, uint64_t) {
                interceptedBlockdim_ = static_cast<int64_t>(blockDim);
                AppendEvent("AICORE_INTERCEPT");
                return 0;
            },
            [this](void*, void*, void*) {
                AppendEvent("STREAM_SYNC");
                return 0;
            });
        dispatcherScope_ = std::make_unique<RtApiDispatcher::ScopedInstall>(hostRtApi_.get());

        MACHINE_LOGI(
            "[E2EHostSim] Prepare blockdim=%ld scheCpuNum=%ld aicoreLogicalNum=%ld aicpuThreadNum=%ld bindCoreNum=%ld",
            profile_.blockdim, profile_.scheCpuNum, profile_.aicoreLogicalNum, profile_.aicpuThreadNum,
            protocol_.bindCoreNum);
        return 0;
    }

    int LaunchRunOnce(DevControlFlowCache* ctrlCache)
    {
        return LaunchCommon([this, ctrlCache]() {
            return EmulationLauncher::EmulationRunOnce(function_, ctrlCache, launchConfig_);
        });
    }

    int LaunchWithTensorData(const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList)
    {
        return LaunchCommon([this, &inputList, &outputList]() {
            return EmulationLauncher::EmulationLaunchDeviceTensorData(function_, inputList, outputList, launchConfig_);
        });
    }

    int Finalize(int launchRc)
    {
        protocol_.realRtCallCount = hostRtApi_ == nullptr ? 0 : hostRtApi_->RealRtCallCount();
        protocol_.logicalClockNs = HostSimClock::NowNs();
        protocol_.interceptedBlockdim = interceptedBlockdim_ > 0 ? interceptedBlockdim_ : profile_.blockdim;

        if (interceptedBlockdim_ > 0 && interceptedBlockdim_ != profile_.blockdim) {
            MACHINE_LOGE(
                DevCommonErr::PARAM_CHECK_FAILED,
                "[E2EHostSim] intercepted blockdim mismatch, expected=%ld actual=%ld",
                profile_.blockdim, interceptedBlockdim_);
            launchRc = launchRc == 0 ? -1 : launchRc;
        }

        if (protocol_.realRtCallCount > 0) {
            MACHINE_LOGE(
                DevCommonErr::PARAM_CHECK_FAILED,
                "[E2EHostSim] real rt calls detected in E2E mode: %ld", protocol_.realRtCallCount);
            launchRc = launchRc == 0 ? -1 : launchRc;
        }

        dispatcherScope_.reset();
        hostRtApi_.reset();
        return launchRc;
    }

private:
    int LaunchCommon(const std::function<int()>& launchFn)
    {
        if (hostRtApi_ != nullptr) {
            (void)hostRtApi_->LaunchAicore(
                nullptr, nullptr, nullptr, nullptr, false,
                static_cast<uint32_t>(std::max<int64_t>(0, profile_.blockdim)), 0);
        }
        int protocolRc = RunProtocolLoop();
        int launchRc = launchFn();
        if (launchRc == 0 && protocolRc != 0) {
            launchRc = protocolRc;
        }
        if (launchRc == 0) {
            AppendEvent("RUN_OK");
        }
        return launchRc;
    }

    void AppendEvent(const std::string& evt)
    {
        std::lock_guard<std::mutex> lock(eventMutex_);
        protocol_.events.emplace_back(evt);
    }

    bool WaitUntil(const std::function<bool()>& predicate, uint64_t timeoutNs, const char* timeoutEvent)
    {
        uint64_t begin = HostSimClock::NowNs();
        while (true) {
            if (predicate()) {
                return true;
            }
            if (protocolAbort_.load(std::memory_order_relaxed)) {
                return false;
            }
            HostSimClock::AdvanceNs(kProtocolPollStepNs);
            if (HostSimClock::NowNs() - begin > timeoutNs) {
                AppendEvent(timeoutEvent);
                return false;
            }
            std::this_thread::yield();
        }
    }

    void ForceRecycleWorkers(size_t coreNum)
    {
        for (size_t i = 0; i < coreNum; ++i) {
            HostRegBus::Global().WriteMainBase(i, AICORE_SAY_GOODBYE);
        }
        AppendEvent("FORCE_RECYCLE");
    }

    void AicoreWorker(size_t logicalId)
    {
        HostCoreContext ctx;
        ctx.blockId = static_cast<int32_t>(logicalId);
        ctx.phyId = protocol_.logicalToPhysical.empty() ? static_cast<int32_t>(logicalId) : protocol_.logicalToPhysical[logicalId];
        HostCoreCtx::SetCurrent(ctx);

        uint64_t hello = (static_cast<uint64_t>(static_cast<uint32_t>(ctx.phyId)) << 32) | AICORE_SAY_HELLO;
        HostRegBus::Global().WriteCond(logicalId, hello);

        while (!protocolAbort_.load(std::memory_order_relaxed)) {
            uint64_t cmd = Low32(HostRegBus::Global().ReadMainBase(logicalId));
            if (cmd == AICORE_SAY_GOODBYE) {
                return;
            }
            if (cmd == AICORE_SAY_ACK) {
                break;
            }
            HostSimClock::AdvanceNs(kProtocolPollStepNs);
            std::this_thread::yield();
        }
        if (protocolAbort_.load(std::memory_order_relaxed)) {
            return;
        }

        HostSimClock::AdvanceNs(kCallSubFuncTaskNs);
        HostRegBus::Global().WriteCond(logicalId, AICORE_FIN_MASK);

        uint64_t stopRsp = 0;
        while (!protocolAbort_.load(std::memory_order_relaxed)) {
            uint64_t cmd = Low32(HostRegBus::Global().ReadMainBase(logicalId));
            if (cmd == AICORE_SAY_GOODBYE) {
                return;
            }
            if (cmd == AICORE_TASK_STOP + 1) {
                stopRsp = AICORE_TASK_STOP;
                break;
            }
            if (cmd == AICORE_FUNC_STOP + 1) {
                stopRsp = AICORE_FUNC_STOP;
                break;
            }
            HostSimClock::AdvanceNs(kProtocolPollStepNs);
            std::this_thread::yield();
        }
        if (protocolAbort_.load(std::memory_order_relaxed)) {
            return;
        }

        HostRegBus::Global().WriteCond(logicalId, stopRsp | AICORE_FIN_MASK);

        while (!protocolAbort_.load(std::memory_order_relaxed)) {
            uint64_t cmd = Low32(HostRegBus::Global().ReadMainBase(logicalId));
            if (cmd == AICORE_SAY_GOODBYE) {
                return;
            }
            HostSimClock::AdvanceNs(kProtocolPollStepNs);
            std::this_thread::yield();
        }
    }

    int RunCtrlProtocolLoop(size_t coreNum)
    {
        AppendEvent("AICPU_CTRL_START");

        bool helloOk = WaitUntil(
            [coreNum]() {
                for (size_t i = 0; i < coreNum; ++i) {
                    if (Low32(HostRegBus::Global().ReadCond(i)) != AICORE_SAY_HELLO) {
                        return false;
                    }
                }
                return true;
            },
            kProtocolTimeoutNs, "HELLO_TIMEOUT");
        if (!helloOk) {
            return -1;
        }
        AppendEvent("HELLO");

        for (size_t i = 0; i < coreNum; ++i) {
            auto hello = HostRegBus::Global().ReadCond(i);
            blockIdToPhyCoreId_[i] = static_cast<int32_t>((hello >> 32) & kLow32Mask);
        }
        protocol_.logicalToPhysical = blockIdToPhyCoreId_;

        for (size_t i = 0; i < coreNum; ++i) {
            HostRegBus::Global().WriteMainBase(i, AICORE_SAY_ACK);
        }
        AppendEvent("ACK");

        bool finOk = WaitUntil(
            [coreNum]() {
                for (size_t i = 0; i < coreNum; ++i) {
                    if ((HostRegBus::Global().ReadCond(i) & AICORE_FIN_MASK) == 0) {
                        return false;
                    }
                }
                return true;
            },
            kProtocolTimeoutNs, "FIN_TIMEOUT");
        if (!finOk) {
            return -1;
        }
        AppendEvent("FIN");

        for (size_t i = 0; i < coreNum; ++i) {
            uint64_t cmd = (i % 2 == 0) ? (AICORE_FUNC_STOP + 1) : (AICORE_TASK_STOP + 1);
            HostRegBus::Global().WriteMainBase(i, cmd);
        }
        AppendEvent("FUNC_STOP");
        AppendEvent("TASK_STOP");

        bool stopOk = WaitUntil(
            [coreNum]() {
                for (size_t i = 0; i < coreNum; ++i) {
                    uint64_t cond = HostRegBus::Global().ReadCond(i);
                    uint64_t condLow = Low32(cond);
                    if ((cond & AICORE_FIN_MASK) == 0) {
                        return false;
                    }
                    if (condLow != AICORE_FUNC_STOP && condLow != AICORE_TASK_STOP) {
                        return false;
                    }
                }
                return true;
            },
            kProtocolTimeoutNs, "STOP_TIMEOUT");
        if (!stopOk) {
            return -1;
        }

        for (size_t i = 0; i < coreNum; ++i) {
            HostRegBus::Global().WriteMainBase(i, AICORE_SAY_GOODBYE);
        }
        AppendEvent("GOODBYE");
        return 0;
    }

    int RunProtocolLoop()
    {
        size_t coreNum = static_cast<size_t>(std::max<int64_t>(0, profile_.aicoreLogicalNum));
        if (coreNum == 0) {
            return 0;
        }

        protocolAbort_.store(false, std::memory_order_relaxed);
        protocolDone_.store(false, std::memory_order_relaxed);

        std::vector<std::thread> aicoreWorkers;
        aicoreWorkers.reserve(coreNum);
        for (size_t logicalId = 0; logicalId < coreNum; ++logicalId) {
            aicoreWorkers.emplace_back([this, logicalId]() { AicoreWorker(logicalId); });
        }

        std::vector<std::thread> scheThreads;
        scheThreads.reserve(static_cast<size_t>(std::max<int64_t>(0, profile_.scheCpuNum)));
        for (int64_t i = 0; i < profile_.scheCpuNum; ++i) {
            scheThreads.emplace_back([this, i]() {
                while (!protocolDone_.load(std::memory_order_relaxed) &&
                    !protocolAbort_.load(std::memory_order_relaxed)) {
                    HostSimClock::AdvanceNs(static_cast<uint64_t>(i + 1));
                    std::this_thread::yield();
                }
            });
        }

        int ctrlRc = RunCtrlProtocolLoop(coreNum);
        if (ctrlRc != 0) {
            protocolAbort_.store(true, std::memory_order_relaxed);
            protocol_.forcedRecycle = true;
            ForceRecycleWorkers(coreNum);
        }
        protocolDone_.store(true, std::memory_order_relaxed);

        for (auto& t : scheThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
        for (auto& worker : aicoreWorkers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        return ctrlRc;
    }

private:
    Function* function_{nullptr};
    DeviceLauncherConfig launchConfig_{};
    E2EExecutionProfile profile_{};
    E2EProtocolSnapshot protocol_{};
    std::vector<int32_t> blockIdToPhyCoreId_{};
    int64_t interceptedBlockdim_{0};
    std::unique_ptr<HostRtApi> hostRtApi_{nullptr};
    std::unique_ptr<RtApiDispatcher::ScopedInstall> dispatcherScope_{nullptr};
    std::atomic<bool> protocolAbort_{false};
    std::atomic<bool> protocolDone_{false};
    std::mutex eventMutex_;
};
} // namespace

E2EExecutionProfile E2EHostSimLauncher::BuildExecutionProfileByBlockDim(int64_t blockdim, int64_t scheCpuNum)
{
    E2EExecutionProfile profile;
    profile.blockdim = blockdim;
    profile.scheCpuNum = scheCpuNum;
    profile.aicoreLogicalNum = blockdim * (kAivPerAic + 1);
    profile.aicpuThreadNum = scheCpuNum + 1; // 1 ctrl + sche
    return profile;
}

int64_t E2EHostSimLauncher::ResolveBindCoreNum(int64_t aicoreLogicalNum)
{
    if (aicoreLogicalNum <= 0) {
        return 0;
    }
    if (aicoreLogicalNum >= kBindCoreNum36) {
        return kBindCoreNum36;
    }
    return kBindCoreNum18;
}

uint64_t E2EHostSimLauncher::BuildDefaultPgMask(int64_t blockdim)
{
    if (blockdim <= 0) {
        return 0;
    }
    if (blockdim >= 64) {
        return ~0ULL;
    }
    uint64_t pgmask = 0;
    for (int64_t i = 0; i < blockdim; ++i) {
        pgmask |= (1ULL << i);
    }
    return pgmask;
}

std::vector<int32_t> E2EHostSimLauncher::BuildLogicalToPhysicalMap(int64_t aicoreLogicalNum, int64_t bindCoreNum)
{
    std::vector<int32_t> map;
    if (aicoreLogicalNum <= 0) {
        return map;
    }
    int64_t bindNum = bindCoreNum <= 0 ? aicoreLogicalNum : bindCoreNum;
    map.resize(static_cast<size_t>(aicoreLogicalNum));
    for (int64_t logicalId = 0; logicalId < aicoreLogicalNum; ++logicalId) {
        map[static_cast<size_t>(logicalId)] = static_cast<int32_t>(logicalId % bindNum);
    }
    return map;
}

E2EProtocolSnapshot E2EHostSimLauncher::SimulateProtocolOnce(
    const E2EExecutionProfile& profile, uint64_t pgmask, int64_t bindCoreNum)
{
    E2EProtocolSnapshot snapshot;
    snapshot.bindCoreNum = bindCoreNum;
    snapshot.pgmask = pgmask;
    snapshot.interceptedBlockdim = profile.blockdim;
    snapshot.realRtCallCount = 0;
    snapshot.logicalClockNs = kCallSubFuncTaskNs;
    snapshot.forcedRecycle = false;
    snapshot.logicalToPhysical = BuildLogicalToPhysicalMap(profile.aicoreLogicalNum, bindCoreNum);
    snapshot.events = {
        "HELLO",
        "ACK",
        "FIN",
        "FUNC_STOP",
        "TASK_STOP",
        "GOODBYE",
    };
    return snapshot;
}

int E2EHostSimLauncher::E2EHostSimRunOnce(
    Function* function, DevControlFlowCache* ctrlCache, const DeviceLauncherConfig& config)
{
    E2EHostSimSession session(function, config);
    int rc = session.Prepare();
    if (rc != 0) {
        return rc;
    }
    rc = session.LaunchRunOnce(ctrlCache);
    return session.Finalize(rc);
}

int E2EHostSimLauncher::E2EHostSimLaunchDeviceTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    const DeviceLauncherConfig& config)
{
    E2EHostSimSession session(function, config);
    int rc = session.Prepare();
    if (rc != 0) {
        return rc;
    }
    rc = session.LaunchWithTensorData(inputList, outputList);
    return session.Finalize(rc);
}

} // namespace npu::tile_fwk::dynamic
