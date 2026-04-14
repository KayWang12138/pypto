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
#include "machine/runtime/e2e_host_sim/host_aicore_entry_adapter.h"
#include "machine/runtime/e2e_host_sim/host_core_context.h"
#include "machine/runtime/e2e_host_sim/host_reg_bus.h"
#include "machine/runtime/e2e_host_sim/host_sim_clock.h"
#include "machine/runtime/rt_api/machine_rt_api.h"
#include "machine/utils/machine_error.h"
#include "interface/machine/device/tilefwk/aicore_entry.h"
#include "tilefwk/aicpu_common.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dynamic {

extern "C" int DynTileFwkBackendKernelServer(void* targ);

namespace {
constexpr int64_t kAivPerAic = 2;
constexpr int64_t kBindCoreNum36 = 36;
constexpr int64_t kBindCoreNum18 = 18;
constexpr int kCtrlThreadNum = 1;
constexpr uint64_t kCallSubFuncTaskNs = 5000;

class E2EHostSimSession {
public:
    E2EHostSimSession(Function* function, const DeviceLauncherConfig& config)
        : function_(function), launchConfig_(config)
    {}

    int Prepare(DevControlFlowCache* ctrlCache, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList)
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

        auto dynAttr = function_->GetDyndevAttribute();
        DeviceLauncher::DeviceInitDistributedContext(memUtils_, dynAttr->commGroupNames, kArgs_);
        DeviceLauncher::DeviceInitTilingData(memUtils_, kArgs_, dynAttr->devProgBinary, ctrlCache, launchConfig_, nullptr);
        DeviceLauncher::DeviceInitKernelInOuts(memUtils_, kArgs_, inputList, outputList, dynAttr->disableL2List);

        protocol_.bindCoreNum = E2EHostSimLauncher::ResolveBindCoreNum(profile_.aicoreLogicalNum);
        protocol_.pgmask = E2EHostSimLauncher::BuildDefaultPgMask(profile_.blockdim);
        protocol_.events.clear();
        protocol_.realRtCallCount = 0;
        protocol_.forcedRecycle = false;
        protocol_.logicalClockNs = 0;
        blockIdToPhyCoreId_ =
            E2EHostSimLauncher::BuildLogicalToPhysicalMap(profile_.aicoreLogicalNum, protocol_.bindCoreNum);
        protocol_.logicalToPhysical = blockIdToPhyCoreId_;

        HostRegBus::Global().Reset(static_cast<size_t>(std::max<int64_t>(0, profile_.aicoreLogicalNum)));
        HostSimClock::Reset(0);

        hostRtApi_ = std::make_unique<HostRtApi>(
            [this](void*, bool, bool, Function*) {
                AppendEvent("AICPU_LAUNCH");
                return 0;
            },
            [this](void*, void*, void*, void*, bool, uint32_t blockDim, uint64_t streamId) {
                (void)streamId;
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

    int LaunchRunOnce()
    {
        interceptedBlockdim_ = profile_.blockdim;
        int protocolRc = RunProtocolLoop();
        if (protocolRc == 0) {
            AppendEvent("RUN_OK");
        }
        return protocolRc;
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

    E2EHostMemoryUtils& GetMemUtils() { return memUtils_; }

private:

    void AppendEvent(const std::string& evt)
    {
        std::lock_guard<std::mutex> lock(eventMutex_);
        protocol_.events.emplace_back(evt);
    }

    DevAscendProgram* GetDevProg() const { return reinterpret_cast<DevAscendProgram*>(kArgs_.cfgdata); }

    KernelArgs* GetKernelArgs(size_t logicalId) const
    {
        auto* devProg = GetDevProg();
        if (devProg == nullptr || logicalId >= static_cast<size_t>(std::max<int64_t>(0, profile_.aicoreLogicalNum))) {
            return nullptr;
        }
        return reinterpret_cast<KernelArgs*>(
            static_cast<uint64_t>(devProg->devArgs.sharedBuffer) + logicalId * SHARED_BUFFER_SIZE);
    }

    void ForceRecycleWorkers(size_t coreNum)
    {
        for (size_t i = 0; i < coreNum; ++i) {
            HostRegBus::Global().WriteMainBase(i, static_cast<uint64_t>(AICORE_TASK_STOP + 1));
            auto* args = GetKernelArgs(i);
            if (args != nullptr) {
                args->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = AICORE_SAY_GOODBYE;
            }
        }
        AppendEvent("FORCE_RECYCLE");
    }

    void AicoreWorker(size_t logicalId)
    {
        HostCoreContext ctx;
        ctx.blockId = static_cast<int32_t>(logicalId);
        ctx.phyId = protocol_.logicalToPhysical.empty() ? static_cast<int32_t>(logicalId) : protocol_.logicalToPhysical[logicalId];
        HostCoreCtx::SetCurrent(ctx);
        AppendEvent(logicalId < static_cast<size_t>(profile_.blockdim) ? "AIC_ENTRY" : "AIV_ENTRY");
        KernelEntry(0, 0, 0, 0, 0, reinterpret_cast<int64_t>(kArgs_.cfgdata));
    }

    int RunProtocolLoop()
    {
        auto* devProg = GetDevProg();
        if (devProg == nullptr) {
            return -1;
        }
        size_t coreNum = static_cast<size_t>(std::max<int64_t>(0, profile_.aicoreLogicalNum));
        if (coreNum == 0) {
            return 0;
        }

        size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
        auto deviceTaskCtrlPoolAddr = devProg->GetRuntimeDataList()->GetRuntimeData() + DEV_ARGS_SIZE;
        (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
        devProg->devArgs.aicpuPerfAddr = 0UL;

        protocolAbort_.store(false, std::memory_order_relaxed);
        protocolDone_.store(false, std::memory_order_relaxed);

        std::vector<std::thread> aicoreWorkers;
        aicoreWorkers.reserve(coreNum);
        for (size_t logicalId = 0; logicalId < coreNum; ++logicalId) {
            aicoreWorkers.emplace_back([this, logicalId]() { AicoreWorker(logicalId); });
        }

        const int launchAiCpuNum = static_cast<int>(devProg->devArgs.nrAicpu + kCtrlThreadNum);
        std::vector<std::thread> aicpuThreads;
        aicpuThreads.reserve(static_cast<size_t>(launchAiCpuNum));
        std::vector<int> aicpuResults(static_cast<size_t>(launchAiCpuNum), 0);

        auto threadFun = [this, &aicpuResults](int threadIndex, uint32_t runMode, const char* eventName) {
            DeviceKernelArgs localArgs = kArgs_;
            localArgs.parameter.runMode = runMode;
            AppendEvent(eventName);
            aicpuResults[static_cast<size_t>(threadIndex)] = DynTileFwkBackendKernelServer(&localArgs);
        };

        aicpuThreads.emplace_back(threadFun, 0, RUN_SPLITTED_STREAM_CTRL, "AICPU_CTRL_START");
        for (int i = 1; i < launchAiCpuNum; ++i) {
            aicpuThreads.emplace_back(threadFun, i, RUN_SPLITTED_STREAM_SCHE, "AICPU_SCHE_START");
        }

        int launchRc = 0;
        for (auto& t : aicpuThreads) {
            if (t.joinable()) {
                t.join();
            }
        }
        for (int rc : aicpuResults) {
            if (rc != 0) {
                launchRc = rc;
                break;
            }
        }
        if (launchRc != 0) {
            protocolAbort_.store(true, std::memory_order_relaxed);
            protocol_.forcedRecycle = true;
            ForceRecycleWorkers(coreNum);
        }
        protocolDone_.store(true, std::memory_order_relaxed);

        for (auto& worker : aicoreWorkers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        protocol_.logicalClockNs = HostSimClock::NowNs();
        return launchRc;
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
    E2EHostMemoryUtils memUtils_;
    DeviceKernelArgs kArgs_{};
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

static int GetMemcpyDeviceToHostKind()
{
#ifdef BUILD_WITH_CANN
    return static_cast<int>(RT_MEMCPY_DEVICE_TO_HOST);
#else
    return 0;
#endif
}

static int GetMemcpyHostToDeviceKind()
{
#ifdef BUILD_WITH_CANN
    return static_cast<int>(RT_MEMCPY_HOST_TO_DEVICE);
#else
    return 0;
#endif
}

static std::vector<DeviceTensorData> toHostTensorData(const std::vector<DeviceTensorData>& devDataList, bool isInput)
{
    std::vector<DeviceTensorData> hostDataList;
    for (auto& devData : devDataList) {
        auto size = devData.GetDataSize();
        void* ptr = malloc(size);
        if (ptr == nullptr) {
            hostDataList.emplace_back(devData.GetDataType(), nullptr, devData.GetShape());
            continue;
        }
        if (isInput && size > 0) {
            int rc = RtApiDispatcher::Current().Memcpy(
                ptr, size, devData.GetAddr(), size, GetMemcpyDeviceToHostKind());
            if (rc != 0) {
                free(ptr);
                ptr = nullptr;
            }
        }
        hostDataList.emplace_back(devData.GetDataType(), ptr, devData.GetShape());
    }
    return hostDataList;
}

static void copyBackToDeviceTensorData(
    const std::vector<DeviceTensorData>& hostDataList, const std::vector<DeviceTensorData>& devDataList)
{
    const size_t copyNum = std::min(hostDataList.size(), devDataList.size());
    for (size_t i = 0; i < copyNum; ++i) {
        auto* dst = devDataList[i].GetAddr();
        auto* src = hostDataList[i].GetAddr();
        auto size = std::min(hostDataList[i].GetDataSize(), devDataList[i].GetDataSize());
        if (dst == nullptr || src == nullptr || size <= 0) {
            continue;
        }
        (void)RtApiDispatcher::Current().Memcpy(dst, size, src, size, GetMemcpyHostToDeviceKind());
    }
}

static void freeHostTensorData(const std::vector<DeviceTensorData>& hostDataList)
{
    for (auto& hostData : hostDataList) {
        free(hostData.GetAddr());
    }
}

int E2EHostSimLauncher::E2EHostSimRunOnce(
    Function* function, DevControlFlowCache* ctrlCache, const DeviceLauncherConfig& config)
{
    E2EHostSimSession session(function, config);

    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) =
        DeviceLauncher::BuildInputOutputFromHost(session.GetMemUtils(), inputDataList, outputDataList);

    DevControlFlowCache* launchCtrlFlowCache = nullptr;
    if (ctrlCache != nullptr) {
        launchCtrlFlowCache = reinterpret_cast<DevControlFlowCache*>(
            session.GetMemUtils().AllocZero(ctrlCache->usedCacheSize, nullptr));
        if (launchCtrlFlowCache) {
            memcpy_s(launchCtrlFlowCache, ctrlCache->usedCacheSize, ctrlCache, ctrlCache->usedCacheSize);
        }
    }

    int rc = session.Prepare(launchCtrlFlowCache, inputDeviceDataList, outputDeviceDataList);
    if (rc != 0) {
        return rc;
    }
    rc = session.LaunchRunOnce();
    return session.Finalize(rc);
}

int E2EHostSimLauncher::E2EHostSimLaunchDeviceTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    const DeviceLauncherConfig& config)
{
    E2EHostSimSession session(function, config);
    DeviceLauncher::ChangeCaptureModeRelax();
    auto inList = toHostTensorData(inputList, true);
    auto outList = toHostTensorData(outputList, false);
    DeviceLauncher::ChangeCaptureModeGlobal();

    int rc = session.Prepare(nullptr, inList, outList);
    if (rc == 0) {
        rc = session.LaunchRunOnce();
    }
    rc = session.Finalize(rc);

    freeHostTensorData(inList);
    if (rc == 0) {
        copyBackToDeviceTensorData(outList, outputList);
    }
    freeHostTensorData(outList);

    return rc;
}

} // namespace npu::tile_fwk::dynamic
