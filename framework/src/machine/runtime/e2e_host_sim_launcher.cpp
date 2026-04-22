/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "e2e_host_sim_launcher.h"

#include <atomic>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <vector>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif

#include "machine/runtime/e2e_host_sim/host_aicore_entry_adapter.h"
#include "machine/runtime/e2e_host_sim/host_core_context.h"
#include "machine/runtime/e2e_host_sim/host_reg_bus.h"
#include "machine/runtime/e2e_host_sim/host_sim_clock.h"
#include "machine/runtime/dump_device_perf.h"
#include "interface/machine/device/tilefwk/aicore_entry.h"
#include "machine/device/dynamic/device_utils.h"

extern "C" int DynTileFwkBackendKernelServer(void* targ);

namespace npu::tile_fwk::dynamic {

constexpr int AIV_NUM_PER_AI_CORE_LOCAL = 2;
constexpr int SCHE_CPU_NUM = 3;
constexpr int AIC_PER_SCHE = 8;
constexpr int AICORE_PER_SCHE = 24;
constexpr int CTRL_THREAD_CPU_BASE = 0;
constexpr int SCHE_THREAD_CPU_BASE = 1;
constexpr int AICORE_THREAD_CPU_BASE_36CORE = 4;
constexpr int AICORE_THREAD_CPU_BASE_24CORE = 4;
constexpr int AICORE_CORES_PER_SCHE_36CORE = 12;
constexpr int AICORE_CORES_PER_SCHE_24CORE = 7;

constexpr uint32_t E2E_HOST_SIM_NR_AIC = 24;
constexpr uint32_t E2E_HOST_SIM_NR_AIV = 48;
constexpr uint32_t E2E_HOST_SIM_NR_VALID_AIC = 24;
constexpr uint32_t E2E_HOST_SIM_NR_AICPU = 4;
constexpr uint32_t E2E_HOST_SIM_SCHE_CPU_NUM = 3;

static int GetSystemCpuCount()
{
#if defined(__linux__)
    cpu_set_t cpuSet;
    CPU_ZERO(&cpuSet);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuSet) == 0) {
        return CPU_COUNT(&cpuSet);
    }
#endif
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return (nprocs > 0) ? static_cast<int>(nprocs) : 24;
}

static int CalcSchedIdxForAicore(int aicoreIdx, int nrValidAic, int scheCpuNum)
{
    bool isAic = (aicoreIdx < nrValidAic);
    int blockdim = nrValidAic;

    if (isAic) {
        int aicPerSche = blockdim / scheCpuNum;
        if (aicPerSche == 0) {
            aicPerSche = 1;
        }
        int schedIdx = aicoreIdx / aicPerSche;
        if (schedIdx >= scheCpuNum) {
            schedIdx = scheCpuNum - 1;
        }
        return schedIdx;
    } else {
        int aivGlobalIdx = aicoreIdx - blockdim;
        int aivPerSche = (2 * blockdim) / scheCpuNum;
        if (aivPerSche == 0) {
            aivPerSche = 1;
        }
        int schedIdx = aivGlobalIdx / aivPerSche;
        if (schedIdx >= scheCpuNum) {
            schedIdx = scheCpuNum - 1;
        }
        return schedIdx;
    }
}

static int CalcBindCpuForAicore(int aicoreIdx, int nrValidAic, int totalCpuCores, int scheCpuNum)
{
    (void)nrValidAic;
    (void)scheCpuNum;
    int cpuBase = (totalCpuCores >= 36) ? AICORE_THREAD_CPU_BASE_36CORE : AICORE_THREAD_CPU_BASE_24CORE;
    int bindCoreNum = totalCpuCores - cpuBase;
    if (bindCoreNum <= 0) {
        bindCoreNum = totalCpuCores;
        cpuBase = 0;
    }
    int hostCpu = cpuBase + (aicoreIdx % bindCoreNum);
    return hostCpu;
}

static void SetThreadAffinity(int cpuId, const char* threadName)
{
#if defined(__linux__)
    if (cpuId < 0) {
        return;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuId, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (threadName != nullptr) {
        pthread_setname_np(pthread_self(), threadName);
    }
#else
    (void)cpuId;
    (void)threadName;
#endif
}

static void DumpE2EHostSimBindingPlan(int nrValidAic, int scheCpuNum, int totalCpuCores, bool enableBind)
{
    int blockdim = nrValidAic;
    if (blockdim <= 0 || scheCpuNum <= 0 || totalCpuCores <= 0) {
        MACHINE_LOGI(
            "[E2E_HOST_SIM][bind_plan] invalid params: blockdim=%d scheCpuNum=%d totalCpuCores=%d enableBind=%d",
            blockdim, scheCpuNum, totalCpuCores, enableBind);
        return;
    }

    int cpuBase = (totalCpuCores >= 36) ? AICORE_THREAD_CPU_BASE_36CORE : AICORE_THREAD_CPU_BASE_24CORE;
    int bindCoreNum = totalCpuCores - cpuBase;
    if (bindCoreNum <= 0) {
        bindCoreNum = totalCpuCores;
        cpuBase = 0;
    }

    int aicPerSche = blockdim / scheCpuNum;
    if (aicPerSche <= 0) {
        aicPerSche = 1;
    }
    int aivPerSche = (2 * blockdim) / scheCpuNum;
    if (aivPerSche <= 0) {
        aivPerSche = 1;
    }

    const int aicoreNum = 3 * blockdim;
    MACHINE_LOGI(
        "[E2E_HOST_SIM][bind_plan] blockdim=%d aicoreNum=%d scheCpuNum=%d enableBind=%d cpuBase=%d bindCoreNum=%d "
        "aicPerSche=%d aivPerSche=%d",
        blockdim, aicoreNum, scheCpuNum, enableBind, cpuBase, bindCoreNum, aicPerSche, aivPerSche);

    for (int s = 0; s < scheCpuNum; ++s) {
        int aicStart = s * aicPerSche;
        int aicEnd = (s == scheCpuNum - 1) ? (blockdim - 1) : ((s + 1) * aicPerSche - 1);
        if (aicEnd >= blockdim) {
            aicEnd = blockdim - 1;
        }
        int aivStart = blockdim + s * aivPerSche;
        int aivEnd = (s == scheCpuNum - 1) ? (3 * blockdim - 1) : (blockdim + (s + 1) * aivPerSche - 1);
        if (aivEnd >= 3 * blockdim) {
            aivEnd = 3 * blockdim - 1;
        }
        MACHINE_LOGI(
            "[E2E_HOST_SIM][bind_plan] sche=%d AIC=[%d,%d] AIV=[%d,%d]", s, aicStart, aicEnd, aivStart, aivEnd);
    }

    std::vector<int> aicCnt(static_cast<size_t>(scheCpuNum), 0);
    std::vector<int> aivCnt(static_cast<size_t>(scheCpuNum), 0);
    for (int i = 0; i < blockdim; ++i) {
        int s = CalcSchedIdxForAicore(i, blockdim, scheCpuNum);
        if (s >= 0 && s < scheCpuNum) {
            aicCnt[static_cast<size_t>(s)]++;
        }
    }
    for (int i = blockdim; i < 3 * blockdim; ++i) {
        int s = CalcSchedIdxForAicore(i, blockdim, scheCpuNum);
        if (s >= 0 && s < scheCpuNum) {
            aivCnt[static_cast<size_t>(s)]++;
        }
    }
    for (int s = 0; s < scheCpuNum; ++s) {
        MACHINE_LOGI(
            "[E2E_HOST_SIM][bind_plan] sche=%d cnt(AIC)=%d cnt(AIV)=%d", s, aicCnt[static_cast<size_t>(s)],
            aivCnt[static_cast<size_t>(s)]);
    }

    const int headN = std::min(aicoreNum, 12);
    for (int i = 0; i < headN; ++i) {
        int s = CalcSchedIdxForAicore(i, blockdim, scheCpuNum);
        int cpu = enableBind ? CalcBindCpuForAicore(i, blockdim, totalCpuCores, scheCpuNum) : -1;
        (void)cpu;
    }
    if (aicoreNum > headN) {
        for (int i = std::max(0, aicoreNum - 6); i < aicoreNum; ++i) {
            int s = CalcSchedIdxForAicore(i, blockdim, scheCpuNum);
            int cpu = enableBind ? CalcBindCpuForAicore(i, blockdim, totalCpuCores, scheCpuNum) : -1;
            (void)s;
            (void)cpu;
        }
    }
}

class ScopedHostSimClockTicker {
public:
    explicit ScopedHostSimClockTicker(std::chrono::microseconds sleepInterval) : sleepInterval_(sleepInterval)
    {
        worker_ = std::thread([this]() {
            auto last = std::chrono::steady_clock::now();
            while (!stop_.load(std::memory_order_relaxed)) {
                auto now = std::chrono::steady_clock::now();
                auto delta = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count());
                if (delta > 0) {
                    HostSimClock::AdvanceNs(delta);
                }
                last = now;
                std::this_thread::sleep_for(sleepInterval_);
            }
        });
    }

    ~ScopedHostSimClockTicker()
    {
        stop_.store(true, std::memory_order_relaxed);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    ScopedHostSimClockTicker(const ScopedHostSimClockTicker&) = delete;
    ScopedHostSimClockTicker& operator=(const ScopedHostSimClockTicker&) = delete;

private:
    std::chrono::microseconds sleepInterval_;
    std::atomic<bool> stop_{false};
    std::thread worker_;
};

static void InitHostSimRegAddrArray(DevAscendProgram* devProg, int aicoreNum)
{
    if (devProg == nullptr || aicoreNum <= 0 || devProg->devArgs.coreRegAddr == 0) {
        return;
    }
    auto* coreRegAddr = reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(devProg->devArgs.coreRegAddr));
    if (coreRegAddr == nullptr) {
        return;
    }
    for (int i = 0; i < aicoreNum; ++i) {
        coreRegAddr[i] = HostRegBus::Global().GetMainBaseAddr(static_cast<size_t>(i));
    }
}

static void NotifyAicoreStop(const DevAscendProgram* devProg, int aicoreNum)
{
    if (devProg == nullptr || devProg->devArgs.sharedBuffer == 0 || aicoreNum <= 0) {
        return;
    }
    std::uint8_t* sharedBufferBase =
        reinterpret_cast<std::uint8_t*>(static_cast<uintptr_t>(devProg->devArgs.sharedBuffer));
    for (int i = 0; i < aicoreNum; ++i) {
        // Keep protocol semantics: send stop command through mainBase.
        HostRegBus::Global().WriteMainBase(static_cast<size_t>(i), static_cast<std::uint64_t>(AICORE_TASK_STOP) + 1ULL);

        // Wake up waiting core and ask it to exit loop.
        auto* args = reinterpret_cast<KernelArgs*>(sharedBufferBase + static_cast<size_t>(i) * SHARED_BUFFER_SIZE);
        args->waveBufferCpuToCore[CPU_TO_CORE_SHAK_BUF_GOODBYE_INDEX] = AICORE_SAY_GOODBYE;
    }
}

static void E2EAicoreWorker(DeviceKernelArgs kArgs, int blockId, int phyId, int bindCpu, int nrValidAic, int scheCpuNum)
{
    char name[32];
    bool isAic = (blockId < nrValidAic);
    const char* coreType = isAic ? "aic" : "aiv";
    int schedIdx = CalcSchedIdxForAicore(blockId, nrValidAic, scheCpuNum);
    (void)sprintf_s(name, sizeof(name), "e2e_%s_s%d_%d", coreType, schedIdx, blockId);
    SetThreadAffinity(bindCpu, name);
    HostCoreContext ctx;
    ctx.blockId = blockId;
    ctx.phyId = phyId;
    HostCoreCtx::SetCurrent(ctx);
    KernelEntry(0, 0, 0, 0, 0, reinterpret_cast<int64_t>(kArgs.cfgdata));
}

static int E2EHostSimLaunchOnce(DeviceKernelArgs& kArgs)
{
    constexpr int threadNum = MAX_LAUNCH_SCHEDULE_AICPU_NUM + static_cast<int>(dynamic::MAX_CONTROL_FLOW_AICPU_NUM);
    std::thread aicpuThreadList[threadNum];
    int aicpuResultList[threadNum] = {0};
    std::atomic<int> idx{0};
    auto* devProg = (DevAscendProgram*)(kArgs.cfgdata);

    devProg->devArgs.nrAic = E2E_HOST_SIM_NR_AIC;
    devProg->devArgs.nrAiv = E2E_HOST_SIM_NR_AIV;
    devProg->devArgs.nrValidAic = E2E_HOST_SIM_NR_VALID_AIC;
    devProg->devArgs.nrAicpu = E2E_HOST_SIM_NR_AICPU;
    devProg->devArgs.scheCpuNum = E2E_HOST_SIM_SCHE_CPU_NUM;

    int aicoreNum = static_cast<int>(devProg->devArgs.nrAic + devProg->devArgs.nrAiv);
    MACHINE_LOGI(
        "[E2E_HOST_SIM] cfgdata=%p sharedBuffer=0x%llx runtimeDataRingBufferAddr=0x%llx "
        "coreRegAddr=0x%llx devDfxArgAddr=0x%llx nrAic=%u nrAiv=%u nrAicpu=%u scheCpuNum=%u",
        kArgs.cfgdata, static_cast<unsigned long long>(devProg->devArgs.sharedBuffer),
        static_cast<unsigned long long>(devProg->devArgs.runtimeDataRingBufferAddr),
        static_cast<unsigned long long>(devProg->devArgs.coreRegAddr),
        static_cast<unsigned long long>(devProg->devArgs.devDfxArgAddr),
        static_cast<unsigned int>(devProg->devArgs.nrAic), static_cast<unsigned int>(devProg->devArgs.nrAiv),
        static_cast<unsigned int>(devProg->devArgs.nrAicpu), static_cast<unsigned int>(devProg->devArgs.scheCpuNum));
    std::vector<std::thread> aicoreThreadList;
    aicoreThreadList.reserve(static_cast<size_t>(aicoreNum));
    HostSimClock::Reset(0);
    ScopedHostSimClockTicker clockTicker(std::chrono::microseconds(100));
    HostRegBus::Global().Reset(static_cast<size_t>(aicoreNum));
    InitHostSimRegAddrArray(devProg, aicoreNum);
    size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
    auto deviceTaskCtrlPoolAddr = devProg->GetRuntimeDataList()->GetRuntimeData() + DEV_ARGS_SIZE;
    (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
    devProg->devArgs.aicpuPerfAddr = 0UL;
    int launchAiCpuNum = static_cast<int>(devProg->devArgs.nrAicpu + dynamic::MAX_CONTROL_FLOW_AICPU_NUM);
    int nrValidAic = static_cast<int>(devProg->devArgs.nrValidAic);
    if (nrValidAic <= 0) {
        nrValidAic = static_cast<int>(devProg->devArgs.nrAic);
    }
    int totalCpuCores = GetSystemCpuCount();
    bool enableBind = (totalCpuCores >= 16);
    MACHINE_LOGI(
        "[E2E_HOST_SIM] Binding config: totalCpuCores=%d, nrValidAic=%d, scheCpuNum=%u, enableBind=%d", totalCpuCores,
        nrValidAic, devProg->devArgs.scheCpuNum, enableBind);
    DumpE2EHostSimBindingPlan(nrValidAic, static_cast<int>(devProg->devArgs.scheCpuNum), totalCpuCores, enableBind);

    auto threadFun = [&](int threadIndex, uint32_t runMode, int bindCpu) {
        char name[64];
        const char* role = (runMode == RUN_SPLITTED_STREAM_CTRL) ? "ctrl" : "sche";
        int schedIdx = (runMode == RUN_SPLITTED_STREAM_CTRL) ? 0 : threadIndex;
        (void)sprintf_s(name, sizeof(name), "e2e_%s_%d", role, schedIdx);
        SetThreadAffinity(bindCpu, name);
        MACHINE_LOGD(
            "[E2E_HOST_SIM] AICPU thread start: threadIndex=%d, bindCpu=%d, runMode=%u, name=%s", threadIndex, bindCpu,
            runMode, name);
        DeviceKernelArgs localArgs = kArgs;
        localArgs.parameter.runMode = runMode;
        aicpuResultList[threadIndex] = DynTileFwkBackendKernelServer(&localArgs);
    };

    int ctrlBindCpu = enableBind ? CTRL_THREAD_CPU_BASE : -1;
    aicpuThreadList[0] = std::thread(threadFun, 0, RUN_SPLITTED_STREAM_CTRL, ctrlBindCpu);

    for (int i = 1; i < launchAiCpuNum; i++) {
        int scheBindCpu = enableBind ? (SCHE_THREAD_CPU_BASE + (i - 1) % SCHE_CPU_NUM) : -1;
        aicpuThreadList[i] = std::thread(threadFun, i, RUN_SPLITTED_STREAM_SCHE, scheBindCpu);
    }

    int scheCpuNum = static_cast<int>(devProg->devArgs.scheCpuNum);
    for (int i = 0; i < aicoreNum; ++i) {
        int bindCpu = enableBind ? CalcBindCpuForAicore(i, nrValidAic, totalCpuCores, scheCpuNum) : -1;
        aicoreThreadList.emplace_back(E2EAicoreWorker, kArgs, i, i, bindCpu, nrValidAic, scheCpuNum);
    }

    for (int i = 0; i < threadNum; i++) {
        if (aicpuThreadList[i].joinable()) {
            aicpuThreadList[i].join();
        }
    }

    // Explicit stop protocol for host-sim aicore workers, instead of relying on timeout.
    NotifyAicoreStop(devProg, aicoreNum);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    for (auto& t : aicoreThreadList) {
        if (t.joinable()) {
            t.join();
        }
    }
    for (int i = 0; i < threadNum; i++) {
        if (aicpuResultList[i] != 0) {
            return aicpuResultList[i];
        }
    }
    return 0;
}

int E2EHostSimLauncher::E2ELaunchOnceWithHostTensorData(
    Function* function, const std::vector<DeviceTensorData>& inputList, const std::vector<DeviceTensorData>& outputList,
    DevControlFlowCache* ctrlCache, E2EHostSimMemoryUtils& memUtils, const DeviceLauncherConfig& config)
{
    MACHINE_LOGI("!!! E2E Launch Host Sim\n");
    DeviceKernelArgs kArgs;
    auto dynAttr = function->GetDyndevAttribute();
    DeviceLauncher::DeviceInitDistributedContext(memUtils, dynAttr->commGroupNames, kArgs);
    DeviceLauncher::DeviceInitTilingData(memUtils, kArgs, dynAttr->devProgBinary, ctrlCache, config, nullptr);
    DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs, inputList, outputList, dynAttr->disableL2List);
    int rc = E2EHostSimLaunchOnce(kArgs);
    return rc;
}
int E2EHostSimLauncher::E2EHostSimRunOnce(
    Function* function, DevControlFlowCache* inputCtrlCache, const DeviceLauncherConfig& config)
{
    auto& inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto& outputDataList = ProgramData::GetInstance().GetOutputDataList();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    E2EHostSimMemoryUtils memUtils;
    std::tie(inputDeviceDataList, outputDeviceDataList) =
        DeviceLauncher::BuildInputOutputFromHost(memUtils, inputDataList, outputDataList);
    DevControlFlowCache* launchCtrlFlowCache = nullptr;
    if (inputCtrlCache != nullptr) {
        launchCtrlFlowCache =
            reinterpret_cast<DevControlFlowCache*>(memUtils.AllocZero(inputCtrlCache->usedCacheSize, nullptr));
        if (launchCtrlFlowCache) {
            memcpy_s(launchCtrlFlowCache, inputCtrlCache->usedCacheSize, inputCtrlCache, inputCtrlCache->usedCacheSize);
        }
    }
    int rc = E2ELaunchOnceWithHostTensorData(
        function, inputDeviceDataList, outputDeviceDataList, launchCtrlFlowCache, memUtils, config);
    return rc;
}

static std::vector<DeviceTensorData> toHostTensorData(const std::vector<DeviceTensorData>& devDataList, bool isInput)
{
    std::vector<DeviceTensorData> hostDataList;
    for (auto& devData : devDataList) {
        auto size = devData.GetDataSize();
        void* ptr = malloc(size);
        if (isInput) {
#ifdef BUILD_WITH_CANN
            rtMemcpy(ptr, size, devData.GetAddr(), size, RT_MEMCPY_DEVICE_TO_HOST);
#endif
        }
        hostDataList.emplace_back(devData.GetDataType(), ptr, devData.GetShape());
    }
    return hostDataList;
}

static void freeHostTensorData(const std::vector<DeviceTensorData>& hostDataList)
{
    for (auto& hostData : hostDataList) {
        free(hostData.GetAddr());
    }
}
int E2EHostSimLauncher::E2EHostSimLaunchDeviceTensorData(
    Function* function, const std::vector<DeviceTensorData>& inDevList, const std::vector<DeviceTensorData>& outDevList,
    const DeviceLauncherConfig& config)
{
    E2EHostSimMemoryUtils memUtils;
    DeviceLauncher::ChangeCaptureModeRelax();
    auto inList = toHostTensorData(inDevList, true);
    auto outList = toHostTensorData(outDevList, false);
    DeviceLauncher::ChangeCaptureModeGlobal();
    int rc = E2ELaunchOnceWithHostTensorData(function, inList, outList, nullptr, memUtils, config);
    freeHostTensorData(inList);
    freeHostTensorData(outList);
    return rc;
}
} // namespace npu::tile_fwk::dynamic
