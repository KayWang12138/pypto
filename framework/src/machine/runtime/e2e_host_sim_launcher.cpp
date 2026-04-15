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

#include <chrono>

#include "machine/runtime/e2e_host_sim/host_aicore_entry_adapter.h"
#include "machine/runtime/e2e_host_sim/host_core_context.h"
#include "machine/runtime/e2e_host_sim/host_reg_bus.h"
#include "machine/runtime/e2e_host_sim/host_sim_clock.h"
#include "machine/runtime/dump_device_perf.h"
#include "interface/machine/device/tilefwk/aicore_entry.h"

extern "C" int DynTileFwkBackendKernelServer(void* targ);

namespace npu::tile_fwk::dynamic {

    static int InitHostSimSharedBuffer(E2EHostSimMemoryUtils& memUtils, DevAscendProgram* devProg, int aicoreNum)
    {
        if (devProg == nullptr || aicoreNum <= 0) {
            return 0;
        }

        const uint32_t totalCoreNum = static_cast<uint32_t>(aicoreNum) + AICPU_NUM_OF_RUN_AICPU_TASKS;
        const auto sharedBufferSize = static_cast<size_t>(totalCoreNum) * static_cast<size_t>(SHARED_BUFFER_SIZE);
        if (devProg->devArgs.sharedBuffer == 0) {
            auto* sharedBuffer = memUtils.AllocZero(sharedBufferSize, nullptr);
            if (sharedBuffer == nullptr) {
                MACHINE_LOGE(
                    DevCommonErr::ALLOC_FAILED, "Alloc host sim sharedBuffer failed, size=%llu",
                    static_cast<unsigned long long>(sharedBufferSize));
                return -1;
            }
            devProg->devArgs.sharedBuffer =
                static_cast<decltype(devProg->devArgs.sharedBuffer)>(reinterpret_cast<uintptr_t>(sharedBuffer));
        }

        if (devProg->devArgs.devDfxArgAddr == 0) {
            auto* devDfxArgs = reinterpret_cast<DevDfxArgs*>(memUtils.AllocZero(sizeof(DevDfxArgs), nullptr));
            if (devDfxArgs == nullptr) {
                MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Alloc host sim devDfxArg failed");
                return -1;
            }
            devProg->devArgs.devDfxArgAddr =
                static_cast<decltype(devProg->devArgs.devDfxArgAddr)>(reinterpret_cast<uintptr_t>(devDfxArgs));
        }

        uint8_t* sharedBufferBase = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(devProg->devArgs.sharedBuffer));
        for (uint32_t i = 0; i < totalCoreNum; ++i) {
            KernelArgs* args = reinterpret_cast<KernelArgs*>(sharedBufferBase + static_cast<size_t>(i) * SHARED_BUFFER_SIZE);
            if (args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX] == 0) {
                const auto metricSize =
                    sizeof(Metrics) + static_cast<size_t>(MAX_DFX_TASK_NUM_PER_CORE) * sizeof(TaskStat);
                auto* metric = memUtils.AllocZero(metricSize, nullptr);
                if (metric == nullptr) {
                    MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Alloc host sim metrics failed, core=%u", i);
                    return -1;
                }
                args->shakeBuffer[SHAK_BUF_DFX_DATA_INDEX] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(metric));
            }
#if ENABLE_AICORE_PRINT
            if (args->shakeBuffer[SHAK_BUF_PRINT_BUFFER_INDEX] == 0) {
                auto* printBuffer = memUtils.AllocZero(PRINT_BUFFER_SIZE, nullptr);
                if (printBuffer == nullptr) {
                    MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Alloc host sim print buffer failed, core=%u", i);
                    return -1;
                }
                args->shakeBuffer[SHAK_BUF_PRINT_BUFFER_INDEX] =
                    static_cast<int64_t>(reinterpret_cast<uintptr_t>(printBuffer));
                args->shakeBufferCpuToCore[SHAK_BUF_PRINT_BUFFER_INDEX] =
                    static_cast<int64_t>(reinterpret_cast<uintptr_t>(printBuffer));
            }
#endif
        }

        return 0;
    }

    static void DumpHostSimAicoreStates(const DevAscendProgram* devProg, int aicoreNum, const char* tag)
    {
        if (devProg == nullptr || devProg->devArgs.sharedBuffer == 0 || aicoreNum <= 0) {
            MACHINE_LOGI("[E2E_HOST_SIM][%s] sharedBuffer is not ready", tag);
            return;
        }

        constexpr int kMaxDumpCoreNum = 8;
        int dumpCoreNum = std::min(aicoreNum, kMaxDumpCoreNum);
        const uint8_t* sharedBufferBase =
            reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(devProg->devArgs.sharedBuffer));
        for (int i = 0; i < dumpCoreNum; ++i) {
            const KernelArgs* args =
                reinterpret_cast<const KernelArgs*>(sharedBufferBase + static_cast<size_t>(i) * SHARED_BUFFER_SIZE);
            const unsigned long long hello = static_cast<unsigned long long>(args->shakeBuffer[0]);
            const unsigned long long stage = static_cast<unsigned long long>(args->shakeBuffer[2]);
            const unsigned long long cond =
                static_cast<unsigned long long>(HostRegBus::Global().ReadCond(static_cast<size_t>(i)));
            MACHINE_LOGI(
                "[E2E_HOST_SIM][%s] core=%d hello=0x%llx stage=%llu cond=0x%llx", tag, i,
                static_cast<unsigned long long>(hello), static_cast<unsigned long long>(stage),
                static_cast<unsigned long long>(cond));
        }
    }

    static void ProbeHostSimAicoreStates(const DevAscendProgram* devProg, int aicoreNum)
    {
        DumpHostSimAicoreStates(devProg, aicoreNum, "before_aicore_launch");
        for (int round = 0; round < 10; ++round) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            DumpHostSimAicoreStates(devProg, aicoreNum, "after_aicore_launch");
        }
    }

    static void E2EAicoreWorker(DeviceKernelArgs kArgs, int blockId, int phyId) {
        HostCoreContext ctx;
        ctx.blockId = blockId;
        ctx.phyId = phyId;
        HostCoreCtx::SetCurrent(ctx);
        KernelEntry(0, 0, 0, 0, 0, reinterpret_cast<int64_t>(kArgs.cfgdata));
    }

    static int E2EHostSimLaunchOnce(DeviceKernelArgs &kArgs, E2EHostSimMemoryUtils& memUtils) {
        constexpr int threadNum = MAX_LAUNCH_SCHEDULE_AICPU_NUM + static_cast<int>(dynamic::MAX_CONTROL_FLOW_AICPU_NUM);
        std::thread aicpuThreadList[threadNum];
        int aicpuResultList[threadNum] = {0};
        std::atomic<int> idx{0};
        auto* devProg = (DevAscendProgram*)(kArgs.cfgdata);
        int aicoreNum = static_cast<int>(devProg->devArgs.nrAic + devProg->devArgs.nrAiv);
        std::vector<std::thread> aicoreThreadList;
        aicoreThreadList.reserve(static_cast<size_t>(aicoreNum));
        HostSimClock::Reset(0);
        HostRegBus::Global().Reset(static_cast<size_t>(aicoreNum));
        if (InitHostSimSharedBuffer(memUtils, devProg, aicoreNum) != 0) {
            return -1;
        }
        size_t shmSize = DEVICE_TASK_CTRL_POOL_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
        auto deviceTaskCtrlPoolAddr = devProg->GetRuntimeDataList()->GetRuntimeData() + DEV_ARGS_SIZE;
        (void)memset_s(reinterpret_cast<void*>(deviceTaskCtrlPoolAddr), shmSize, 0, shmSize);
        devProg->devArgs.aicpuPerfAddr = 0UL;
        int launchAiCpuNum = static_cast<int>(devProg->devArgs.nrAicpu + dynamic::MAX_CONTROL_FLOW_AICPU_NUM);

        auto threadFun = [&](int threadIndex, uint32_t runMode) {
            int tidx = idx++;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(tidx, &cpuset);
            char name[64];
            (void)sprintf_s(name, sizeof(name), "aicput%d", tidx);
            MACHINE_LOGD("start thread: %s ", name);
            pthread_setname_np(pthread_self(), name);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            DeviceKernelArgs localArgs = kArgs;
            localArgs.parameter.runMode = runMode;
            aicpuResultList[threadIndex] = DynTileFwkBackendKernelServer(&localArgs);
        };

        aicpuThreadList[0] = std::thread(threadFun, 0, RUN_SPLITTED_STREAM_CTRL);

        for (int i = 1; i < launchAiCpuNum; i++) {
            aicpuThreadList[i] = std::thread(threadFun, i, RUN_SPLITTED_STREAM_SCHE);
        }

        for (int i = 0; i < aicoreNum; ++i) {
            aicoreThreadList.emplace_back(E2EAicoreWorker, kArgs, i, i);
        }

        ProbeHostSimAicoreStates(devProg, aicoreNum);

        for (int i = 0; i < threadNum; i++) {
            if (aicpuThreadList[i].joinable()) {
                aicpuThreadList[i].join();
            }
        }

        DumpHostSimAicoreStates(devProg, aicoreNum, "after_aicpu_join");

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
        Function* function, const std::vector<DeviceTensorData>& inputList,
        const std::vector<DeviceTensorData>& outputList, DevControlFlowCache* ctrlCache, E2EHostSimMemoryUtils& memUtils,
        const DeviceLauncherConfig& config) 
    {
        MACHINE_LOGI("!!! E2E Launch Host Sim\n");
        DeviceKernelArgs kArgs;
        auto dynAttr = function->GetDyndevAttribute();
        DeviceLauncher::DeviceInitDistributedContext(memUtils, dynAttr->commGroupNames, kArgs);
        DeviceLauncher::DeviceInitTilingData(memUtils, kArgs, dynAttr->devProgBinary, ctrlCache, config, nullptr);
        DeviceLauncher::DeviceInitKernelInOuts(memUtils, kArgs, inputList, outputList, dynAttr->disableL2List);
        int rc = E2EHostSimLaunchOnce(kArgs, memUtils);
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
}
