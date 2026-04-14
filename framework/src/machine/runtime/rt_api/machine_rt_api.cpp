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
 * \file machine_rt_api.cpp
 * \brief runtime rt api dispatcher and implementations
 */

#include "machine/runtime/rt_api/machine_rt_api.h"

#include <cstdlib>
#include <string>
#include "securec.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/device_runner.h"
#include "machine/runtime/host_prof.h"
#include "machine/utils/machine_error.h"

namespace npu::tile_fwk::dynamic {

std::atomic<IMachineRtApi*> RtApiDispatcher::currentApi_{nullptr};

namespace {
constexpr int kRtOk = 0;
constexpr int kRtErr = -1;
} // namespace


int RealRtApi::Malloc(void** ptr, size_t size, bool useHugePage)
{
    if (ptr == nullptr || size == 0) {
        return kRtErr;
    }
#ifdef BUILD_WITH_CANN
    if (useHugePage) {
        uint8_t* devPtr = nullptr;
        machine::GetRA()->AllocDevAddr(&devPtr, size);
        *ptr = devPtr;
        return devPtr == nullptr ? kRtErr : kRtOk;
    }
    auto ret = rtMalloc(ptr, size, RT_MEMORY_HBM, 0);
    return ret == RT_ERROR_NONE ? kRtOk : kRtErr;
#else
    (void)useHugePage;
    *ptr = std::malloc(size);
    return *ptr == nullptr ? kRtErr : kRtOk;
#endif
}

int RealRtApi::Free(void* ptr, bool useHugePage)
{
    if (ptr == nullptr) {
        return kRtOk;
    }
#ifdef BUILD_WITH_CANN
    if (useHugePage) {
        machine::GetRA()->FreeDevAddr(reinterpret_cast<uint8_t*>(ptr));
        return kRtOk;
    }
    auto ret = rtFree(ptr);
    return ret == RT_ERROR_NONE ? kRtOk : kRtErr;
#else
    (void)useHugePage;
    std::free(ptr);
    return kRtOk;
#endif
}

int RealRtApi::Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind)
{
    if (dst == nullptr || src == nullptr || dstSize < size) {
        return kRtErr;
    }
#ifdef BUILD_WITH_CANN
    auto ret = rtMemcpy(dst, dstSize, src, size, static_cast<rtMemcpyKind_t>(kind));
    return ret == RT_ERROR_NONE ? kRtOk : kRtErr;
#else
    (void)kind;
    return memcpy_s(dst, dstSize, src, size) == EOK ? kRtOk : kRtErr;
#endif
}

int RealRtApi::Memset(void* dst, size_t dstSize, int value, size_t size)
{
    if (dst == nullptr || dstSize < size) {
        return kRtErr;
    }
#ifdef BUILD_WITH_CANN
    auto ret = rtMemset(dst, dstSize, value, size);
    return ret == RT_ERROR_NONE ? kRtOk : kRtErr;
#else
    return memset_s(dst, dstSize, value, size) == EOK ? kRtOk : kRtErr;
#endif
}

int RealRtApi::LaunchAicpu(void* rtArgs, bool tripleStream, bool debugEnable, Function* function)
{
#ifdef BUILD_WITH_CANN
    auto* argsEx = reinterpret_cast<rtAicpuArgsEx_t*>(rtArgs);
    if (argsEx == nullptr || function == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "RealRtApi::LaunchAicpu invalid args.");
        return kRtErr;
    }

    auto ctrlStream = reinterpret_cast<aclrtStream>(machine::GetRA()->GetCtrlStream());
    auto schedStream = reinterpret_cast<aclrtStream>(machine::GetRA()->GetScheStream());
    auto& devRunner = DeviceRunner::Get();
    devRunner.GetHostProfInstance().SetProfFunction(function);

    auto* aicpuArgs = reinterpret_cast<AiCpuArgs*>(argsEx->args);
    const int nrAicpu = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.nrAicpu);
    int ret = 0;
    if (tripleStream) {
        auto startTime = MsprofSysCycleTime();
        aicpuArgs->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
        ret = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, argsEx, nullptr, ctrlStream,
            RT_KERNEL_USE_SPECIAL_TIMEOUT);
        devRunner.ReportHostProfInfo(ctrlStream, startTime, 1, MSPROF_GE_TASK_TYPE_AI_CPU, false);
        if (ret != RT_ERROR_NONE) {
            return ret;
        }

        aicpuArgs->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
        startTime = MsprofSysCycleTime();
        const int scheCpuNum = static_cast<int>(DeviceLauncher::GetDevProg(function)->devArgs.scheCpuNum);
        ret = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", nrAicpu, argsEx, nullptr, schedStream,
            RT_KERNEL_USE_SPECIAL_TIMEOUT);
        devRunner.ReportHostProfInfo(schedStream, startTime, scheCpuNum, MSPROF_GE_TASK_TYPE_AI_CPU, false);
    } else {
        aicpuArgs->kArgs.parameter.runMode = RUN_UNIFIED_STREAM;
        auto startTime = MsprofSysCycleTime();
        ret = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", nrAicpu, argsEx, nullptr, schedStream,
            RT_KERNEL_USE_SPECIAL_TIMEOUT);
        devRunner.ReportHostProfInfo(schedStream, startTime, nrAicpu, MSPROF_GE_TASK_TYPE_AI_CPU, false);
    }
    (void)debugEnable;
    return ret;
#else
    (void)rtArgs;
    (void)tripleStream;
    (void)debugEnable;
    (void)function;
    return kRtOk;
#endif
}

int RealRtApi::LaunchAicore(
    void* aicoreStream, void* kernel, void* rtArgs, void* rtTaskCfg, bool debugEnable, uint32_t blockDim,
    uint64_t tilingKey)
{
#ifdef BUILD_WITH_CANN
    auto& devRunner = DeviceRunner::Get();
    auto stream = reinterpret_cast<aclrtStream>(aicoreStream);
    auto* argsEx = reinterpret_cast<rtArgsEx_t*>(rtArgs);
    auto* taskCfg = reinterpret_cast<rtTaskCfgInfo_t*>(rtTaskCfg);
    if (argsEx == nullptr || taskCfg == nullptr) {
        MACHINE_LOGE(DevCommonErr::NULLPTR, "RealRtApi::LaunchAicore invalid args.");
        return kRtErr;
    }

    auto startTime = MsprofSysCycleTime();
    auto ret = rtKernelLaunchWithHandleV2(kernel, tilingKey, blockDim, argsEx, nullptr, stream, taskCfg);
    devRunner.ReportHostProfInfo(stream, startTime, blockDim, MSPROF_GE_TASK_TYPE_MIX_AIC, true);

    if (debugEnable) {
        auto scheStream = reinterpret_cast<aclrtStream>(machine::GetRA()->GetScheStream());
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(scheStream, nullptr, stream);
        if (rc != 0) {
            MACHINE_LOGE(HostLauncherErr::SYNC_FAILED, "sync failed");
            return rc;
        }
        devRunner.DumpAiCoreExecutionTimeData();
        ASSERT(machine::GetRA()->CheckAllSentinels());
    }

    if (IsPtoDataDumpEnabled()) {
        auto scheStream = reinterpret_cast<aclrtStream>(machine::GetRA()->GetScheStream());
        int rc = DeviceRunner::Get().DynamicLaunchSynchronize(scheStream, nullptr, stream);
        if (rc != 0) {
            MACHINE_LOGE(HostLauncherErr::SYNC_FAILED, "sync failed");
            return rc;
        }
        uint32_t hostPid = GetProcessId();
        std::string sourceDir = "output/dump_tensor_" + std::to_string(hostPid);
        std::string targetDir = config::LogTopFolder() + "/dump_tensor_" + std::to_string(hostPid);
        if (IsPathExist(sourceDir)) {
            std::rename(sourceDir.c_str(), targetDir.c_str());
        }
    }
    return ret;
#else
    (void)aicoreStream;
    (void)kernel;
    (void)rtArgs;
    (void)rtTaskCfg;
    (void)debugEnable;
    (void)blockDim;
    (void)tilingKey;
    return kRtOk;
#endif
}

int RealRtApi::StreamSync(void* aicpuStream, void* ctrlStream, void* aicoreStream)
{
#ifdef BUILD_WITH_CANN
    return DeviceRunner::Get().DynamicLaunchSynchronize(
        reinterpret_cast<rtStream_t>(aicpuStream), reinterpret_cast<rtStream_t>(ctrlStream),
        reinterpret_cast<rtStream_t>(aicoreStream));
#else
    (void)aicpuStream;
    (void)ctrlStream;
    (void)aicoreStream;
    return kRtOk;
#endif
}

HostRtApi::HostRtApi(
    AicpuLaunchFn aicpuLaunch, AicoreLaunchFn aicoreLaunch, StreamSyncFn streamSync, bool enforceNoRealRt)
    : aicpuLaunch_(std::move(aicpuLaunch)), aicoreLaunch_(std::move(aicoreLaunch)),
      streamSync_(std::move(streamSync)), enforceNoRealRt_(enforceNoRealRt)
{}

int HostRtApi::RecordFallbackFailure() const
{
    realRtCallCount_.fetch_add(1, std::memory_order_relaxed);
    return enforceNoRealRt_ ? kRtErr : kRtOk;
}

int HostRtApi::Malloc(void** ptr, size_t size, bool useHugePage)
{
    (void)size;
    (void)useHugePage;
    *ptr = nullptr;
    return 0;
}

int HostRtApi::Free(void* ptr, bool useHugePage)
{
    (void)ptr;
    (void)useHugePage;
    return 0;
}

int HostRtApi::Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind)
{
    (void)dst;
    (void)dstSize;
    (void)src;
    (void)size;
    (void)kind;
    return 0;
}

int HostRtApi::Memset(void* dst, size_t dstSize, int value, size_t size)
{
    (void)dst;
    (void)dstSize;
    (void)value;
    (void)size;
    return 0;
}

int HostRtApi::LaunchAicpu(void* rtArgs, bool tripleStream, bool debugEnable, Function* function)
{
    if (!aicpuLaunch_) {
        return RecordFallbackFailure();
    }
    return aicpuLaunch_(rtArgs, tripleStream, debugEnable, function);
}

int HostRtApi::LaunchAicore(
    void* aicoreStream, void* kernel, void* rtArgs, void* rtTaskCfg, bool debugEnable, uint32_t blockDim,
    uint64_t tilingKey)
{
    if (!aicoreLaunch_) {
        return RecordFallbackFailure();
    }
    return aicoreLaunch_(aicoreStream, kernel, rtArgs, rtTaskCfg, debugEnable, blockDim, tilingKey);
}

int HostRtApi::StreamSync(void* aicpuStream, void* ctrlStream, void* aicoreStream)
{
    if (!streamSync_) {
        return RecordFallbackFailure();
    }
    return streamSync_(aicpuStream, ctrlStream, aicoreStream);
}

RtApiDispatcher::ScopedInstall::ScopedInstall(IMachineRtApi* api) : previous_(RtApiDispatcher::Install(api)) {}

RtApiDispatcher::ScopedInstall::~ScopedInstall() { (void)RtApiDispatcher::Install(previous_); }

IMachineRtApi& RtApiDispatcher::Current()
{
    auto* api = currentApi_.load(std::memory_order_acquire);
    if (api == nullptr) {
        return Real();
    }
    return *api;
}

IMachineRtApi* RtApiDispatcher::Install(IMachineRtApi* api)
{
    return currentApi_.exchange(api, std::memory_order_acq_rel);
}

RealRtApi& RtApiDispatcher::Real()
{
    static RealRtApi realApi;
    return realApi;
}

} // namespace npu::tile_fwk::dynamic
