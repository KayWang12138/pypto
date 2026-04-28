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
 * \file device_runner.cpp
 * \brief
 */

#include "machine/runtime/device_runner.h"
#include <cstdint>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <limits.h>
#include "securec.h"
#include "machine/runtime/runtime.h"
#include "machine/runtime/device_launcher.h"
#include "machine/runtime/load_aicpu_op.h"
#include "machine/utils/machine_ws_intf.h"
#include "machine/device/dynamic/device_common.h"
#include "interface/utils/file_utils.h"
#include "machine/utils/device_switch.h"
#include "interface/utils/common.h"
#include "interface/configs/config_manager.h"
#include "interface/utils/op_info_manager.h"
#include "load_aicpu_op.h"
#include "tilefwk/platform.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "machine/platform/platform_manager.h"
#include "machine/runtime/device_error_tracking.h"
#include "nlohmann/json.hpp"
#include "dump_device_perf.h"
#include "machine/host/perf_analysis.h"
#include "tilefwk/pypto_fwk_log.h"
#include "interface/machine/host/host_machine.h"
#include "adapter/api/msprof_api.h"
#include "adapter/api/acl_api.h"
#include "adapter/api/runtime_api.h"

using json = nlohmann::json;

constexpr int32_t AICORE_ADDR_TYPE = 2; // nocache Addr type for aicore/aicpu map
constexpr int32_t PMU_ADDR_TYPE = 3;    // nGnRnE Addr type for Geting pmuInfo
constexpr int32_t PATH_LENGTH = 64;
constexpr uint32_t LOG_BUF_SIZE = 64 * 1024;
bool g_IsNullLaunched = false;
bool g_is_machine_trace_addr_inited = false;
constexpr uint32_t MIX_BLOCK_DIM = 2;
constexpr uint32_t HIGHT_BIT = 16;

constexpr uint32_t SUB_CORE = 3;
constexpr uint32_t AIV_PER_AICORE = 2;

extern "C" {
__attribute__((weak)) int AdxDataDumpServerUnInit();
__attribute__((weak)) int dlog_getlevel(int32_t moduled, int32_t* enableEvent);
__attribute__((weak)) int drvDeviceGetPhyIdByIndex(uint32_t logicDevId, uint32_t* phyDevId);
}
namespace npu::tile_fwk {

namespace {

void ExchangeCaputerMode(const bool& isCapture)
{
    if (isCapture) {
        AclMdlRICaptureMode mode = AclMdlRICaptureMode::GLOBAL;
        AclMdlRICaptureThreadExchangeMode(&mode);
        MACHINE_LOGI("captureMode is: %d", static_cast<int>(mode));
    }
}

void* MachinePerfTraceDevMalloc(int size)
{
    uint8_t* devPtr = nullptr;
    auto alignSize = MemSizeAlign(size);
    if (RuntimeMalloc(reinterpret_cast<void**>(&devPtr), alignSize, TWO_MB_HUGE_PAGE_FLAGS, 0) != 0) {
        MACHINE_LOGW("Mem alloc failed");
        return nullptr;
    }
    return devPtr;
}

} // namespace

DeviceRunner& DeviceRunner::Get()
{
    static DeviceRunner runner;
    std::call_once(runner.once_, [&]() { runner.Init(); });
    return runner;
}

HostProf& DeviceRunner::GetHostProfInstance() { return profiler_.GetHostProf(); }

void* DeviceRunner::DevAlloc(int size)
{
    uint8_t* devPtr = nullptr;
    machine::GetRA()->AllocDevAddr(&devPtr, size);
    int rc = RuntimeMemset(devPtr, size, 0, size);
    if (rc != 0) {
        machine::GetRA()->FreeTensor(devPtr);
        MACHINE_LOGE(RtErr::RT_MEMSET_FAILED, "RuntimeMemset failed size=%d rc=%d\n", size, rc);
        return nullptr;
    }
    return devPtr;
}

void DeviceRunner::GetModuleLogLevel(DeviceArgs& args)
{
    int logLevel = -1;
    if (dlog_getlevel != nullptr) {
        int32_t enableLog = -1;
        logLevel = dlog_getlevel(PYPTO, &enableLog);
    }
    DevDfxArgs devDfxArg;
    devDfxArg.logLevel = logLevel;
    uint32_t logicalDevId = GetLogDeviceId();
    uint32_t phyDevId = 0;
    if (drvDeviceGetPhyIdByIndex != nullptr) {
        drvDeviceGetPhyIdByIndex(logicalDevId, &phyDevId);
    } else {
        MACHINE_LOGW("Get device Local deviceId failed");
    }
    MACHINE_LOGI("Current device info: logical devId: %u, phyDevId: %u", logicalDevId, phyDevId);
    devDfxArg.deviceId = phyDevId;
    if (profiler_.GetEnableDumpDevPref()) {
        devDfxArg.isOpenPerfTrace = 1;
    }
    MACHINE_LOGI("Get PYPTO dfxAddr: %lu log level is: %d, openPerTrace: %d, deviceId: %u\n",
        state_.args.devDfxArgAddr, logLevel, devDfxArg.isOpenPerfTrace, devDfxArg.deviceId);
    auto size = sizeof(DevDfxArgs);
    auto ret = RuntimeMemcpy(reinterpret_cast<void*>(args.devDfxArgAddr), size, &devDfxArg, size, RtMemcpyKind::HOST_TO_DEVICE);
    if (ret != 0) {
        MACHINE_LOGW("rtmemcpy failed, so couldn't get device log");
    }
}

void DeviceRunner::InitDynamicArgs(DeviceArgs& args)
{
    state_.devArgsAddr = reinterpret_cast<uint64_t>(DevAlloc(sizeof(DeviceArgs)));
    RuntimeMemcpy(
        reinterpret_cast<void*>(state_.GetDevArgsPtr()), sizeof(DeviceArgs), &args, sizeof(DeviceArgs), RtMemcpyKind::HOST_TO_DEVICE);

    for (uint64_t i = 0; i < args.nrAic + args.nrAiv + AICPU_NUM_OF_RUN_AICPU_TASKS; i++) {
        perfData_.push_back(MachinePerfTraceDevMalloc(MAX_DFX_TASK_NUM_PER_CORE * sizeof(TaskStat) + sizeof(Metrics)));
    }

    if (GetEnvVar("DUMP_DEVICE_PERF") == "true") {
        auto aicpuDevPtr = MachinePerfTraceDevMalloc(MAX_ROUND_NUM * sizeof(MetricPerf));
        if (aicpuDevPtr == 0) {
            MACHINE_LOGW("Aicpu per addr malloc failed");
            return;
        }
        state_.args.aicpuPerfAddr = npu::tile_fwk::dynamic::PtrToValue(aicpuDevPtr);
        profiler_.SetEnableDump(true);
    }
}

void DeviceRunner::ResetPerData()
{
    profiler_.ResetPerData(state_.args, perfData_);
}

void DeviceRunner::InitMetaData(DeviceArgs& targetArgs) const
{
    targetArgs.runtimeDataRingBufferAddr = state_.args.runtimeDataRingBufferAddr;
    targetArgs.sharedBuffer = state_.args.sharedBuffer;
    targetArgs.coreRegAddr = state_.args.coreRegAddr;
    targetArgs.nrAic = state_.args.nrAic;
    targetArgs.nrAiv = state_.args.nrAiv;
    targetArgs.corePmuRegAddr = state_.args.corePmuRegAddr;
    targetArgs.corePmuAddr = state_.args.corePmuAddr;
    targetArgs.taskWastTime = state_.args.taskWastTime;
    targetArgs.pmuEventAddr = state_.args.pmuEventAddr;
    targetArgs.aicpuPerfAddr = state_.args.aicpuPerfAddr;
    targetArgs.devDfxArgAddr = state_.args.devDfxArgAddr;
}

int DeviceRunner::InitDeviceArgsCore(
    DeviceArgs& args, const std::vector<int64_t>& regs, const std::vector<int64_t>& regsPmu)
{
    uint32_t totalCoreCount = regs.size();
    uint32_t aicCount = totalCoreCount / SUB_CORE;
    uint32_t aivCount = aicCount * AIV_PER_AICORE;
    args.nrAic = aicCount;
    args.nrAiv = aivCount;
    state_.blockDim = dynamic::GetCfgBlockdim();
    args.nrValidAic = state_.blockDim;
    args.nrAicpu = state_.aicpuNum;
    args.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(state_.blockDim, state_.aicpuNum, args.archInfo);
    int nrCore = regs.size() + AICPU_NUM_OF_RUN_AICPU_TASKS;
    args.sharedBuffer = reinterpret_cast<uint64_t>(DevAlloc(nrCore * SHARED_BUFFER_SIZE));
    args.coreRegAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * sizeof(uint64_t)));
    args.corePmuRegAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * sizeof(uint64_t)));
    args.corePmuAddr = reinterpret_cast<uint64_t>(DevAlloc(nrCore * PMU_BUFFER_SIZE));
    args.taskWastTime = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DevAlloc(sizeof(uint64_t))));
    size_t shmSize = sizeof(dynamic::RuntimeDataRingBufferHead) + dynamic::DEVICE_SHM_SIZE +
                     dynamic::DEVICE_TASK_QUEUE_SIZE * state_.aicpuNum;
    uint64_t shmAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(DevAlloc(shmSize)));
    args.runtimeDataRingBufferAddr = shmAddr;
    std::vector<int64_t> pmuEvtType;
    PmuCommon::InitPmuEventType(args.archInfo, pmuEvtType);
    args.pmuEventAddr = reinterpret_cast<uint64_t>(DevAlloc(pmuEvtType.size() * sizeof(int64_t)));
    args.devDfxArgAddr = reinterpret_cast<uint64_t>(DevAlloc(sizeof(DevDfxArgs)));

    if (args.devDfxArgAddr == 0) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Alloc devDfx info failed");
        return -1;
    }

    if (args.sharedBuffer == 0 || args.coreRegAddr == 0 || args.corePmuAddr == 0 || args.corePmuRegAddr == 0) {
        return -1;
    }
    size_t size = nrCore * sizeof(uint64_t);
    RuntimeMemcpy(reinterpret_cast<void*>(args.coreRegAddr), size, regs.data(), size, RtMemcpyKind::HOST_TO_DEVICE);
    RuntimeMemcpy(reinterpret_cast<void*>(args.corePmuRegAddr), size, regsPmu.data(), size,
                  RtMemcpyKind::HOST_TO_DEVICE);
    size = pmuEvtType.size() * sizeof(int64_t);
    RuntimeMemcpy(reinterpret_cast<void*>(args.pmuEventAddr), size, pmuEvtType.data(), size,
                  RtMemcpyKind::HOST_TO_DEVICE);
    MACHINE_LOGI(
        "aic %u aiv %u  state_.blockDim %d sharedBuffer %lx coreRegAddr %lx corePmuRegAddr %lx\n", args.nrAic, args.nrAiv,
        state_.blockDim, args.sharedBuffer, args.coreRegAddr, args.corePmuRegAddr);
    InitDynamicArgs(args);
    GetModuleLogLevel(args);
    return 0;
}

int DeviceRunner::InitDeviceArgs()
{
    profiler_.GetHostProf().RegHostProf();

    addressMappingTable_[ArchInfo::DAV_2201] = [this](std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu) {
        std::vector<int64_t> aiv;
        std::vector<int64_t> aic;
        std::vector<int64_t> aivPmu;
        std::vector<int64_t> aicPmu;
        if (machine::GetRA()->GetAicoreRegInfo(aic, aiv, ADDR_MAP_TYPE_REG_AIC_CTRL) != 0) {
            return -1;
        }
        if (machine::GetRA()->GetAicoreRegInfo(aicPmu, aivPmu, ADDR_MAP_TYPE_REG_AIC_PMU_CTRL) != 0) {
            return 0;
        }
        regs.insert(regs.end(), aic.begin(), aic.end());
        regs.insert(regs.end(), aiv.begin(), aiv.end());
        regsPmu.insert(regsPmu.end(), aicPmu.begin(), aicPmu.end());
        regsPmu.insert(regsPmu.end(), aivPmu.begin(), aivPmu.end());
        return 0;
    };

    addressMappingTable_[ArchInfo::DAV_3510] = [](std::vector<int64_t>& regs, std::vector<int64_t>& regsPmu) {
        return machine::GetRA()->GetAicoreRegInfoForDAV3510(regs, regsPmu);
    };

    memset_s(&state_.args, sizeof(state_.args), 0, sizeof(state_.args));
    std::vector<int64_t> regs;
    std::vector<int64_t> regsPmu;

    state_.args.archInfo = static_cast<ArchInfo>(Platform::Instance().GetSoc().GetNPUArch());
    if (state_.args.archInfo == ArchInfo::DAV_3510) {
        state_.aicpuNum = npu::tile_fwk::dynamic::DEVICE_MAX_AICPU_NUM;
    }
    int cpuNum = static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum());
    state_.args.maxAicpuNum = cpuNum;
    state_.aicpuNum = state_.aicpuNum < cpuNum ? state_.aicpuNum : cpuNum;
    auto it = addressMappingTable_.find(state_.args.archInfo);
    if (it != addressMappingTable_.end()) {
        if (it->second(regs, regsPmu) != 0) {
            return -1;
        }
    }
    kernelLaunchManager_.InitAiCpuSoBin(state_.args);
    return InitDeviceArgsCore(state_.args, regs, regsPmu);
}

uint64_t DeviceRunner::GetTasksTime() const
{
    uint64_t buffer;
    int rc = RuntimeMemcpy(
        reinterpret_cast<void*>(&buffer), sizeof(uint64_t),
        reinterpret_cast<void*>(static_cast<uintptr_t>(state_.args.taskWastTime)), sizeof(uint64_t),
        RtMemcpyKind::DEVICE_TO_HOST);
    (void)rc;
    return buffer;
}

bool DeviceRunner::GetValidGetPgMask() const { return machine::GetRA()->GetValidGetPgMask(); }

void DeviceRunner::AllocDfxMetricMemory()
{
    profiler_.AllocDfxMetricMemory(state_.args, perfData_);
}

void DeviceRunner::Dump()
{
    profiler_.Dump(state_.args, perfData_);
}

/**************************** DynamicFunction *****************************/
void DeviceRunner::DumpAiCoreExecutionTimeData()
{
    profiler_.DumpAiCoreExecutionTimeData(state_.args, perfData_);
}

void DeviceRunner::DumpAiCorePmuData() { profiler_.DumpAiCorePmuData(); }

void DeviceRunner::SynchronizeDeviceToHostProfData()
{
    profiler_.SynchronizeDeviceToHostProfData(state_.args, perfData_);
}

int DeviceRunner::DynamicLaunchSynchronize(RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream)
{
    int rc = streamSync_.Synchronize(aicpuStream, ctrlStream, aicoreStream);
    if (IsPtoDataDumpEnabled()) {
        MACHINE_LOGD("DataDumpServerInit is called \n");
        AdxDataDumpServerUnInit();
    }
    return rc;
}

bool DeviceRunner::GetEnableDumpDevPref() const { return profiler_.GetEnableDumpDevPref(); }

void DeviceRunner::ResetMetrics(const uint32_t& coreId)
{
    profiler_.ResetMetrics(coreId, perfData_);
}

void DeviceRunner::SetDebugEnable()
{
    profiler_.SetDebugEnable(state_.args, perfData_);
}

int DeviceRunner::RunPrepare()
{
    return profiler_.RunPrepare(state_.args, perfData_);
}

int DeviceRunner::RunPreSync(RtStream scheStream, RtStream ctrlStream, RtStream aicoreStream)
{
    return streamSync_.PreSync(scheStream, ctrlStream, aicoreStream);
}

int DeviceRunner::DynamicKernelLaunch(
    RtStream aicpuStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs, int blockdim)
{
    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuInit);
    uint64_t startTime = MspfSysCycleTime();
    auto rc = kernelLaunchManager_.LaunchAiCpu(aicpuStream, kernelArgs, state_.aicpuNum);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(aicpuStream, startTime, state_.aicpuNum, MSPF_GE_TASK_TYPE_AI_CPU);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAicpuRun);

    startTime = MspfSysCycleTime();
    rc = kernelLaunchManager_.LaunchAiCore(aicoreStream, kernelArgs, blockdim);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "launch aicpu failed %d\n", rc);
        return rc;
    }
    ReportHostProfInfo(aicoreStream, startTime, blockdim, MSPF_GE_TASK_TYPE_MIX_AIC, true);

    HOST_PERF_TRACE(TracePhase::RunDevKernelLaunchAIcore);
    return rc;
}

int DeviceRunner::DynamicTripleStreamLaunch(
    RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream, DeviceKernelArgs* kernelArgs, int blockdim)
{
    uint64_t startTime = MspfSysCycleTime();
    int rc = kernelLaunchManager_.LaunchTripleStream(schedStream, ctrlStream, aicoreStream, kernelArgs, blockdim, state_.aicpuNum);
    if (rc < 0) {
        return rc;
    }
    // Note: profiling and post-sync handled by caller or in DynamicLaunch
    rc = streamSync_.PostSync(ctrlStream, aicoreStream);
    return rc;
}

int DeviceRunner::DynamicLaunch(
    RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, [[maybe_unused]] int64_t taskId,
    DeviceKernelArgs* kernelArgs, int blockdim, int launchAicpuNum)
{
#ifdef BUILD_WITH_NEW_CANN
    if (!g_IsNullLaunched) {
        auto ret = LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kernelArgs, 1, "PyptoNull");
        if (ret != 0) {
            MACHINE_LOGE(HostLauncherErr::LAUNCH_BUILTIN_OP_NULL_FAILED, "launch built null failed");
            return ret;
        }
        g_IsNullLaunched = true;
    }
#endif
    int rc = RunPrepare();
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_PREPARE_FAILED, "Prepare failed.");
        return rc;
    }
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitRunPrepare);

    state_.blockDim = blockdim;
    state_.aicpuNum = launchAicpuNum;
    // for dump perfInfo update device args
    state_.args.nrValidAic = blockdim;
    state_.args.nrAicpu = launchAicpuNum;
    state_.args.scheCpuNum = dynamic::CalcSchAicpuNumByBlockDim(state_.blockDim, state_.aicpuNum, state_.args.archInfo);

    ExchangeCaputerMode(state_.isCapture);
    if (ctrlStream == nullptr) {
        return DynamicKernelLaunch(aicpuStream, aicoreStream, kernelArgs, state_.blockDim);
    }
    return DynamicTripleStreamLaunch(aicpuStream, ctrlStream, aicoreStream, kernelArgs, state_.blockDim);
}

void DeviceRunner::ReportHostProfInfo(
    RtStream stream, uint64_t startTime, uint32_t blockDim, uint16_t taskType, bool isCore)
{
    profiler_.ReportProfilingInfo(stream, startTime, blockDim, taskType, isCore);
}

int DeviceRunner::DynamicRun(
    RtStream aicpuStream, RtStream ctrlStream, RtStream aicoreStream, int64_t taskId,
    DeviceKernelArgs* kernelArgs, int blockdim, int launchAicpuNum)
{
    int rc = DynamicLaunch(aicpuStream, ctrlStream, aicoreStream, taskId, kernelArgs, blockdim, launchAicpuNum);
    if (rc < 0) {
        return rc;
    }
    if (state_.isCapture) {
        return 0;
    }
    return DynamicLaunchSynchronize(aicpuStream, ctrlStream, aicoreStream);
}

/**************************** DynamicFunction *****************************/

void DeviceRunner::SetBinData(const std::vector<uint8_t>& binBuf)
{
    kernelLaunchManager_.SetBinData(binBuf);
}

int DeviceRunner::RegisterKernelBin(void** hdl, std::vector<uint8_t>* funcBinBuf)
{
    return kernelLaunchManager_.RegisterKernelBin(hdl, funcBinBuf);
}

int DeviceRunner::Init(void)
{
    int32_t devId = 0;
    (void)RuntimeGetDevice(&devId);
    char path[PATH_LENGTH];
    sprintf_s(path, PATH_LENGTH, "/tmp/aicpu%d.lock", devId);
    lock_.Init(path);
    std::string builtInOpPath = config::LogTopFolder() + "/built_in";
    CreateMultiLevelDir(builtInOpPath);
    LoadAicpuOp::GetInstance().GenBuiltInOpInfo(builtInOpPath);
    if (LoadAicpuOp::GetInstance().GetBuiltInOpBinHandle() != 0) {
        MACHINE_LOGE(DevCommonErr::GET_HANDLE_FAILED, "Get builtInOp Funchandle failed\n");
        return -1;
    }

    InitializeErrorCallback();

    if (streamSync_.Init() != 0) {
        MACHINE_LOGE(RtErr::RT_EVENT_FAILED, "StreamSynchronizer init failed.");
        return -1;
    }
    if (InitDeviceArgs() != 0) {
        MACHINE_LOGE(HostLauncherErr::PREPARE_ARGS_FAILED, "prepareArgs failed\n");
        return -1;
    }
    void* hdl = kernelLaunchManager_.GetBinHandle();
    if (kernelLaunchManager_.RegisterKernelBin(&hdl) != 0) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "RegisterKernelBin failed\n");
        return -1;
    }
    if (!(config::GetRuntimeOption<int64_t>(CFG_RUN_MODE) == CFG_RUN_MODE_SIM
            && config::GetSimConfig(KEY_ACCURACY_LEVEL, 2) == 2)) {
        kernelLaunchManager_.InitAicpuServer(state_.GetDevArgsPtr());
    }
    profiler_.StartMachinePerfTraceDumpThread(state_.args, perfData_);
    return 0;
}

void DeviceRunner::StartMachinePerfTraceDumpThread()
{
    profiler_.StartMachinePerfTraceDumpThread(state_.args, perfData_);
}

void DeviceRunner::StopMachinePerfTraceDumpThread()
{
    profiler_.StopMachinePerfTraceDumpThread();
}

DeviceRunner::~DeviceRunner()
{
    MACHINE_LOGD("Start to cleanup perfData");
    profiler_.StopMachinePerfTraceDumpThread();
    for (size_t i = 0; i < perfData_.size(); i++) {
        if (perfData_[i] != nullptr) {
            RuntimeFree(perfData_[i]);
            perfData_[i] = nullptr;
        }
    }
    perfData_.clear();
}
} // namespace npu::tile_fwk

