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
 * \file kernel_launch_manager.cpp
 * \brief Implementation of kernel launch primitives.
 */

#include "machine/runtime/kernel_launch_manager.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"
#include "machine/runtime/runtime.h"
#include "machine/runtime/load_aicpu_op.h"
#include "machine/utils/machine_ws_intf.h"
#include "interface/utils/op_info_manager.h"
#include "interface/utils/file_utils.h"

namespace npu::tile_fwk {

void KernelLaunchManager::SetBinData(const std::vector<uint8_t>& binBuf)
{
    binBuf_ = binBuf;
    MACHINE_LOGD("Set kernel size:%zu", binBuf_.size());
}

int KernelLaunchManager::RegisterKernelBin(void** hdl, std::vector<uint8_t>* funcBinBuf)
{
    if (*hdl) {
        binHdl_ = *hdl;
        MACHINE_LOGD("RegisterKernelBin reuse cache.");
        return 0;
    }
    void* bin = nullptr;
    size_t binSize = 0;
    std::vector<uint8_t>* srcBin = (funcBinBuf == nullptr) ? &binBuf_ : funcBinBuf;
    if (srcBin == nullptr || srcBin->size() == 0) {
        return 0;
    }
    bin = srcBin->data();
    binSize = srcBin->size();
    MACHINE_LOGD("Reg dynamic bin size %zu.", binSize);

    RtDevBinary binary{.magic = RT_DEV_BINARY_MAGIC_ELF, .version = 0, .data = bin, .length = binSize};
    int rc = RuntimeRegisterAllKernel(&binary, hdl);
    if (rc != 0) {
        MACHINE_LOGE(HostLauncherErr::REGISTER_KERNEL_FAILED, "RegisterKernelBin failed\n");
    }
    binHdl_ = *hdl;
    MACHINE_LOGD("finish RegisterKernelBin.");
    return rc;
}

int KernelLaunchManager::LaunchAiCore(RtStream aicoreStream, DeviceKernelArgs* kernelArgs, int blockDim)
{
    RtArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    std::vector<void*> kArgs = {nullptr, nullptr, nullptr, nullptr, nullptr, kernelArgs->cfgdata};
    rtArgs.args = kArgs.data();
    rtArgs.argsSize = kArgs.size() * sizeof(int64_t);
    uint64_t tilingKey = OpInfoManager::GetInstance().GetOpTilingKey();
    RtTaskCfgInfo cfg = {};
    cfg.schemMode = static_cast<uint8_t>(npu::tile_fwk::RtSchemModeType::BATCH);
    return RuntimeKernelLaunchWithHandleV2(binHdl_, tilingKey, blockDim, &rtArgs, nullptr, aicoreStream, &cfg);
}

int KernelLaunchManager::LaunchAiCpu(RtStream aicpuStream, DeviceKernelArgs* kArgs, int aicpuNum)
{
#ifdef BUILD_WITH_NEW_CANN
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, kArgs, aicpuNum, "PyptoRun");
#endif
    auto args = reinterpret_cast<dynamic::AiCpuArgs*>(kArgs->inputs);
    RtAicpuArgsEx rtArgs;
    uint64_t argsSize = reinterpret_cast<uint64_t>(kArgs->outputs);
    kArgs->inputs = nullptr;
    args->kArgs = *kArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = args;
    rtArgs.argsSize = argsSize;
    rtArgs.kernelNameAddrOffset = offsetof(dynamic::AiCpuArgs, kernelName);
    rtArgs.soNameAddrOffset = offsetof(dynamic::AiCpuArgs, soName);
    rtArgs.hostInputInfoNum = 1;
    RtHostInputInfo hostInputInfo;
    hostInputInfo.addrOffset = reinterpret_cast<int8_t*>(&args->kArgs.inputs) - reinterpret_cast<int8_t*>(args);
    hostInputInfo.dataOffset = sizeof(dynamic::AiCpuArgs);
    rtArgs.hostInputInfoPtr = &hostInputInfo;
    rtArgs.timeout = dynamic::AICPU_EXECUTE_TIMEOUT;
    MACHINE_LOGI("Copy flow addrOffset %u argsSize %u", hostInputInfo.addrOffset, hostInputInfo.dataOffset);
    return RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", aicpuNum, &rtArgs, nullptr,
        aicpuStream, RT_KERNEL_USE_SPECIAL_TIMEOUT);
}

int KernelLaunchManager::LaunchKernelPair(RtStream aicpuStream, RtStream aicoreStream,
                                          DeviceKernelArgs* kernelArgs, int blockDim, int aicpuNum)
{
    int rc = LaunchAiCpu(aicpuStream, kernelArgs, aicpuNum);
    if (rc < 0) {
        return rc;
    }
    rc = LaunchAiCore(aicoreStream, kernelArgs, blockDim);
    return rc;
}

int KernelLaunchManager::LaunchTripleStream(RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream,
                                            DeviceKernelArgs* kernelArgs, int blockDim, int aicpuNum)
{
    LoadAicpuOp::GetInstance().CustomAiCpuSoLoad();
    auto args = reinterpret_cast<dynamic::AiCpuArgs*>(kernelArgs->inputs);
    RtAicpuArgsEx rtArgs;
    uint64_t argsSize = reinterpret_cast<uint64_t>(kernelArgs->outputs);
    kernelArgs->inputs = nullptr;
    args->kArgs = *kernelArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = args;
    rtArgs.argsSize = argsSize;
    rtArgs.hostInputInfoNum = 1;
    rtArgs.kernelNameAddrOffset = offsetof(dynamic::AiCpuArgs, kernelName);
    rtArgs.soNameAddrOffset = offsetof(dynamic::AiCpuArgs, soName);
    RtHostInputInfo hostInputInfo;
    hostInputInfo.addrOffset = reinterpret_cast<int8_t*>(&args->kArgs.inputs) - reinterpret_cast<int8_t*>(args);
    hostInputInfo.dataOffset = sizeof(dynamic::AiCpuArgs);
    rtArgs.hostInputInfoPtr = &hostInputInfo;
    MACHINE_LOGI("Copy flow addrOffset %u argsSize %u", hostInputInfo.addrOffset, hostInputInfo.dataOffset);

    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_CTRL;
    int rc = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", 1, &rtArgs, nullptr,
        (AclRtStream)ctrlStream, 0);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "triple stream launch ctrl aicpu failed %d\n", rc);
        return rc;
    }

    args->kArgs.parameter.runMode = RUN_SPLITTED_STREAM_SCHE;
    rc = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", aicpuNum, &rtArgs, nullptr,
        (AclRtStream)schedStream, 0);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICPU_FAILED, "triple stream launch sche aicpu failed %d\n", rc);
        return rc;
    }

    rc = LaunchAiCore(aicoreStream, kernelArgs, blockDim);
    if (rc < 0) {
        MACHINE_LOGE(HostLauncherErr::LAUNCH_AICORE_FAILED, "triple stream launch aicore failed %d\n", rc);
        return rc;
    }
    return rc;
}

int KernelLaunchManager::InitAicpuServer(DeviceArgs* devArgs)
{
    auto aicpuStream = machine::GetRA()->GetScheStream();
#ifdef BUILD_WITH_NEW_CANN
    DeviceKernelArgs kArgs;
    return LoadAicpuOp::GetInstance().LaunchBuiltInOp(aicpuStream, &kArgs, 1, "PyptoInit");
#endif
    struct Args {
        DeviceKernelArgs kArgs;
        const char kernelName[32] = {"DynTileFwkKernelServerInit"};
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;

    args.kArgs.cfgdata = (int64_t*)devArgs;

    RtAicpuArgsEx rtArgs;
    memset_s(&rtArgs, sizeof(rtArgs), 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);
    int ret = RuntimeAicpuKernelLaunchExWithArgs(
        static_cast<uint32_t>(npu::tile_fwk::RtKernelType::AICPU_KFC), "AST_DYN_AICPU", 1, &rtArgs, nullptr,
        aicpuStream, 0);
    if (ret != RT_SUCCESS) {
        MACHINE_LOGE(RtErr::RT_LAUNCH_FAILED, "Aicpu server init failed %d", ret);
        return ret;
    }
    return RuntimeStreamSynchronize(aicpuStream);
}

void KernelLaunchManager::InitAiCpuSoBin(DeviceArgs& devArgs)
{
    std::vector<char> buffer;
    std::string fileName = GetCurrentSharedLibPath() + "/libtilefwk_backend_server.so";
    if (!ReadBytesFromFile(fileName, buffer)) {
        MACHINE_LOGE(
            DevCommonErr::FILE_ERROR, "Read bin form tilefwk_backend_server.so failed, please check the so[%s]",
            fileName.c_str());
        return;
    }
    size_t aicpuDataLength = buffer.size();
    uint8_t* dAicpuData = nullptr;
    machine::GetRA()->AllocDevAddr(&dAicpuData, aicpuDataLength);
    if (dAicpuData == nullptr) {
        MACHINE_LOGE(DevCommonErr::ALLOC_FAILED, "Alloc aicpu so bin failed");
        return;
    }
    RuntimeMemcpy(dAicpuData, aicpuDataLength, reinterpret_cast<void*>(buffer.data()), aicpuDataLength,
                  RtMemcpyKind::HOST_TO_DEVICE);
    devArgs.aicpuSoBin = reinterpret_cast<uint64_t>(dAicpuData);
    devArgs.aicpuSoLen = buffer.size();
    devArgs.deviceId = GetLogDeviceId();
    HOST_PERF_TRACE(TracePhase::RunDevKernelInitAicpuSo);
}

} // namespace npu::tile_fwk
