/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file device_runner.cpp
 * \brief
 */

#ifdef BUILD_WITH_CANN
#include "device_launcher.h"

#include "machine/host/backend.h"

namespace npu::tile_fwk::dynamic {

void (*forceLinkLibraryCompiler)() = &npu::tile_fwk::ForceLinkLibraryCompiler;

DeviceLauncherContext &DeviceLauncherContext::Get() {
    static DeviceLauncherContext context;
    return context;
}

int DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(
        Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize, CachedOperator *cachedOperator,
        const DeviceLauncherConfig &config) {
    std::cout << "!!! Kernel Launch " << "\n";
    if (function != nullptr && function->GetDyndevAttribute() != nullptr) {
        DeviceRunner::SetBinData(function->GetDyndevAttribute()->kernelBinary);
    }
    int rc = aclInit(nullptr);
    if (rc == 0 || rc == ACL_ERROR_REPEAT_INITIALIZE) {
        SetDefaultDevice();
        AstKernelArgs kArgs;
        DeviceInitTilingData(DeviceMemoryUtils(), kArgs, function, config, cachedOperator);
        DeviceInitKernelInOuts(DeviceMemoryUtils(), kArgs, inputList, outputList, cachedOperator);
        rc = DeviceRunner::Get().DynamicLaunch(aicpuStream, aicoreStream, 0, &kArgs, config.blockdim, config.aicpuNum);
        if (rc < 0) {
            return rc;
        }
        if (streamSynchronize) {
            rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, aicoreStream);
        }
    }
    return rc;
}

int DeviceLauncher::DeviceSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream) {
    int rc = DeviceRunner::Get().DynamicLaunchSynchronize(aicpuStream, aicoreStream);
    return rc;
}

int DeviceLauncher::DeviceRunOnce(Function *function, const DeviceLauncherConfig &config) {
    auto &inputDataList = ProgramData::GetInstance().GetInputDataList();
    auto &outputDataList = ProgramData::GetInstance().GetOutputDataList();
    auto aicpuStream = machine::GetRA()->GetStreamAICPU();
    auto aicoreStream = machine::GetRA()->GetStream();
    std::vector<DeviceTensorData> inputDeviceDataList;
    std::vector<DeviceTensorData> outputDeviceDataList;
    std::tie(inputDeviceDataList, outputDeviceDataList) = BuildInputOutputFromHost(DeviceMemoryUtils(), inputDataList, outputDataList);
    int rc = DeviceLaunchOnceWithDeviceTensorData(function, inputDeviceDataList, outputDeviceDataList, aicpuStream, aicoreStream, true, nullptr, config);
    CopyFromDev(DeviceMemoryUtils(), outputDataList);
    if (HasInplaceArgs(function)) {
        CopyFromDev(DeviceMemoryUtils(), inputDataList);
    }
    return rc;
}

DeviceStream DeviceGetAicpuStream() {
    rtStream_t aicpuStreamValue = machine::GetRA()->GetStreamAICPU();
    return reinterpret_cast<DeviceStream>(aicpuStreamValue);
}

DeviceStream DeviceGetAicoreStream() {
    rtStream_t aicoreStreamValue = machine::GetRA()->GetStream();
    return reinterpret_cast<DeviceStream>(aicoreStreamValue);
}

int ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(
        ExportedOperator *op, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        DeviceStream aicpuStream, DeviceStream aicoreStream, bool streamSynchronize,
        const DeviceLauncherConfig &config) {
    rtStream_t aicpuStreamValue = reinterpret_cast<rtStream_t>(aicpuStream);
    rtStream_t aicoreStreamValue = reinterpret_cast<rtStream_t>(aicoreStream);
    return DeviceLauncher::DeviceLaunchOnceWithDeviceTensorData(op->GetFunction(), inputList, outputList, aicpuStreamValue, aicoreStreamValue, streamSynchronize, op, config);
}

int DeviceSynchronize(DeviceStream aicpuStream, DeviceStream aicoreStream) {
    rtStream_t aicpuStreamValue = reinterpret_cast<rtStream_t>(aicpuStream);
    rtStream_t aicoreStreamValue = reinterpret_cast<rtStream_t>(aicoreStream);
    return DeviceLauncher::DeviceSynchronize(aicpuStreamValue, aicoreStreamValue);
}

int DeviceRunOnce(Function *function, const DeviceLauncherConfig &config) {
    return DeviceLauncher::DeviceRunOnce(function, config);
}

int HasInplaceArgs(Function *function) {
    return DeviceLauncher::HasInplaceArgs(function);
}

void DeviceLauncherInit() {
    DeviceLauncherContext::Get().DeviceInit();
}

void DeviceLauncherFini() {
    DeviceLauncherContext::Get().DeviceFini();
}

static std::unordered_map<ExportedOperator *, std::shared_ptr<ExportedOperator>> exportedOperatorDict;

ExportedOperator *ExportedOperatorBegin() {
    std::shared_ptr<ExportedOperator> op = std::make_shared<ExportedOperator>();
    exportedOperatorDict[op.get()] = op;
    return op.get();
}

void ExportedOperatorEnd(ExportedOperator *op) {
    op->ResetFunction(Program::GetInstance().GetLastFunction());
}

}

#endif
