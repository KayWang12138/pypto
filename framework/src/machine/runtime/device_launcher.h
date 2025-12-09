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
 * \file device_launcher.h
 * \brief
 */

#ifndef SRC_MACHINE_DEVICE_LAUNCHER_H
#define SRC_MACHINE_DEVICE_LAUNCHER_H

#include <cstdint>

#include "machine/runtime/device_launcher_binding.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "machine/device/dynamic/device_common.h"
#include "machine/runtime/device_memory_utils.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "machine/platform/platform_manager.h"

namespace npu::tile_fwk::dynamic {

int GetCfgBlockdim();

class DeviceLauncherContext {
public:
    void DeviceInit() {
        // 使能 Aihac 后端
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
#ifdef ENABLE_STEST_BINARY_CACHE
        // BinaryCache
        oriEnableBinaryCache = config::GetHostConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, true);
#endif
#ifdef ENABLE_STEST_DUMP_JSsON
        oriEnableDumpJson = config::GetPassConfig(KEY_PRINT_FUNCTION, oriEnableDumpJson);
        config::GetPassConfig(KEY_PRINT_FUNCTION, true);
#endif
        // Reset Program

        Program::GetInstance().Reset();
        ProgramData::GetInstance().Reset();

        config::SetHostOption(ONLY_CODEGEN, true);
    }

    void DeviceFini() {
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
#ifdef ENABLE_STEST_BINARY_CACHE
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
#endif
#ifdef ENABLE_STEST_DUMO_JSON
        config::SetHostConfig(KEY_PRINT_FUNCTION, oriEnablePrintJson);
#endif
    }
    static DeviceLauncherContext &Get();

protected:
    bool oriEnableAihacBackend = false;
#ifdef ENABLE_STEST_BINARY_CACHE
    bool oriEnableBinaryCache = false;
#endif
#ifdef ENABLE_STEST_DUMO_JSON
    bool oriEnableDumpJson = false;
#endif
};

class DeviceLauncher {
public:
    static constexpr uint32_t kDefaultAicNum = 25;
    static constexpr uint32_t kDefaultAivNum = 50;
    static std::vector<uint8_t>& GetDevProg(Function *func) {
        return func->GetDyndevAttribute()->devProgBinary;
    }

    static bool HasInplaceArgs(Function *function) {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(GetDevProg(function).data()));
        return devProg->outputInplaceSlotList.size() != 0;
    }

    template<typename DeviceMemoryTy>
    static void DeviceInitTilingData(DeviceMemoryTy devMem, AstKernelArgs &kArgs, const std::vector<uint8_t> &devProgData,
        const DeviceLauncherConfig &config, CachedOperator *cachedOperator) {
#ifdef BUILD_WITH_CANN
        int maxBlockDim = GetCfgBlockdim();
#else
        int maxBlockDim = 25;
#endif
        DeviceLauncherConfig &launchConfig = const_cast<DeviceLauncherConfig &>(config);
        if (config.blockdim == 0 || config.blockdim > maxBlockDim) {
            launchConfig.blockdim = maxBlockDim;
        }

        int scheCpuNum = 1;
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProgData.data()));
        if (PlatformManager::Instance().GetAicVersion() == "AIC-C-310") {
            devProg->devArgs.socVersion = SocVersion::AIC_310;
            launchConfig.aicpuNum = PlatformManager::Instance().GetAiCpuCnt() - 1;
            scheCpuNum = CalcSchAicpuNumByBlockDim(launchConfig.blockdim, launchConfig.aicpuNum, false);
        } else {
            scheCpuNum = CalcSchAicpuNumByBlockDim(launchConfig.blockdim, launchConfig.aicpuNum);
        }
        devProg->devArgs.nrAic = kDefaultAicNum;
        devProg->devArgs.nrAiv = kDefaultAivNum;
        devProg->devArgs.nrValidAic = config.blockdim;
        devProg->devArgs.scheCpuNum = scheCpuNum;
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
        size_t shmSize = DEVICE_SHM_SIZE + DEVICE_TASK_QUEUE_SIZE * devProg->devArgs.scheCpuNum;
        uint64_t shmAddr = (uint64_t)devMem.AllocZero(shmSize, CachedOperator::GetMetaDataDevAddrHolder(cachedOperator));
        devProg->devArgs.startArgsAddr = shmAddr;
        devProg->devArgs.taskCtrl = shmAddr + DEV_ARGS_SIZE;
        devProg->devArgs.taskQueue = shmAddr + DEV_ARGS_SIZE + DEVICE_TASK_CTRL_SIZE;
        int minCpuNum = devProg->devArgs.scheCpuNum + 1;
        if (config.aicpuNum < minCpuNum || config.aicpuNum > DEVICE_MAX_AICPU_NUM) {
            launchConfig.aicpuNum = minCpuNum + 1;
        }
        devProg->devArgs.nrAicpu = config.aicpuNum;
        ALOG_DEBUG_F("Set aicore blockdim:%d aicpu blockdim:%d.", config.blockdim, config.aicpuNum);
        devProg->devArgs.enableCtrl = 1; // need set 0 if use custom cpu launch ctrl cpu
        devProg->memBudget.tensor.dynDAssembleDests = AlignUp(config.dynWorkspaceSize, TENSOR_ADDR_ALIGNMENT);
        devProg->workspaceSize = devProg->memBudget.Total();

        devProg->l2CacheOffset = devMem.GetL2Offset();
        ASSERT(devProg->commGroupNum == config.hcclContext.size());
        ASSERT(devProg->commGroupNum <= (sizeof(devProg->hcclContext) / sizeof(uint64_t)));
        for (size_t i = 0; i < devProg->commGroupNum; i++) {
            devProg->hcclContext[i] = config.hcclContext[i];
        }
        if (config.workspaceAddr) {
            kArgs.workspace = (int64_t *)config.workspaceAddr;
        } else if (kArgs.workspace == nullptr) {
            kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize, CachedOperator::GetWorkspaceDevAddrHolder(cachedOperator));
        }
        if (devProg->controlFlowCache.isRecording && !devMem.IsDevice()) {
            kArgs.cfgdata = (int64_t *)devProg;
        } else if (CachedOperator::GetCfgDataDevAddrHolder(cachedOperator) && *CachedOperator::GetCfgDataDevAddrHolder(cachedOperator)) {
            /* Already copied, do not copy again. */
            kArgs.cfgdata = (int64_t *)*CachedOperator::GetCfgDataDevAddrHolder(cachedOperator);
        } else {
            kArgs.cfgdata = (int64_t *)devMem.CopyToDev(devProgData, CachedOperator::GetCfgDataDevAddrHolder(cachedOperator));
        }
        kArgs.machineConfig = devProg->devArgs.machineConfig;
        return;
    }

    template<typename DeviceMemoryTy>
    static void DeviceInitKernelInOuts(
            DeviceMemoryTy devMem,
            AstKernelArgs &kArgs,
            const std::vector<DeviceTensorData> &inputList,
            const std::vector<DeviceTensorData> &outputList) {
        auto buildInouts = [&](const std::vector<DeviceTensorData> &tensorDataList) {
            std::vector<DevTensorData> geTensors;
            for (size_t k = 0; k < tensorDataList.size(); k++) {
                auto &tensorData = tensorDataList[k];
                uint64_t addr = 0;
                if (tensorData.GetDevAddr() != 0) {
                    addr = (uint64_t)tensorData.GetDevAddr();
                }
                geTensors.emplace_back(DevAscendTensorDataCreator::Create(addr, tensorData.GetShape()));
            }
            std::vector<int64_t> encoded = DevAscendTensorDataCreator::Encode(geTensors);
            return encoded;
        };
        std::vector<int64_t> encodedInputList = buildInouts(inputList);
        std::vector<int64_t> encodedOutputList = buildInouts(outputList);
        kArgs.inputs = devMem.CopyToDev(encodedInputList, nullptr);
        kArgs.outputs = devMem.CopyToDev(encodedOutputList, nullptr);
        ALOG_INFO_F("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata);
        return;
    }

    template<typename DeviceMemoryTy>
    static std::pair<std::vector<DeviceTensorData>, std::vector<DeviceTensorData>> BuildInputOutputFromHost(
            DeviceMemoryTy devMem,
            const std::vector<RawTensorDataPtr> &inputDataList,
            const std::vector<RawTensorDataPtr> &outputDataList) {
        std::vector<DeviceTensorData> inputDeviceDataList;
        std::vector<DeviceTensorData> outputDeviceDataList;
        for (size_t k = 0; k < inputDataList.size(); k++) {
            auto &inputData = inputDataList[k];
            std::vector<int64_t> shape;
            if (inputData) {
                inputData->SetDevPtr(nullptr);
                shape.insert(shape.end(), inputData->GetShape().begin(), inputData->GetShape().end());
                inputDeviceDataList.emplace_back(inputData->GetDataType(), (uintdevptr_t)devMem.CopyToDev(*inputData), shape);
            } else {
                inputDeviceDataList.emplace_back(DT_UINT8, 0, shape);
            }
        }
        for (size_t k = 0; k < outputDataList.size(); k++) {
            auto &outputData = outputDataList[k];
            std::vector<int64_t> shape;
            if (outputData) {
                outputData->SetDevPtr(nullptr);
                shape.insert(shape.end(), outputData->GetShape().begin(), outputData->GetShape().end());
                outputDeviceDataList.emplace_back(outputData->GetDataType(), (uintdevptr_t)devMem.CopyToDev(*outputData), shape);
            } else {
                outputDeviceDataList.emplace_back(DT_UINT8, 0, shape);
            }
        }
        return std::make_pair(inputDeviceDataList, outputDeviceDataList);
    }

    template<typename DeviceMemoryTy>
    static void CopyFromDev(
            DeviceMemoryTy devMem,
            const std::vector<RawTensorDataPtr> &outputs) {
        for (auto &output : outputs) {
            if (output) {
                devMem.CopyFromDev(*output);
            }
        }
    }

#ifdef BUILD_WITH_CANN
    static void ChangeCaptureMode(aclmdlRICaptureMode &mode);
    static int GetStreamCaptureInfo(rtStream_t aicoreStream, aclmdlRI &rtModel, bool &isCapture);
    static int SetCaptureStream(rtStream_t aicoreStream, rtStream_t aicpuStream);
    static int DeviceLaunchOnceWithDeviceTensorData(
            Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize, CachedOperator *cachedOperator,
            const DeviceLauncherConfig &config = DeviceLauncherConfig());

    static int DeviceSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream);
#else
using aclmdlRICaptureMode = uint32_t;
using rtStream_t = uint64_t;
using aclmdlRI = void *;
    static void ChangeCaptureMode(aclmdlRICaptureMode &mode) {
        (void)mode;
        return;
    }
    static int GetStreamCaptureInfo(rtStream_t aicoreStream, aclmdlRI &rtModel, bool &isCapture) {
        (void)aicoreStream;
        (void)rtModel;
        (void)isCapture;
        return 0;
    }
    static int SetCaptureStream(rtStream_t aicoreStream, rtStream_t aicpuStream) {
        (void)aicoreStream;
        (void)aicpuStream;
        return 0;
    }
    static int DeviceLaunchOnceWithDeviceTensorData(
            Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize, CachedOperator *cachedOperator, uintptr_t workspacePtr,
            const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        (void)function;
        (void)inputList;
        (void)outputList;
        (void)aicpuStream;
        (void)aicoreStream;
        (void)streamSynchronize;
        (void)cachedOperator;
        (void)workspacePtr;
        (void)config;
        return 0;
    }

    static int DeviceSynchronize(rtStream_t aicpuStream, rtStream_t aicoreStream) {
        (void)aicoreStream;
        (void)aicpuStream;
        return 0;
    }
#endif
    static int DeviceRunOnce(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig());

    static void DeviceRunCacheKernelEnable(Function *func, bool enabled);
    static bool DeviceRunCacheKernelEnable(Function *func);
    static void DeviceRunCacheKernelSet(Function *func, uint8_t *devProg);
    static uint8_t *DeviceRunCacheKernelGet(Function *func);
};
}
#endif//SRC_MACHINE_DEVICE_LAUNCHER_H
