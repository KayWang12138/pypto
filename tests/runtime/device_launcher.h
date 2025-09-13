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
 * \file device_launcher.h
 * \brief
 */

#ifndef SRC_MACHINE_DEVICE_LAUNCHER_H
#define SRC_MACHINE_DEVICE_LAUNCHER_H

#include <cstdint>

#include "device_launcher_binding.h"

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "runtime.h"
#include "device_runner.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"

namespace npu::tile_fwk::dynamic {

class DeviceLauncherContext {
public:
    void DeviceInit() {
        // 使能 Aihac 后端
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
#ifdef ENABLE_TESTS_STEST_BINARY_CACHE
        // BinaryCache
        oriEnableBinaryCache = config::GetHostConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, true);
#endif
#ifdef ENABLE_TESTS_STEST_DUMP_JSsON
        oriEnableDumpJson = config::GetPassConfig(KEY_PRINT_FUNCTION, oriEnableDumpJson);
        config::GetPassConfig(KEY_PRINT_FUNCTION, true);
#endif
        // Reset Program

        Program::GetInstance().Reset();
        ProgramData::GetInstance().Reset();

        config::SetHostConfig(KEY_ONLY_CODEGEN, true);
    }

    void DeviceFini() {
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
#ifdef ENABLE_TESTS_STEST_BINARY_CACHE
        config::SetHostConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
#endif
#ifdef ENABLE_TESTS_STEST_DUMO_JSON
        config::SetHostConfig(KEY_PRINT_FUNCTION, oriEnablePrintJson);
#endif
    }
    static DeviceLauncherContext &Get();

protected:
    bool oriEnableAihacBackend = false;
#ifdef ENABLE_TESTS_STEST_BINARY_CACHE
    bool oriEnableBinaryCache = false;
#endif
#ifdef ENABLE_TESTS_STEST_DUMO_JSON
    bool oriEnableDumpJson = false;
#endif
};

struct DeviceMemoryUtils {
    uint8_t *AllocDev(size_t size) {
        uint8_t *devPtr = nullptr;
        machine::GetRA()->AllocDevAddr(&devPtr, size);
        return devPtr;
    }

    uint8_t *CopyToDev(uint8_t *data, uint64_t size) {
        uint8_t *devPtr = AllocDev(size);
        rtMemcpy(devPtr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
        return devPtr;
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data) {
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T));
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) {
        rtMemcpy(data, size, devPtr, size, RT_MEMCPY_DEVICE_TO_HOST);
    }

    uint8_t *CopyToDev(RawTensorData &data) {
        if (data.GetDevPtr() == nullptr) {
            auto devPtr = CopyToDev((uint8_t *)data.data(), data.size());
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData &t) {
        CopyFromDev(t.data(), t.GetDevPtr(), t.size());
    }
};

class DeviceLauncher {
public:
    static constexpr uint32_t kDefaultAicNum = 25;
    static constexpr uint32_t kDefaultAivNum = 50;

    static const std::vector<uint8_t>& GetDevProg(Function *func) {
        return func->GetDyndevAttribute()->devProgBinary;
    }

    static bool HasInplaceArgs(Function *function) {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(GetDevProg(function).data()));
        return devProg->inplaceSlotList.size() != 0;
    }

    template<typename DeviceMemoryTy>
    static void DeviceInitTilingData(
            DeviceMemoryTy devMem,
            AstKernelArgs &kArgs,
            Function *func,
            const DeviceLauncherConfig &config) {
        const std::vector<uint8_t> &devProgData = GetDevProg(func);
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProgData.data()));
        devProg->devArgs.nrAic = kDefaultAicNum;
        devProg->devArgs.nrAiv = kDefaultAivNum;
        devProg->devArgs.nrAicpu = config.aicpuNum;
        devProg->devArgs.nrValidAic = config.blockdim;
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
        devProg->workspaceSize = devProg->aicoreLocalWorkspaceSize + devProg->aicpuCoherentWorkspaceSize +
                                 devProg->debugDumpTensorMemReq + config.dynWorkspaceSize;
        devProg->l2CacheOffset = machine::GetRA()->GetL2Offset();
        ASSERT((devProg->commGroupNum == config.hcclContext.size()) &&
            (devProg->commGroupNum <= (sizeof(devProg->hcclContext) / sizeof(uint64_t))));
        for (size_t i = 0; i < devProg->commGroupNum; i++) {
            devProg->hcclContext[i] = config.hcclContext[i];
        }
        kArgs.workspace = (int64_t *)devMem.AllocDev(devProg->workspaceSize);
        kArgs.cfgdata = (int64_t *)devMem.CopyToDev(GetDevProg(func));
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
            std::vector<int64_t> outs = DevAscendTensorDataCreator::Encode(geTensors);
            return devMem.CopyToDev(outs);
        };
        kArgs.inputs = buildInouts(inputList);
        kArgs.outputs = buildInouts(outputList);
        ALOG_INFO_F("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata);
        return;
    }

    template<typename DeviceMemoryTy>
    static std::pair<std::vector<DeviceTensorData>, std::vector<DeviceTensorData>> BuildInputOutput(
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
                inputDeviceDataList.emplace_back((uintdevptr_t)devMem.CopyToDev(*inputData), shape);
            } else {
                inputDeviceDataList.emplace_back(0, shape);
            }
        }
        for (size_t k = 0; k < outputDataList.size(); k++) {
            auto &outputData = outputDataList[k];
            std::vector<int64_t> shape;
            if (outputData) {
                outputData->SetDevPtr(nullptr);
                shape.insert(shape.end(), outputData->GetShape().begin(), outputData->GetShape().end());
                outputDeviceDataList.emplace_back((uintdevptr_t)devMem.CopyToDev(*outputData), shape);
            } else {
                outputDeviceDataList.emplace_back(0, shape);
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

    static int DeviceLaunchOnceWithDeviceTensorData(
            Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            rtStream_t aicpuStream, rtStream_t aicoreStream, bool streamSynchronize,
            const DeviceLauncherConfig &config = DeviceLauncherConfig());

    static int DeviceRunOnce(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig());
};

}

#endif//SRC_MACHINE_DEVICE_LAUNCHER_H