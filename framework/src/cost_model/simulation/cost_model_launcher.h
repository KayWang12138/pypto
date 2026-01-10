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
 * \file cost_model_launcher.h
 * \brief
 */

#pragma once

#include <thread>
#include <cstdint>
#include <unistd.h>
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/host/device_agent_task.h"
#include "machine/device/dynamic/costmodel_utils.h"
#include "machine/runtime/machine_agent.h"
#include "machine/runtime/device_launcher.h"
#include "cost_model/simulation/backend.h"
#include "machine/runtime/host_prof.h"


namespace npu::tile_fwk::dynamic{

class HostAgentStub {
public:
    HostAgentStub(HostAgentStub &other) = delete;

    void operator=(const HostAgentStub &other) = delete;

    static HostAgentStub *GetAgent() {
        static HostAgentStub inst;
        return &inst;
    }

    uint8_t* AllocHostAddr(uint64_t size) {
        if (size == 0) {
            ALOG_ERROR_F("malloc size is 0!");
            return nullptr;
        }
        auto hostPtr = (uint8_t *)malloc(size);
        allocatedHostAddr.emplace_back(hostPtr);
        return hostPtr;
    }

    void Finalize() {
        if (hostInited) {
            DestroyMemory();
        }
    }

    ~HostAgentStub() { Finalize(); }

protected:
    HostAgentStub() {
        Init();
    }

    void DestroyMemory() {
        for (uint8_t *addr : allocatedHostAddr) {
            free(addr);
        }
    }

private:
    void Init() {
        hostInited = true;
    }

private:
    bool hostInited{false};

    std::vector<uint8_t *> allocatedHostAddr;
};

struct MemoryHelper {
    MemoryHelper(bool isTest) : isTest_(isTest) {}

    bool IsDevice() { return !isTest_; }

    uint8_t *CopyToDev(uint8_t *data, uint64_t size) {
        auto ptr = npu::tile_fwk::dynamic::HostAgentStub::GetAgent()->AllocHostAddr(size);
        memcpy_s(ptr, size, data, size);
        return ptr;
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) {
        if (isTest_) {
            memcpy_s(data, size, devPtr, size);
        }
    }

    uint8_t *AllocDev(size_t size, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        uint8_t *devPtr = nullptr;
        devPtr = npu::tile_fwk::dynamic::HostAgentStub::GetAgent()->AllocHostAddr(size);
        return devPtr;
    }

    uint8_t *AllocZero(uint64_t size, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        uint8_t *devPtr = AllocDev(size, nullptr);
        memset_s(devPtr, size, 0, size);
        return devPtr;
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T));
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

    uint64_t GetL2Offset() {
        return 0;
    }

    bool isTest_{true};
};

class CostModelLauncher : public DeviceLauncher {
public:
    static void CostModelRunOnce(Function *function, const std::vector<RawTensorDataPtr> &inputs,
        const std::vector<RawTensorDataPtr> &outputs, const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        auto runner = CostModelLauncher(function, config);
        runner.RunDynamic(inputs, outputs);
        RunStatic();
    }

    // Run with incast/outcast from ProgramData
    static void CostModelRunOnce(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        auto &inputs = ProgramData::GetInstance().GetInputDataList();
        auto &outputs = ProgramData::GetInstance().GetOutputDataList();
        auto runner = CostModelLauncher(function, config);
        runner.RunDynamic(inputs, outputs);
        RunStatic();
    }

private:
    CostModelLauncher(Function *function, const DeviceLauncherConfig &config) : function_(function), config_(config) {}

    void RunDynamic(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        if (function_ == nullptr || function_->GetDyndevAttribute() == nullptr) {
            return;
        }

        DevAscendProgram *functionDevProg = reinterpret_cast<DevAscendProgram *>(function_->GetDyndevAttribute()->devProgBinary.data());
        if (config_.controlFlowCache) {
            functionDevProg->controlFlowCache.isRecording = true;
        }
        RunModel(inputs, outputs);
    }

    static void RunStatic() {
    }

    void RunModel(const std::vector<RawTensorDataPtr> &inputs, const std::vector<RawTensorDataPtr> &outputs) {
        if (!config_.runModel) {
            return;
        }
        AstKernelArgs kArgs;
        config_.onBoard = false;
        DeviceLauncherConfigFillDeviceInfo(config_);
        DeviceInitTilingData(MemoryHelper(true), kArgs, function_->GetDyndevAttribute()->devProgBinary, config_, nullptr);
        InitKernelInOuts(kArgs, inputs, outputs, true);
        std::cout << "Run CostModel " << "\n";
        CostModelAgent costModeAgent;
        costModeAgent.RunCostModel(kArgs.costmodeldata);
        std::cout << "Run TestModel " << "\n";
        RunTestMode(&kArgs);
        std::cout << "Run DynCostModel " << "\n";
        costModeAgent.RunDynCostModel();
    }

    void InitKernelInOuts(AstKernelArgs &kArgs, const std::vector<RawTensorDataPtr> &inputTensors,
        const std::vector<RawTensorDataPtr> &outputTensors, bool isTest) {
        std::vector<DeviceTensorData> inputList;
        std::vector<DeviceTensorData> outputList;
        std::tie(inputList, outputList) = BuildInputOutputFromHost(MemoryHelper(isTest), inputTensors, outputTensors);
        DeviceInitKernelInOuts(MemoryHelper(isTest), kArgs, inputList, outputList, {}, false);
        ALOG_INFO_F("Inputs %p outputs %p workspace %p cfgdata %p", kArgs.inputs, kArgs.outputs, kArgs.workspace,
            kArgs.cfgdata);
    }

private:
    Function *function_;
    DeviceLauncherConfig config_;
}; // CostModelLauncher
} // npu::tile_fwk::dynamic
