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

#ifndef SRC_RUNTIME_DEVICE_LAUNCHER_BINDING_H
#define SRC_RUNTIME_DEVICE_LAUNCHER_BINDING_H

#include <cstdint>
#include <vector>

#include "interface/function/function.h"
#include "interface/program/program.h"
#include "machine/utils/dynamic/dev_encode.h"
#include <runtime/rt.h>

namespace npu::tile_fwk::dynamic {

using DeviceStream = unsigned long long;
DeviceStream DeviceGetAicpuStream();
DeviceStream DeviceGetAicoreStream();

class DeviceTensorData {
public:
    DeviceTensorData(DataType dtype, uintptr_t devAddr, const std::vector<int64_t> &shape)
        : dtype_(dtype), devAddr_(devAddr), shape_(shape){}
    uintptr_t GetDevAddr() const { return devAddr_; }
    const std::vector<int64_t> &GetShape() const { return shape_; }
    DataType GetDataType() const { return dtype_; }
    int64_t GetDataSize() const {
        int64_t size = BytesOf(dtype_);
        for (auto dim : shape_) {
            size *= dim;
        }
        return size;
    }

    static DeviceTensorData Create(const std::shared_ptr<LogicalTensor> &t) {
        return DeviceTensorData(t->Datatype(), 0, t->GetShape());
    }
private:
    DataType dtype_;
    uintptr_t devAddr_;
    std::vector<int64_t> shape_;
};

struct DeviceLauncherConfig {
    bool onBoard{true};
    int blockdim{0};
    int aicpuNum{5};
    int64_t dynWorkspaceSize{0};
    int64_t repeatNum{1};
    bool runModel{true};
    std::vector<uint64_t> hcclContext;
    bool controlFlowCache{false};
    bool cpuSeparate{false};

    DeviceLauncherConfig() = default;
    DeviceLauncherConfig(bool onboard, int tblockdim, int taicpunum)
        : onBoard(onboard), blockdim(tblockdim), aicpuNum(taicpunum) {}
    DeviceLauncherConfig(int64_t tdynWorkspaceSize) : dynWorkspaceSize(tdynWorkspaceSize) {}
    DeviceLauncherConfig(int64_t tdynWorkspaceSize, int64_t trepeatNum)
        : dynWorkspaceSize(tdynWorkspaceSize), repeatNum(trepeatNum) {}
    DeviceLauncherConfig(const std::vector<std::uint64_t> &addrs) : hcclContext(addrs) {}
};

class CachedOperator {
public:
    static uint8_t **GetWorkspaceDevAddrHolder(CachedOperator *cachedOperator) {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->workspaceDevAddr_;
    }
    static uint8_t **GetCfgDataDevAddrHolder(CachedOperator *cachedOperator) {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->cfgDataDevAddr_;
    }
    static uint8_t **GetMetaDataDevAddrHolder(CachedOperator *cachedOperator) {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->metaDataDevAddr_;
    }

    void UpdateInputOutput(
            const std::vector<std::shared_ptr<LogicalTensor>> &inputList,
            const std::vector<std::shared_ptr<LogicalTensor>> &outputList) {
        for (auto &input : inputList) {
            inputList_.emplace_back(DeviceTensorData::Create(input));
        }
        for (auto &output : outputList) {
            outputList_.emplace_back(DeviceTensorData::Create(output));
        }
    }
    const std::vector<DeviceTensorData> &GetInputList() { return inputList_; }
    const std::vector<DeviceTensorData> &GetOutputList() { return outputList_; }
    static rtBinHandle *GetBinHandleHolder(CachedOperator *cachedOperator) {
        return cachedOperator == nullptr ? nullptr : &cachedOperator->binHandle_;
    }
private:
    uint8_t *workspaceDevAddr_{nullptr};
    uint8_t *cfgDataDevAddr_{nullptr};
    uint8_t *metaDataDevAddr_{nullptr};
    std::vector<DeviceTensorData> inputList_;
    std::vector<DeviceTensorData> outputList_;
    rtBinHandle binHandle_{nullptr};
};

class ExportedOperator : public CachedOperator {
public:
    void ResetFunction(Function *func) {
        func_ = Program::GetInstance().GetFunctionSharedPtr(func);
    }

    Function *GetFunction() const { return func_.get(); }
    uint64_t GetWorkSpaceSize() const {
        const std::vector<uint8_t> &devProgData = func_->GetDyndevAttribute()->devProgBinary;
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(devProgData.data())); 
        return devProg->memBudget.metadata.Total() + devProg->memBudget.tensor.Total() +
               devProg->memBudget.aicoreSpilled + devProg->memBudget.debug.dumpTensor;
    }
private:
    std::shared_ptr<Function> func_;
};

int ExportedOperatorDeviceLaunchOnceWithDeviceTensorData(
        ExportedOperator *op, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        DeviceStream aicpuStream, DeviceStream aicoreStream, bool streamSynchronize, uintptr_t workspacePtr,
        const DeviceLauncherConfig &config = DeviceLauncherConfig());

int DeviceSynchronize(DeviceStream aicpuStream, DeviceStream aicoreStream);

int DeviceRunOnce(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig());

int HasInplaceArgs(Function *function);

void DeviceLauncherInit();

void DeviceLauncherFini();

ExportedOperator *ExportedOperatorBegin();

void ExportedOperatorEnd(ExportedOperator *op);

}

#endif//SRC_MACHINE_DEVICE_LAUNCHER_H
