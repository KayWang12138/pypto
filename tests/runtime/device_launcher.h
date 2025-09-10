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

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "runtime.h"
#include "device_runner.h"

namespace npu::tile_fwk::dynamic {

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
};

class DeviceTensorData {
public:
    DeviceTensorData(uintdevptr_t devAddr, const std::vector<int64_t> &shape) : devAddr_(devAddr), shape_(shape) {}
    uintdevptr_t GetDevAddr() const { return devAddr_; }
    const std::vector<int64_t> &GetShape() const { return shape_; }
private:
    uintdevptr_t devAddr_;
    std::vector<int64_t> shape_;
};

struct DeviceLauncherConfig {
    bool onBoard{true};
    int blockdim{25};
    int aicpuNum{5};
    int64_t dynWorkspaceSize{0};
    int64_t repeatNum{1};
    bool runModel{true};
    std::vector<uint64_t> hcclContext;

    DeviceLauncherConfig() = default;
    DeviceLauncherConfig(bool onboard, int tblockdim, int taicpunum) : onBoard(onboard), blockdim(tblockdim), aicpuNum(taicpunum) {}
    DeviceLauncherConfig(int tdynWorkspaceSize) : dynWorkspaceSize(tdynWorkspaceSize) {}
    DeviceLauncherConfig(int tdynWorkspaceSize, int64_t trepeatNum) : dynWorkspaceSize(tdynWorkspaceSize), repeatNum(trepeatNum){}
    DeviceLauncherConfig(const std::vector<std::uint64_t> &addrs) : hcclContext(addrs) {}
};

class DeviceLauncher {
protected:
    static constexpr uint32_t kDefaultAicNum = 25;
    static constexpr uint32_t kDefaultAivNum = 50;

    static const std::vector<uint8_t>& GetDevProg(Function *func) {
        return func->GetDyndevAttribute()->devProgBinary;
    }

    template<typename DeviceMemoryTy>
    static void DeviceInitTilingData(
            DeviceMemoryTy devMem,
            AstKernelArgs &kArgs,
            Function *func,
            const DeviceLauncherConfig &config) {
        auto *devProg = reinterpret_cast<DevAscendProgram *>(const_cast<uint8_t*>(GetDevProg(func).data()));
        devProg->devArgs.nrAic = kDefaultAicNum;
        devProg->devArgs.nrAiv = kDefaultAivNum;
        devProg->devArgs.nrAicpu = config.aicpuNum;
        devProg->devArgs.nrValidAic = config.blockdim;
        devProg->devArgs.taskType = DEVICE_TASK_TYPE_DYN;
        devProg->workspaceSize = devProg->aicoreLocalWorkspaceSize + devProg->aicpuCoherentWorkspaceSize
                                 + config.dynWorkspaceSize;
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
            for (auto tensorData : tensorDataList) {
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

    static void DeviceRunOnceWithDeviceTensorData(
            Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            rtStream_t aicpuStream, rtStream_t aicoreStream,
            const DeviceLauncherConfig &config = DeviceLauncherConfig()) {
        std::cout << "!!! Kernel Launch " << "\n";
        if (function != nullptr && function->GetDyndevAttribute() != nullptr) {
            DeviceRunner::SetBinData(function->GetDyndevAttribute()->kernelBinary);
        }
        int rc = aclInit(nullptr);
        if (rc == 0 || rc == ACL_ERROR_REPEAT_INITIALIZE) {
            rtSetDevice(npu::tile_fwk::stubs::DeviceStub::GetCurrentDeviceId());
            AstKernelArgs kArgs;
            DeviceInitTilingData(DeviceMemoryUtils(), kArgs, function, config);
            DeviceInitKernelInOuts(DeviceMemoryUtils(), kArgs, inputList, outputList);
            rc = DeviceRunner::Get().DynamicRun(aicpuStream, aicoreStream, 0, &kArgs, config.blockdim, config.aicpuNum);
            EXPECT_EQ(rc, 0);
        }
    }
};

}

#endif//SRC_MACHINE_DEVICE_LAUNCHER_H