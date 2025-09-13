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

#ifndef SRC_RUNTIME_DEVICE_LAUNCHER_BINDING_H
#define SRC_RUNTIME_DEVICE_LAUNCHER_BINDING_H

#include <cstdint>
#include <vector>

#include "interface/function/function.h"

namespace npu::tile_fwk::dynamic {

typedef unsigned long long DeviceStream;

DeviceStream DeviceGetAicpuStream();
DeviceStream DeviceGetAicoreStream();

class DeviceTensorData {
public:
    DeviceTensorData(uintptr_t devAddr, const std::vector<int64_t> &shape) : devAddr_(devAddr), shape_(shape) {}
    uintptr_t GetDevAddr() const { return devAddr_; }
    const std::vector<int64_t> &GetShape() const { return shape_; }
private:
    uintptr_t devAddr_;
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

int DeviceLaunchOnceWithDeviceTensorData(
        Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        DeviceStream aicpuStream, DeviceStream aicoreStream, bool streamSynchronize,
        const DeviceLauncherConfig &config = DeviceLauncherConfig());

int DeviceRunOnce(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig());

void DeviceLauncherInit();

void DeviceLauncherFini();

}

#endif//SRC_MACHINE_DEVICE_LAUNCHER_H