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

#ifndef SRC_MACHINE_EMULATION_LAUNCHER_H
#define SRC_MACHINE_EMULATION_LAUNCHER_H

#include <cstdint>

#include "device_launcher_binding.h"

#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "machine/utils/dynamic/dev_encode.h"
#include "runtime.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "tilefwk/data_type.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/configs/config_manager.h"

#include "device_launcher.h"

namespace npu::tile_fwk::dynamic {

struct EmulationMemoryUtils {
    uint8_t *AllocDev(size_t size, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        uint8_t *devPtr = machine::GetRA()->AllocHostAddr(size);
        return devPtr;
    }

    uint8_t *CopyToDev(uint8_t *data, uint64_t size) {
        uint8_t *devPtr = AllocDev(size, nullptr);
        memcpy_s(devPtr, size, data, size);
        return devPtr;
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data, uint8_t **cachedDevAddrHolder) {
        (void)cachedDevAddrHolder;
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T));
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) {
        memcpy_s(data, size, devPtr, size);
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

class EmulationLauncher {
public:
    static int EmulationLaunchOnceWithHostTensorData(
            Function *function, const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
            const DeviceLauncherConfig &config = DeviceLauncherConfig());

    static int EmulationRunOnce(Function *function, const DeviceLauncherConfig &config = DeviceLauncherConfig());
};

}

#endif//SRC_MACHINE_EMULATION_LAUNCHER_H