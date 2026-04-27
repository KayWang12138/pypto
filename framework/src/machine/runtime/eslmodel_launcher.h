/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
* \file eslmodel_memory_utils.h
* \brief
*/

#pragma once

#include <sys/mman.h>
#include "adapter/api/acl_define.h"
#include "adapter/api/runtime_api.h"
#include "machine/runtime/device_launcher_binding.h"
#include "machine/runtime/runtime.h"
#include "machine/platform/platform_manager.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk::dynamic {
struct MmapRecord {
    void* addr;
    size_t size;
};

class MmapGlobalManager {
public:
    static void AddRecord(void* addr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back({addr, size});
    }

    static void UnmapAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& rec : records_) {
            if (rec.addr != nullptr && rec.addr != MAP_FAILED) {
                munmap(rec.addr, rec.size);
            }
        }
        records_.clear();
    }

private:
    static std::vector<MmapRecord> records_;
    static std::mutex mutex_;
};

inline std::vector<MmapRecord> MmapGlobalManager::records_;
inline std::mutex MmapGlobalManager::mutex_;



class EslModelLauncher {
public:
    static int EslModelRunOnce(void *kernel, const DeviceLauncherConfig &config = DeviceLauncherConfig());
    static int EslModelLaunchDeviceTensorData(Function *function,
        const std::vector<DeviceTensorData> &inputList, const std::vector<DeviceTensorData> &outputList,
        RtStream aicpuStream, RtStream aicoreStream, void *kernel, const DeviceLauncherConfig &config);
    static void ExchangeCaputerMode(const bool &isCapture);
    static int DynamicKernelLaunchEsl(DeviceKernelArgs *kArgs, AclRtStream aicoreStream, void *kernel);
    static int EslModelLaunchAicore(AclRtStream aicoreStream, void *kernel, DeviceKernelArgs *kernelArgs);
    static void CopyInputOutputData();
    static int EslModelLiteRunOnce(Function *function, std::vector<DeviceTensorData> &tensors);
};
}
