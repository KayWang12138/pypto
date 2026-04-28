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
 * \file kernel_launch_manager.h
 * \brief Manages kernel binary registration and low-level kernel launching.
 */

#pragma once

#include <cstdint>
#include <vector>
#include "adapter/api/runtime_define.h"
#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk {

class KernelLaunchManager {
public:
    KernelLaunchManager() = default;

    // Kernel binary management
    void SetBinData(const std::vector<uint8_t>& binBuf);
    int RegisterKernelBin(void** hdl, std::vector<uint8_t>* funcBinBuf = nullptr);
    RtBinHandle GetBinHandle() const { return binHdl_; }

    // Low-level kernel launch primitives
    int LaunchAiCore(RtStream aicoreStream, DeviceKernelArgs* kernelArgs, int blockDim);
    int LaunchAiCpu(RtStream aicpuStream, DeviceKernelArgs* kArgs, int aicpuNum);

    // Composite launch: AICPU + AICore
    int LaunchKernelPair(RtStream aicpuStream, RtStream aicoreStream,
                         DeviceKernelArgs* kernelArgs, int blockDim, int aicpuNum);

    // Triple stream launch: ctrl + sched + aicore
    int LaunchTripleStream(RtStream schedStream, RtStream ctrlStream, RtStream aicoreStream,
                           DeviceKernelArgs* kernelArgs, int blockDim, int aicpuNum);

    // AICPU server initialization
    int InitAicpuServer(DeviceArgs* devArgs);
    void InitAiCpuSoBin(DeviceArgs& devArgs);

private:
    RtBinHandle binHdl_{nullptr};
    std::vector<uint8_t> binBuf_;
};

} // namespace npu::tile_fwk
