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
 * \file device_state.h
 * \brief Shared device state encapsulating DeviceArgs and runtime launch parameters.
 */

#pragma once

#include "interface/machine/device/tilefwk/aicpu_common.h"
#include "machine/device/dynamic/device_common.h"
#include "tilefwk/platform.h"

namespace npu::tile_fwk {

struct DeviceState {
    DeviceArgs args{};
    int blockDim{24};
    int aicpuNum{5};
    bool isCapture{false};
    bool initFlag{false};

    uint64_t devArgsAddr{0}; // device memory address of args copy

    uint64_t GetBlockNum() const { return args.nrAic + args.nrAiv; }

    DeviceArgs* GetDevArgsPtr() const
    {
        return reinterpret_cast<DeviceArgs*>(static_cast<uintptr_t>(devArgsAddr));
    }

    void UpdateLaunchParams(int blockdim, int aicpuNum)
    {
        blockDim = blockdim;
        args.nrValidAic = blockdim;
        args.nrAicpu = aicpuNum;
        args.scheCpuNum = CalcSchAicpuNumByBlockDim(static_cast<uint32_t>(blockdim),
                                                    static_cast<uint32_t>(aicpuNum),
                                                    args.archInfo);
    }

    void UpdateBlockDimFromPlatform()
    {
        auto blk = Platform::Instance().GetSoc().GetAICoreNum();
        blockDim = blk > 0 ? static_cast<int>(blk) : 24;
        args.nrValidAic = blockDim;
    }

    void UpdateAicpuNumFromPlatform()
    {
        int cpuNum = static_cast<int>(Platform::Instance().GetSoc().GetAICPUNum());
        aicpuNum = aicpuNum < cpuNum ? aicpuNum : cpuNum;
        args.maxAicpuNum = cpuNum;
        args.nrAicpu = aicpuNum;
        args.scheCpuNum = CalcSchAicpuNumByBlockDim(static_cast<uint32_t>(blockDim),
                                                    static_cast<uint32_t>(aicpuNum),
                                                    args.archInfo);
    }
};

} // namespace npu::tile_fwk
