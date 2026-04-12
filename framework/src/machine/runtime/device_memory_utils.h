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
 * \file device_memory_utils.h
 * \brief
 */

#pragma once

#ifdef BUILD_WITH_CANN

#include "machine/runtime/runtime.h"
#include "machine/runtime/device_runner.h"
#include "machine/runtime/memory/memory_manager.h"
#include "machine/platform/platform_manager.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk::dynamic {
struct DeviceMemoryUtils {
    DeviceMemoryUtils(bool isHugePage = true)
        : isUseHugePage_(isHugePage), devMemMgr_(std::make_shared<DeviceMemoryManager>(isHugePage))
    {}
    static bool IsDevice() { return true; }
    uint8_t* AllocDev(size_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = cachedDevAddrHolder == nullptr ? nullptr : *cachedDevAddrHolder;
        if (devPtr == nullptr) {
            devPtr = reinterpret_cast<uint8_t*>(devMemMgr_->Alloc(size));
            if (cachedDevAddrHolder != nullptr) {
                *cachedDevAddrHolder = devPtr;
            }
        }
        return devPtr;
    }

    uint8_t* AllocZero(uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        (void)devMemMgr_->Memset(devPtr, size, 0, size);
        return devPtr;
    }

    uint8_t* CopyToDev(uint8_t* data, uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        (void)devMemMgr_->Memcpy(devPtr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
        return devPtr;
    }

    void CopyToDev(uint8_t* devPtr, uint8_t* data, uint64_t size)
    {
        (void)devMemMgr_->Memcpy(devPtr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
    }

    template <typename T>
    T* CopyToDev(std::vector<T> data, uint8_t** cachedDevAddrHolder)
    {
        return (T*)CopyToDev((uint8_t*)data.data(), data.size() * sizeof(T), cachedDevAddrHolder);
    }

    void CopyFromDev(uint8_t* data, uint8_t* devPtr, uint64_t size)
    {
        (void)devMemMgr_->Memcpy(data, size, devPtr, size, RT_MEMCPY_DEVICE_TO_HOST);
    }

    uint8_t* CopyToDev(RawTensorData& data)
    {
        if (data.GetDevPtr() == nullptr) {
            auto* devPtr = reinterpret_cast<uint8_t*>(devMemMgr_->Alloc(data.size()));
            if (devPtr == nullptr) {
                return nullptr;
            }
            (void)devMemMgr_->Memcpy(devPtr, data.size(), (uint8_t*)data.data(), data.size(), RT_MEMCPY_HOST_TO_DEVICE);
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData& data) { CopyFromDev(data.data(), data.GetDevPtr(), data.size()); }

    void Free(uint8_t* mem)
    {
        if (mem && (!isUseHugePage_)) {
            (void)devMemMgr_->Free(mem);
        }
    }

    uint64_t GetL2Offset() { return machine::GetRA()->GetL2Offset(); }

    bool isUseHugePage_{true};
    std::shared_ptr<DeviceMemoryManager> devMemMgr_{nullptr};
};
} // namespace npu::tile_fwk::dynamic
#endif
