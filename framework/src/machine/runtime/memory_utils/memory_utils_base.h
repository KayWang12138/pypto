/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file memory_utils_base.h
 * \brief CRTP base for device/emulation/eslmodel memory utils
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>
#include "interface/interpreter/raw_tensor_data.h"
#include "tilefwk/pypto_fwk_log.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk::dynamic {

// CRTP: Curiously Recurring Template Pattern
// Derived must provide:
//   - static bool IsDevice();
//   - uint8_t* DoAlloc(size_t size);
//   - void DoFree(uint8_t* ptr);
//   - void DoMemcpyH2D(uint8_t* dst, uint8_t* src, size_t size);
//   - void DoMemcpyD2H(uint8_t* dst, uint8_t* src, size_t size);
//   - void DoMemset(uint8_t* ptr, size_t size);
//   - uint64_t GetL2OffsetImpl();
template <typename Derived>
class MemoryUtilsBase {
public:
    uint8_t* AllocDev(size_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = nullptr;
        const bool needAlloc = (cachedDevAddrHolder == nullptr) || (*cachedDevAddrHolder == nullptr);
        if (needAlloc) {
            devPtr = static_cast<Derived*>(this)->DoAlloc(size);
            if (cachedDevAddrHolder != nullptr && *cachedDevAddrHolder == nullptr) {
                *cachedDevAddrHolder = devPtr;
            }
        } else {
            devPtr = *cachedDevAddrHolder;
        }
        return devPtr;
    }

    uint8_t* AllocZero(uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        if (devPtr != nullptr) {
            static_cast<Derived*>(this)->DoMemset(devPtr, size);
        }
        return devPtr;
    }

    uint8_t* CopyToDev(uint8_t* data, uint64_t size, uint8_t** cachedDevAddrHolder)
    {
        uint8_t* devPtr = AllocDev(size, cachedDevAddrHolder);
        if (devPtr != nullptr && data != nullptr) {
            static_cast<Derived*>(this)->DoMemcpyH2D(devPtr, data, size);
        }
        return devPtr;
    }

    template <typename T>
    T* CopyToDev(std::vector<T> data, uint8_t** cachedDevAddrHolder)
    {
        return reinterpret_cast<T*>(CopyToDev(
            reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(T), cachedDevAddrHolder));
    }

    void CopyToDev(uint8_t* devPtr, uint8_t* data, uint64_t size)
    {
        if (devPtr != nullptr && data != nullptr) {
            static_cast<Derived*>(this)->DoMemcpyH2D(devPtr, data, size);
        }
    }

    void CopyFromDev(uint8_t* data, uint8_t* devPtr, uint64_t size)
    {
        if (data != nullptr && devPtr != nullptr) {
            static_cast<Derived*>(this)->DoMemcpyD2H(data, devPtr, size);
        }
    }

    uint8_t* CopyToDev(RawTensorData& data)
    {
        if (data.GetDevPtr() == nullptr) {
            auto* derived = static_cast<Derived*>(this);
            uint8_t* devPtr = derived->DoAlloc(data.size());
            if (devPtr == nullptr) {
                return nullptr;
            }
            derived->DoMemcpyH2D(devPtr, reinterpret_cast<uint8_t*>(data.data()), data.size());
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData& data)
    {
        CopyFromDev(data.data(), data.GetDevPtr(), data.size());
    }

    uint64_t GetL2Offset()
    {
        return static_cast<Derived*>(this)->GetL2OffsetImpl();
    }

    void Free(uint8_t* mem)
    {
        if (mem != nullptr) {
            static_cast<Derived*>(this)->DoFree(mem);
        }
    }

protected:
    bool isUseHugePage_ = true;
};

} // namespace npu::tile_fwk::dynamic
