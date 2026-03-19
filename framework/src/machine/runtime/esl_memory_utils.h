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

#ifdef __ESL_SIMULATION__
#ifdef BUILD_WITH_CANN

#include <sys/mman.h>
#include "machine/runtime/runtime.h"
#include "machine/runtime/device_runner.h"
#include "machine/platform/platform_manager.h"
#include "interface/interpreter/raw_tensor_data.h"

namespace npu::tile_fwk::dynamic {
struct EslModelMemoryUtils {
    EslModelMemoryUtils(bool isHugePage = true) :isUseHugePage_(isHugePage) {}
    static bool IsDevice() { return true; }
    uint8_t *AllocDev(size_t size, uint8_t **cachedDevAddrHolder) {
        uint8_t *devPtr = nullptr;
        if (cachedDevAddrHolder == nullptr) {
            if (isUseHugePage_) {
                machine::GetRA()->AllocDevAddr(&devPtr, size);
            } else {
                rtMalloc((void**)&devPtr, size, RT_MEMORY_HBM, 0);
            }
        } else if (*cachedDevAddrHolder == nullptr) {
            if (isUseHugePage_) {
                machine::GetRA()->AllocDevAddr(&devPtr, size);
            } else {
                rtMalloc((void**)&devPtr, size, RT_MEMORY_HBM, 0);
            }
            *cachedDevAddrHolder = devPtr;
        } else {
            devPtr = *cachedDevAddrHolder;
        }
        MapEslAddrToHostAddr(reinterpret_cast<uintptr_t>(devPtr), size);
        return devPtr;
    }

    uint8_t *AllocZero(uint64_t size, uint8_t **cachedDevAddrHolder) {
        uint8_t *devPtr = AllocDev(size, cachedDevAddrHolder);
        (void)rtMemset(devPtr, size, 0, size);
        return devPtr;
    }

    uint8_t *CopyToDev(uint8_t *data, uint64_t size, uint8_t **cachedDevAddrHolder) {
        uint8_t *devPtr = AllocDev(size, cachedDevAddrHolder);
        rtMemcpy(devPtr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
        MemCopytoMapAddr(devPtr, data, size);
        return devPtr;
    }

    void CopyToDev(uint8_t *devPtr, uint8_t *data, uint64_t size) {
        rtMemcpy(devPtr, size, data, size, RT_MEMCPY_HOST_TO_DEVICE);
    }

    template <typename T>
    T *CopyToDev(std::vector<T> data, uint8_t **cachedDevAddrHolder) {
        return (T *)CopyToDev((uint8_t *)data.data(), data.size() * sizeof(T), cachedDevAddrHolder);
    }

    void CopyFromDev(uint8_t *data, uint8_t *devPtr, uint64_t size) {
        rtMemcpy(data, size, devPtr, size, RT_MEMCPY_DEVICE_TO_HOST);
    }

    uint8_t *CopyToDev(RawTensorData &data) {
        if (data.GetDevPtr() == nullptr) {
            uint8_t *devPtr = nullptr;
            machine::GetRA()->AllocDevAddr(&devPtr, data.size(), true);
            if (devPtr == nullptr) {
                return nullptr;
            }
            MapEslAddrToHostAddr(reinterpret_cast<uintptr_t>(devPtr), data.size());
            rtMemcpy(devPtr, data.size(), (uint8_t *)data.data(), data.size(), RT_MEMCPY_HOST_TO_DEVICE);
            MemCopytoMapAddr(devPtr, (uint8_t *)data.data(), data.size());
            data.SetDevPtr(devPtr);
        }
        return data.GetDevPtr();
    }

    void CopyFromDev(RawTensorData &data) {
        CopyFromDev(data.data(), data.GetDevPtr(), data.size());
    }

    void Free(uint8_t* mem) {
        if (mem && (!isUseHugePage_)) {
            rtFree(mem);
        }
    }

    void FreeTensor(uint8_t *devAddr) {
        machine::GetRA()->FreeTensor(devAddr);
    }

    uint64_t GetL2Offset() {
        return machine::GetRA()->GetL2Offset();
    }

    bool isUseHugePage_{true};

    uintptr_t AlignAddress(uintptr_t addr, size_t size, bool alignUp) {
        if (size == 0) {
            return addr;
        }
        if (alignUp) {
            return ((addr + size - 1) / size) * size;
        }
        return (addr / size) * size;
    }

    void* MapEslAddrToHostAddr(uintptr_t eslAddr, uintptr_t size) {
        long pageSize = sysconf(_SC_PAGESIZE);

        // todo : workspace size is 0
        void *hostAddr = mmap(
            (void *) AlignAddress(eslAddr, pageSize, false),
            AlignAddress(size, pageSize, true) + pageSize,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
            -1,
            0
        );
        return hostAddr;
    }

    void MemCopytoMapAddr(uint8_t *dst, uint8_t *src, uintptr_t size) {
        memcpy_s(dst, size, src, size);
    }
};
}
#endif
#endif // __ESL_SIMULATION__