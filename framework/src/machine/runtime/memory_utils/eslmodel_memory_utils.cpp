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
 * \file eslmodel_memory_utils.cpp
 * \brief Memory utilities for ESL model (mmap-based device emulation)
 */

#include "eslmodel_memory_utils.h"
#include <sys/mman.h>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include "machine/runtime/runtime.h"

namespace npu::tile_fwk::dynamic {

// MmapGlobalManager
std::vector<MmapRecord> MmapGlobalManager::records_;
std::mutex MmapGlobalManager::mutex_;

void MmapGlobalManager::AddRecord(void* addr, size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back({addr, size});
}

void MmapGlobalManager::UnmapAll()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& rec : records_) {
        if (rec.addr != nullptr && rec.addr != MAP_FAILED) {
            munmap(rec.addr, rec.size);
        }
    }
    records_.clear();
}

// EslModelMemoryUtils
EslModelMemoryUtils::EslModelMemoryUtils(bool isHugePage)
{
    isUseHugePage_ = isHugePage;
}

bool EslModelMemoryUtils::IsDevice()
{
    return false;
}

void EslModelMemoryUtils::UnmapAllMappings()
{
    MmapGlobalManager::UnmapAll();
}

uint8_t* EslModelMemoryUtils::DoAlloc(size_t size)
{
    uint8_t* devPtr = nullptr;
    if (isUseHugePage_) {
        machine::GetRA()->AllocDevAddr(&devPtr, size);
    } else {
        RuntimeMalloc(reinterpret_cast<void**>(&devPtr), size, RT_MEMORY_HBM, 0);
    }
    if (devPtr != nullptr) {
        MapEslAddrToHostAddr(reinterpret_cast<uintptr_t>(devPtr), size);
    }
    return devPtr;
}

void EslModelMemoryUtils::DoFree(uint8_t* ptr)
{
    if (ptr != nullptr && !isUseHugePage_) {
        RuntimeFree(ptr);
    }
}

void EslModelMemoryUtils::DoMemcpyH2D(uint8_t* dst, uint8_t* src, size_t size)
{
    RuntimeMemcpy(dst, size, src, size, RtMemcpyKind::HOST_TO_DEVICE);
    MemCopytoMapAddr(dst, src, size);
}

void EslModelMemoryUtils::DoMemcpyD2H(uint8_t* dst, uint8_t* src, size_t size)
{
    RuntimeMemcpy(dst, size, src, size, RtMemcpyKind::DEVICE_TO_HOST);
}

void EslModelMemoryUtils::DoMemset(uint8_t* ptr, size_t size)
{
    RuntimeMemset(ptr, size, 0, size);
}

uint64_t EslModelMemoryUtils::GetL2OffsetImpl()
{
    return machine::GetRA()->GetL2Offset();
}

void EslModelMemoryUtils::FreeTensor(uint8_t* devAddr)
{
    machine::GetRA()->FreeTensor(devAddr);
}

uintptr_t EslModelMemoryUtils::AlignAddress(uintptr_t addr, size_t size, bool alignUp)
{
    if (size == 0) {
        return addr;
    }
    if (alignUp) {
        return ((addr + size - 1) / size) * size;
    }
    return (addr / size) * size;
}

void* EslModelMemoryUtils::MapEslAddrToHostAddr(uintptr_t eslAddr, uintptr_t size)
{
    long pageSize = sysconf(_SC_PAGESIZE);
    auto alignSize = AlignAddress(size, pageSize, true) + pageSize;
    void* hostAddr = mmap(
        reinterpret_cast<void*>(AlignAddress(eslAddr, pageSize, false)),
        alignSize,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
        -1,
        0);
    if (hostAddr == MAP_FAILED) {
        perror("mmap failed");
        fprintf(stderr, "Failed to map ESL address 0x%lx, size: %zu\n", eslAddr, size);
        return MAP_FAILED;
    }
    MmapGlobalManager::AddRecord(hostAddr, alignSize);
    return hostAddr;
}

void EslModelMemoryUtils::MemCopytoMapAddr(uint8_t* dst, uint8_t* src, uintptr_t size)
{
    errno_t result = memcpy_s(dst, size, src, size);
    if (result != 0) {
        std::cerr << "Memory copy failed with error code: " << result << std::endl;
    }
}

} // namespace npu::tile_fwk::dynamic
