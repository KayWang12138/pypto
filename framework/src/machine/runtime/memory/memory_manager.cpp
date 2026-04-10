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
 * \file memory_manager.cpp
 * \brief unified runtime memory manager abstraction
 */

#include "machine/runtime/memory/memory_manager.h"

#include <cstdlib>
#include "securec.h"

#ifdef BUILD_WITH_CANN
#include "acl/acl_rt.h"
#include "machine/runtime/runtime.h"
#endif

namespace npu::tile_fwk::dynamic {

namespace {
constexpr int kMemOpSuccess = 0;
constexpr int kMemOpFail = -1;
} // namespace

HostMemoryManager::~HostMemoryManager() { ReleaseAll(); }

void* HostMemoryManager::Alloc(size_t size)
{
    if (size == 0) {
        return nullptr;
    }
    auto* ptr = std::malloc(size);
    if (ptr != nullptr) {
        allocated_.insert(ptr);
    }
    return ptr;
}

int HostMemoryManager::Free(void* ptr)
{
    if (ptr == nullptr) {
        return kMemOpSuccess;
    }
    auto it = allocated_.find(ptr);
    if (it != allocated_.end()) {
        std::free(ptr);
        allocated_.erase(it);
        return kMemOpSuccess;
    }
    return kMemOpFail;
}

int HostMemoryManager::Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind)
{
    (void)kind;
    if (dst == nullptr || src == nullptr || dstSize < size) {
        return kMemOpFail;
    }
    return memcpy_s(dst, dstSize, src, size);
}

int HostMemoryManager::Memset(void* dst, size_t dstSize, int value, size_t size)
{
    if (dst == nullptr || dstSize < size) {
        return kMemOpFail;
    }
    return memset_s(dst, dstSize, value, size);
}

void HostMemoryManager::ReleaseAll()
{
    for (auto* ptr : allocated_) {
        std::free(ptr);
    }
    allocated_.clear();
}

void* DeviceMemoryManager::Alloc(size_t size)
{
    if (size == 0) {
        return nullptr;
    }
#ifdef BUILD_WITH_CANN
    uint8_t* devPtr = nullptr;
    if (useHugePage_) {
        machine::GetRA()->AllocDevAddr(&devPtr, size);
        return devPtr;
    }
    auto ret = rtMalloc(reinterpret_cast<void**>(&devPtr), size, RT_MEMORY_HBM, 0);
    return ret == RT_ERROR_NONE ? devPtr : nullptr;
#else
    return std::malloc(size);
#endif
}

int DeviceMemoryManager::Free(void* ptr)
{
    if (ptr == nullptr) {
        return kMemOpSuccess;
    }
#ifdef BUILD_WITH_CANN
    if (useHugePage_) {
        return kMemOpSuccess;
    }
    auto ret = rtFree(ptr);
    return ret == RT_ERROR_NONE ? kMemOpSuccess : kMemOpFail;
#else
    std::free(ptr);
    return kMemOpSuccess;
#endif
}

int DeviceMemoryManager::Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind)
{
    if (dst == nullptr || src == nullptr || dstSize < size) {
        return kMemOpFail;
    }
#ifdef BUILD_WITH_CANN
    auto ret = rtMemcpy(dst, dstSize, src, size, static_cast<rtMemcpyKind_t>(kind));
    return ret == RT_ERROR_NONE ? kMemOpSuccess : kMemOpFail;
#else
    (void)kind;
    return memcpy_s(dst, dstSize, src, size);
#endif
}

int DeviceMemoryManager::Memset(void* dst, size_t dstSize, int value, size_t size)
{
    if (dst == nullptr || dstSize < size) {
        return kMemOpFail;
    }
#ifdef BUILD_WITH_CANN
    auto ret = rtMemset(dst, dstSize, value, size);
    return ret == RT_ERROR_NONE ? kMemOpSuccess : kMemOpFail;
#else
    return memset_s(dst, dstSize, value, size);
#endif
}

std::unique_ptr<IMemoryManager> MemoryManagerFactory::Create(MemoryMode mode, bool useHugePage)
{
    if (mode == MemoryMode::DEVICE) {
        return std::make_unique<DeviceMemoryManager>(useHugePage);
    }
    return std::make_unique<HostMemoryManager>();
}

} // namespace npu::tile_fwk::dynamic

