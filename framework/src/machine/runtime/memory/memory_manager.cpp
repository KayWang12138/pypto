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
#include "machine/runtime/rt_api/machine_rt_api.h"

#ifdef BUILD_WITH_CANN
#include "acl/acl_rt.h"
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
    void* ptr = nullptr;
    int rc = RtApiDispatcher::Current().Malloc(&ptr, size, useHugePage_);
    return rc == kMemOpSuccess ? ptr : nullptr;
}

int DeviceMemoryManager::Free(void* ptr)
{
    if (ptr == nullptr) {
        return kMemOpSuccess;
    }
    return RtApiDispatcher::Current().Free(ptr, useHugePage_) == kMemOpSuccess ? kMemOpSuccess : kMemOpFail;
}

int DeviceMemoryManager::Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind)
{
    if (dst == nullptr || src == nullptr || dstSize < size) {
        return kMemOpFail;
    }
    return RtApiDispatcher::Current().Memcpy(dst, dstSize, src, size, kind) == kMemOpSuccess ? kMemOpSuccess :
                                                                                                 kMemOpFail;
}

int DeviceMemoryManager::Memset(void* dst, size_t dstSize, int value, size_t size)
{
    if (dst == nullptr || dstSize < size) {
        return kMemOpFail;
    }
    return RtApiDispatcher::Current().Memset(dst, dstSize, value, size) == kMemOpSuccess ? kMemOpSuccess :
                                                                                             kMemOpFail;
}

std::unique_ptr<IMemoryManager> MemoryManagerFactory::Create(MemoryMode mode, bool useHugePage)
{
    if (mode == MemoryMode::DEVICE) {
        return std::make_unique<DeviceMemoryManager>(useHugePage);
    }
    return std::make_unique<HostMemoryManager>();
}

} // namespace npu::tile_fwk::dynamic
