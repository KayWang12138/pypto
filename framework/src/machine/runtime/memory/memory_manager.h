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
 * \file memory_manager.h
 * \brief unified runtime memory manager abstraction
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>

namespace npu::tile_fwk::dynamic {

enum class MemoryMode {
    HOST = 0,
    DEVICE = 1,
};

class IMemoryManager {
public:
    virtual ~IMemoryManager() = default;
    virtual void* Alloc(size_t size) = 0;
    virtual int Free(void* ptr) = 0;
    virtual int Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind) = 0;
    virtual int Memset(void* dst, size_t dstSize, int value, size_t size) = 0;
    virtual bool IsDevice() const = 0;
};

class HostMemoryManager : public IMemoryManager {
public:
    HostMemoryManager() = default;
    ~HostMemoryManager() override;

    void* Alloc(size_t size) override;
    int Free(void* ptr) override;
    int Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind) override;
    int Memset(void* dst, size_t dstSize, int value, size_t size) override;
    bool IsDevice() const override { return false; }

    void ReleaseAll();

private:
    std::unordered_set<void*> allocated_;
};

class DeviceMemoryManager : public IMemoryManager {
public:
    explicit DeviceMemoryManager(bool useHugePage = true) : useHugePage_(useHugePage) {}
    ~DeviceMemoryManager() override = default;

    void* Alloc(size_t size) override;
    int Free(void* ptr) override;
    int Memcpy(void* dst, size_t dstSize, const void* src, size_t size, int kind) override;
    int Memset(void* dst, size_t dstSize, int value, size_t size) override;
    bool IsDevice() const override { return true; }

private:
    bool useHugePage_{true};
};

class MemoryManagerFactory {
public:
    static std::unique_ptr<IMemoryManager> Create(MemoryMode mode, bool useHugePage = true);
};

} // namespace npu::tile_fwk::dynamic

