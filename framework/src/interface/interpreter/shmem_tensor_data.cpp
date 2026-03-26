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
 * \file shmem_tensor_data.cpp
 * \brief 共享内存Tensor数据管理实现
 */

#include "shmem_tensor_data.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <algorithm>

namespace npu::tile_fwk {

std::shared_ptr<ShmemTensorData> ShmemTensorData::CreateShared(
    const std::string &shmName, size_t size, int rankId, bool isCreator) {

    auto shmemData = std::make_shared<ShmemTensorData>();
    shmemData->shmName_ = shmName;
    shmemData->size_ = size;
    shmemData->rankId_ = rankId;
    shmemData->isCreator_ = isCreator;

    int flags = O_RDWR;
    if (isCreator) {
        flags |= O_CREAT | O_EXCL;
    }

    shmemData->shmFd_ = shm_open(shmName.c_str(), flags, 0666);
    if (shmemData->shmFd_ == -1) {
        if (isCreator && errno == EEXIST) {
            shmemData->shmFd_ = shm_open(shmName.c_str(), O_RDWR, 0666);
        }
        if (shmemData->shmFd_ == -1) {
            throw std::runtime_error("Failed to open shared memory: " + shmName);
        }
    }

    if (isCreator) {
        if (ftruncate(shmemData->shmFd_, size) == -1) {
            close(shmemData->shmFd_);
            throw std::runtime_error("Failed to set shared memory size");
        }
    }

    shmemData->dataPtr_ = static_cast<uint8_t*>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, shmemData->shmFd_, 0)
    );

    if (shmemData->dataPtr_ == MAP_FAILED) {
        close(shmemData->shmFd_);
        if (isCreator) {
            shm_unlink(shmName.c_str());
        }
        throw std::runtime_error("Failed to mmap shared memory");
    }

    return shmemData;
}

ShmemTensorData::~ShmemTensorData() {
    if (dataPtr_ != nullptr && dataPtr_ != MAP_FAILED) {
        munmap(dataPtr_, size_);
    }
    if (shmFd_ != -1) {
        close(shmFd_);
    }
    if (isCreator_) {
        shm_unlink(shmName_.c_str());
    }
}

void ShmemTensorData::Set(const std::vector<int64_t> &offset,
                          const void *data, size_t size) {
    size_t byteOffset = CalculateByteOffset(offset);
    if (byteOffset + size > size_) {
        throw std::runtime_error("Shared memory write out of bounds");
    }
    std::memcpy(dataPtr_ + byteOffset, data, size);
}

void ShmemTensorData::Add(const std::vector<int64_t> &offset,
                          const void *data, size_t size) {
    size_t byteOffset = CalculateByteOffset(offset);
    if (byteOffset + size > size_) {
        throw std::runtime_error("Shared memory write out of bounds");
    }

    std::atomic<int32_t> *atomicPtr = reinterpret_cast<std::atomic<int32_t>*>(
        dataPtr_ + byteOffset);
    const int32_t *valuePtr = static_cast<const int32_t*>(data);
    size_t count = size / sizeof(int32_t);

    for (size_t i = 0; i < count; i++) {
        atomicPtr[i].fetch_add(valuePtr[i], std::memory_order_relaxed);
    }
}

void ShmemTensorData::Get(const std::vector<int64_t> &offset,
                          void *data, size_t size) {
    size_t byteOffset = CalculateByteOffset(offset);
    if (byteOffset + size > size_) {
        throw std::runtime_error("Shared memory read out of bounds");
    }
    std::memcpy(data, dataPtr_ + byteOffset, size);
}

void ShmemTensorData::Signal(const std::vector<int64_t> &offset, int32_t value) {
    size_t byteOffset = CalculateByteOffset(offset);
    if (byteOffset + sizeof(int32_t) > size_) {
        throw std::runtime_error("Signal out of bounds");
    }

    std::atomic<int32_t> *signalPtr = reinterpret_cast<std::atomic<int32_t>*>(
        dataPtr_ + byteOffset);
    signalPtr->store(value, std::memory_order_release);
}

void ShmemTensorData::WaitUntil(const std::vector<int64_t> &offset,
                                 int32_t expected) {
    size_t byteOffset = CalculateByteOffset(offset);
    if (byteOffset + sizeof(int32_t) > size_) {
        throw std::runtime_error("Wait out of bounds");
    }

    std::atomic<int32_t> *signalPtr = reinterpret_cast<std::atomic<int32_t>*>(
        dataPtr_ + byteOffset);

    int spinCount = 0;
    while (signalPtr->load(std::memory_order_acquire) != expected) {
        if (++spinCount % 1000 == 0) {
            std::this_thread::yield();
        }
    }
}

size_t ShmemTensorData::CalculateByteOffset(const std::vector<int64_t> &offset) {
    if (offset.empty()) {
        return 0;
    }

    size_t result = 0;
    for (size_t i = 0; i < offset.size(); i++) {
        result = result * 1 + static_cast<size_t>(offset[i]);
    }
    return result * 4;
}

} // namespace npu::tile_fwk
