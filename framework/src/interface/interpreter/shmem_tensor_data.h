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
 * \file shmem_tensor_data.h
 * \brief 共享内存Tensor数据管理，用于多进程间通信
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <sys/mman.h>

namespace npu::tile_fwk {

class ShmemTensorData {
public:
    ShmemTensorData() = default;
    ~ShmemTensorData();

    static std::shared_ptr<ShmemTensorData> CreateShared(
        const std::string &shmName,
        size_t size,
        int rankId,
        bool isCreator);

    void Set(const std::vector<int64_t> &offset, const void *data, size_t size);
    void Add(const std::vector<int64_t> &offset, const void *data, size_t size);
    void Get(const std::vector<int64_t> &offset, void *data, size_t size);
    void Signal(const std::vector<int64_t> &offset, int32_t value);
    void WaitUntil(const std::vector<int64_t> &offset, int32_t expected);

    uint8_t* GetData() { return dataPtr_; }
    size_t GetSize() { return size_; }
    int GetRankId() { return rankId_; }
    bool IsCreator() { return isCreator_; }

private:

    size_t CalculateByteOffset(const std::vector<int64_t> &offset);

    std::string shmName_;
    int shmFd_{-1};
    uint8_t *dataPtr_{nullptr};
    size_t size_{0};
    int rankId_{0};
    bool isCreator_{false};
};

} // namespace npu::tile_fwk
