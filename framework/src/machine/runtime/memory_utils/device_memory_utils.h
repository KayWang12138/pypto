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
 * \file device_memory_utils.h
 * \brief
 */

#pragma once

#include "memory_utils_base.h"

namespace npu::tile_fwk::dynamic {

class DeviceMemoryUtils : MemoryUtilsBase<DeviceMemoryUtils> {
public:
    using MemoryUtilsBase::AllocDev;
    using MemoryUtilsBase::AllocZero;
    using MemoryUtilsBase::CopyToDev;
    using MemoryUtilsBase::CopyFromDev;
    using MemoryUtilsBase::GetL2Offset;
    using MemoryUtilsBase::Free;

    DeviceMemoryUtils(bool isHugePage = true);

    static bool IsDevice();

    uint8_t* DoAlloc(size_t size);
    void DoFree(uint8_t* ptr);
    void DoMemcpyH2D(uint8_t* dst, uint8_t* src, size_t size);
    void DoMemcpyD2H(uint8_t* dst, uint8_t* src, size_t size);
    void DoMemset(uint8_t* ptr, size_t size);
    uint64_t GetL2OffsetImpl();

    // DeviceMemoryUtils-specific: free tensor via RuntimeAgent
    void FreeTensor(uint8_t* devAddr);
};

} // namespace npu::tile_fwk::dynamic
