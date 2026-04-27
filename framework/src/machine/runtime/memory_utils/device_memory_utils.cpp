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
 * \file device_memory_utils.cpp
 * \brief
 */

#include "device_memory_utils.h"
#include "machine/runtime/runtime.h"

namespace npu::tile_fwk::dynamic {

DeviceMemoryUtils::DeviceMemoryUtils(bool isHugePage)
{
    isUseHugePage_ = isHugePage;
}

bool DeviceMemoryUtils::IsDevice()
{
    return true;
}

uint8_t* DeviceMemoryUtils::DoAlloc(size_t size)
{
    uint8_t* devPtr = nullptr;
    if (isUseHugePage_) {
        machine::GetRA()->AllocDevAddr(&devPtr, size);
    } else {
        RuntimeMalloc(reinterpret_cast<void**>(&devPtr), size, RT_MEMORY_HBM, 0);
    }
    return devPtr;
}

void DeviceMemoryUtils::DoFree(uint8_t* ptr)
{
    if (!isUseHugePage_) {
        RuntimeFree(ptr);
    }
    // hugePage mode: managed by DevMemoryPool, no individual free
}

void DeviceMemoryUtils::DoMemcpyH2D(uint8_t* dst, uint8_t* src, size_t size)
{
    RuntimeMemcpy(dst, size, src, size, RtMemcpyKind::HOST_TO_DEVICE);
}

void DeviceMemoryUtils::DoMemcpyD2H(uint8_t* dst, uint8_t* src, size_t size)
{
    RuntimeMemcpy(dst, size, src, size, RtMemcpyKind::DEVICE_TO_HOST);
}

void DeviceMemoryUtils::DoMemset(uint8_t* ptr, size_t size)
{
    RuntimeMemset(ptr, size, 0, size);
}

uint64_t DeviceMemoryUtils::GetL2OffsetImpl()
{
    return machine::GetRA()->GetL2Offset();
}

void DeviceMemoryUtils::FreeTensor(uint8_t* devAddr)
{
    machine::GetRA()->FreeTensor(devAddr);
}

} // namespace npu::tile_fwk::dynamic
