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
 * \file emulation_memory_utils.cpp
 * \brief Memory utilities for emulation mode (host-side malloc)
 */

#include "emulation_memory_utils.h"
#include "tilefwk/error_code.h"
#include "tilefwk/pypto_fwk_log.h"

namespace npu::tile_fwk::dynamic {

EmulationMemoryUtils::EmulationMemoryUtils() = default;
EmulationMemoryUtils::~EmulationMemoryUtils() = default;

bool EmulationMemoryUtils::IsDevice()
{
    return false;
}

uint8_t* EmulationMemoryUtils::DoAlloc(size_t size)
{
    if (size == 0 || size > 0x500000000) {
        MACHINE_LOGE(DevCommonErr::PARAM_INVALID, "AllocDev failed: size=%zu", size);
        return nullptr;
    }
    uint8_t* rawPtr = reinterpret_cast<uint8_t*>(malloc(size));
    if (rawPtr == nullptr) {
        return nullptr;
    }
    EmulationAllocatePtrs_.push_back(std::shared_ptr<uint8_t>(rawPtr, free));
    return rawPtr;
}

void EmulationMemoryUtils::DoFree(uint8_t* ptr)
{
    // shared_ptr auto-manages lifetime, no manual free needed
    (void)ptr;
}

void EmulationMemoryUtils::DoMemcpyH2D(uint8_t* dst, uint8_t* src, size_t size)
{
    memcpy_s(dst, size, src, size);
}

void EmulationMemoryUtils::DoMemcpyD2H(uint8_t* dst, uint8_t* src, size_t size)
{
    memcpy_s(dst, size, src, size);
}

void EmulationMemoryUtils::DoMemset(uint8_t* ptr, size_t size)
{
    memset_s(ptr, size, 0, size);
}

uint64_t EmulationMemoryUtils::GetL2OffsetImpl()
{
    return 0;
}

} // namespace npu::tile_fwk::dynamic
