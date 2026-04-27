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
 * \file aikernel_runtime.h
 * \brief
 */

#ifndef AIKERNEL_RUNTIME_H
#define AIKERNEL_RUNTIME_H

#include "tilefwk/aikernel_data.h"
#include "tilefwk/error_code.h"

#ifdef __DEVICE__
#include "machine/utils/device_log.h"
#endif

constexpr int MAIN_BLOCK_INDEX = 1;

constexpr uint64_t NSEC_PER_SEC = 1000000000ULL;

#ifdef __DAV_V310
    constexpr uint64_t SYNC_TIMEOUT = 960000000000ULL;
#else
    constexpr uint64_t SYNC_TIMEOUT = 48000000000ULL;
#endif

constexpr uint64_t SYNC_WARN_INTERVAL = SYNC_TIMEOUT / 4ULL;

__always_inline uint64_t GetCycles()
{
#if defined(__aarch64__) && defined(__DEVICE__)
    uint64_t cycles;
    asm volatile("mrs %0, cntvct_el0" : "=r"(cycles));
    return cycles;
#else
    return 0U;
#endif
}

__always_inline uint64_t GetFreq()
{
#if defined(__aarch64__) && defined(__DEVICE__)
    uint64_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
#else
    return NSEC_PER_SEC;
#endif
}

__always_inline void WaitAicoreStart([[maybe_unused]] npu::tile_fwk::DevStartArgsBase* startArgs)
{
#if defined(__aarch64__) && defined(__DEVICE__)
    using namespace npu::tile_fwk;
    
    uint64_t start = GetCycles();
    uint64_t lastWarnCycles = 0;
    
    while (startArgs->syncFlag != 1) {
        uint64_t elapsedCycles = GetCycles() - start;
        
        if (elapsedCycles > SYNC_TIMEOUT) {
            uint64_t elapsedSec = (elapsedCycles * NSEC_PER_SEC) / GetFreq();
            DEV_ERROR(SchedErr::TASK_WAIT_TIMEOUT,
                      "#aicore.start: WaitAicoreStart timeout, elapsed %lu sec.", elapsedSec);
            break;
        }
        
        if (elapsedCycles > lastWarnCycles + SYNC_WARN_INTERVAL) {
            uint64_t elapsedSec = (elapsedCycles * NSEC_PER_SEC) / GetFreq();
            DEV_WARN("#aicore.start: WaitAicoreStart still waiting, elapsed %lu sec.", elapsedSec);
            lastWarnCycles = elapsedCycles;
        }
    }
#endif
}

#ifdef __TILE_FWK_AICPU__

#define RuntimeGetStartArgs() startArgs
#define RuntimeGetSymbol(idx) (symbolTable[idx])

#else

#define RuntimeGetStartArgs() AiCoreRuntimeGetStartArgs(param)
#define RuntimeGetSymbol(idx) \
    (param->exprTbl[(idx) + MAIN_BLOCK_INDEX]) // inserted the mainBlock expression at position 0 of the expressionSet

#endif

#define RuntimeGetInputShapeDimSize(input) ((input)->shape.dimSize)
#define RuntimeGetInputShapeDim(input, n) ((input)->shape.dim[(n)])
#define RUNTIME_GetInputShapeDimSize(inputIndex) \
    RuntimeGetInputShapeDimSize(&(RuntimeGetStartArgs())->devTensorList[(inputIndex)])
#define RUNTIME_GetInputShapeDim(inputIndex, n) \
    RuntimeGetInputShapeDim(&(RuntimeGetStartArgs())->devTensorList[(inputIndex)], (n))

#define RUNTIME_int8_t int8_t
#define RUNTIME_int16_t int16_t
#define RUNTIME_int32_t int32_t
#define RUNTIME_int64_t int64_t
#define RUNTIME_uint8_t uint8_t
#define RUNTIME_uint16_t uint16_t
#define RUNTIME_uint32_t uint32_t
#define RUNTIME_uint64_t uint64_t
#define RUNTIME_bool int8_t

template <typename T>
INLINE T RuntimeGetInputData(__gm__ npu::tile_fwk::DevTensorData* input, int64_t off0)
{
    return ((T*)input->address)[off0];
}

template <typename T>
INLINE T RuntimeGetInputData(__gm__ npu::tile_fwk::DevTensorData* input, int64_t off0, int64_t off1)
{
    int64_t off = input->shape.dim[1] * off0 + off1;
    return ((T*)input->address)[off];
}

template <typename T>
INLINE T RuntimeGetInputData(__gm__ npu::tile_fwk::DevTensorData* input, int64_t off0, int64_t off1, int64_t off2)
{
    int64_t off = input->shape.dim[1] * off0 + off1;
    off = off * input->shape.dim[2] + off2;
    return ((T*)input->address)[off];
}

template <typename T>
INLINE T
RuntimeGetInputData(__gm__ npu::tile_fwk::DevTensorData* input, int64_t off0, int64_t off1, int64_t off2, int64_t off3)
{
    int64_t off = input->shape.dim[1] * off0 + off1;
    off = off * input->shape.dim[2] + off2;
    off = off * input->shape.dim[3] + off3;
    return ((T*)input->address)[off];
}

#define RUNTIME_GetInputData(index, dtype, ...) \
    RuntimeGetInputData<dtype>(&(RuntimeGetStartArgs())->devTensorList[(index)], __VA_ARGS__)

#define RUNTIME_GetSymbol(idx) RuntimeGetSymbol(idx)

#endif
