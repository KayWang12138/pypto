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

constexpr int MAIN_BLOCK_INDEX = 1;
constexpr uint64_t SYNC_TIMEOUT = 48000000000;

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

__always_inline void WaitAicoreStart([[maybe_unused]] npu::tile_fwk::DevStartArgsBase* startArgs)
{
#if defined(__aarch64__) && defined(__DEVICE__)
    uint64_t start = GetCycles();
    while (startArgs->syncFlag != 1) {
        if (GetCycles() - start > SYNC_TIMEOUT) {
            break;
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

#define RuntimeDType1 int8_t    // DT_INT8
#define RuntimeDType2 int16_t   // DT_INT16
#define RuntimeDType3 int32_t   // DT_INT32
#define RuntimeDType4 int64_t   // DT_INT64
#define RuntimeDType11 uint8_t  // DT_UINT8
#define RuntimeDType12 uint16_t // DT_UINT16
#define RuntimeDType13 uint32_t // DT_UINT32
#define RuntimeDType14 uint64_t // DT_UINT64
#define RuntimeDType15 int8_t   // DT_BOOL

#define RuntimeDTypeX(x) RuntimeDType##x

template <typename T>
INLINE T RUNTIMEGetInputData(npu::tile_fwk::DevTensorData* input, int64_t off0)
{
    return ((T*)input->address)[off0];
}

template <typename T>
INLINE T RUNTIMEGetInputData(npu::tile_fwk::DevTensorData* input, int64_t off0, int64_t off1)
{
    int64_t off = input->shape.dim[1] * off0 + off1;
    return ((T*)input->address)[off];
}

template <typename T>
INLINE T RUNTIMEGetInputData(npu::tile_fwk::DevTensorData* input, int64_t off0, int64_t off1, int64_t off2)
{
    int64_t off = input->shape.dim[1] * off0 + off1;
    off = off * input->shape.dim[2] + off2;
    return ((T*)input->address)[off];
}

template <typename T>
INLINE T
RUNTIMEGetInputData(npu::tile_fwk::DevTensorData* input, int64_t off0, int64_t off1, int64_t off2, int64_t off3)
{
    int64_t off = input->shape.dim[1] * off0 + off1;
    off = off * input->shape.dim[2] + off2;
    off = off * input->shape.dim[3] + off3;
    return ((T*)input->address)[off];
}

#define RUNTIME_GetInputData(index, dtype, ...) \
    RUNTIMEGetInputData<RuntimeDTypeX(dtype)>(&(RuntimeGetStartArgs())->devTensorList[(index)], __VA_ARGS__)

#define RUNTIME_GetSymbol(idx) RuntimeGetSymbol(idx)

#endif
