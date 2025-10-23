/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen.h
 * \brief
 */

#pragma once

#include <sys/cdefs.h>
#include <cstdint>
#include <vector>

namespace npu::tile_fwk {

using CallRootEntryType = void *(*)(void *, uint64_t);

enum CallRootStage {
    T_CALLROOT_ALLOC = 0,
    T_CALLROOT_STITCH = 1,
    T_CALLROOT_LOG = 2,
    T_CALLROOT_SHMEM_ALLOC = 3,
    T_CALLROOT_MAX = 4,
};

using Call1EntryType = uint64_t (*)(uint64_t);

using Call2EntryType = uint64_t (*)(uint64_t, uint64_t);

using Call3EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t);

using Call4EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t);

using Call5EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

using Call5EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

#define RUNTIME_FINISH_FUNCKEY (static_cast<uint64_t>(-1))

#define RuntimeGetInputShapeDimSize(input) ((input)->shape.dimSize)
#define RuntimeGetInputShapeDim(input, n) ((input)->shape.dim[(n)])
#define RuntimeGetInputDataInt32Dim1(input, off0) (((int32_t *)(input)->address)[(off0)])
#define RuntimeGetInputDataInt32Dim2(input, off0, off1) \
    (((int32_t *)(input)->address)[(off0) * (input)->shape.dim[1] + (off1)])
#define RuntimeGetInputDataInt32Dim3(input, off0, off1, off2) \
    (((int32_t *)(input)->address)[(off0) * (input)->shape.dim[1] * (input)->shape.dim[2] + (off1) * (input)->shape.dim[2] + (off2)])
#define RuntimeGetInputDataInt32Dim4(input, off0, off1, off2, off3) \
    (((int32_t *)(input)->address)[((off0 * (input)->shape.dim[1] + off1) * (input)->shape.dim[2] + off2) * (input)->shape.dim[3] + (off3)])
#define RuntimeIsLoopBegin(idx, begin) (idx) == (begin)
#define RuntimeIsLoopEnd(idx, end) (idx) >= (end)

__always_inline
int64_t RuntimeGetViewValidShapeDim(int64_t validshape, int64_t viewOffset, int64_t viewshape) {
    validshape -= viewOffset;
    if (validshape > viewshape)
        validshape = viewshape;
    else if (validshape < 0)
        validshape = 0;
    return validshape;
}

__always_inline
int64_t RuntimeMax(int64_t input1, int64_t input2) {
    if (input1 > input2)
        return input1;
    else
        return input2;
}

__always_inline
int64_t RuntimeMin(int64_t input1, int64_t input2) {
    if (input1 < input2)
        return input1;
    else
        return input2;
}

__always_inline
int64_t RuntimeEq(int64_t input1, int64_t input2) {
    return input1 == input2;
}

__always_inline
int64_t RuntimeNe(int64_t input1, int64_t input2) {
    return input1 != input2;
}

#define RUNTIME_GetInputShapeDimSize(inputIndex) \
    RuntimeGetInputShapeDimSize(&(startArgs)->inputTensorList[(inputIndex)])
#define RUNTIME_GetInputShapeDim(inputIndex, n) \
    RuntimeGetInputShapeDim(&(startArgs)->inputTensorList[(inputIndex)], (n))
#define RUNTIME_GetInputDataInt32Dim1(inputIndex, off0) \
    RuntimeGetInputDataInt32Dim1(&(startArgs)->inputTensorList[(inputIndex)], (off0))
#define RUNTIME_GetInputDataInt32Dim2(inputIndex, off0, off1) \
    RuntimeGetInputDataInt32Dim2(&(startArgs)->inputTensorList[(inputIndex)], (off0), (off1))
#define RUNTIME_GetInputDataInt32Dim3(inputIndex, off0, off1, off2) \
    RuntimeGetInputDataInt32Dim3(&(startArgs)->inputTensorList[(inputIndex)], (off0), (off1), (off2))
#define RUNTIME_GetInputDataInt32Dim4(inputIndex, off0, off1, off2, off3) \
    RuntimeGetInputDataInt32Dim4(&(startArgs)->inputTensorList[(inputIndex)], (off0), (off1), (off2), (off3))
#define RUNTIME_IsLoopBegin(idx, begin) RuntimeIsLoopBegin((idx), (begin))
#define RUNTIME_IsLoopEnd(idx, end) RuntimeIsLoopEnd((idx), (end))

#define RUNTIME_GetViewValidShapeDim(validShape, viewOffset, viewShape) RuntimeGetViewValidShapeDim(validShape, viewOffset, viewShape)
#define RUNTIME_Max(lhs, rhs) RuntimeMax(lhs, rhs)
#define RUNTIME_Min(lhs, rhs) RuntimeMin(lhs, rhs)

#define RUNTIME_GetSymbol(idx)          (symbolTable[idx])

}  // namespace npu::tile_fwk
