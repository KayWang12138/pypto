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

constexpr int32_t DEV_SHAPE_DIM_MAX = 5;

struct DevAscendShape {
    int dimSize{0};
    int dim[DEV_SHAPE_DIM_MAX];
};

struct DevAscendTensorData {
    uint64_t address{0};
    DevAscendShape shape;
};

struct DevStartArgsBase {
    DevAscendTensorData *inputTensorList;
    uint64_t inputTensorSize;
    DevAscendTensorData *outputTensorList;
    uint64_t outputTensorSize;
};

using CallRootEntryType = void *(*)(void *, uint64_t);

enum CallRootStage {
    T_CALLROOT_ALLOC = 0,
    T_CALLROOT_STITCH = 1,
    T_CALLROOT_LOG = 2,
    T_CALLROOT_MAX = 3,
};

using Call1EntryType = uint64_t (*)(uint64_t);

using Call2EntryType = uint64_t (*)(uint64_t, uint64_t);

using Call3EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t);

using Call4EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t);

using Call5EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

using Call5EntryType = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

#define RUNTIME_FINISH_FUNCKEY ((uint64_t)(-1))

#define RuntimeGetInputShapeDimSize(input) ((input)->shape.dimSize)
#define RuntimeGetInputShapeDim(input, n) ((input)->shape.dim[(n)])
#define RuntimeGetInputDataInt32Dim1(input, off0) (((int32_t *)(input)->address)[(off0)])
#define RuntimeGetInputDataInt32Dim2(input, off0, off1) \
    (((int32_t *)(input)->address)[(off0) * (input)->shape.dim[1] + (off1)])
#define RuntimeGetInputDataInt32Dim3(input, off0, off1, off2) \
    (((int32_t *)(input)->address)[(off0) * (input)->shape.dim[1] * (input)->shape.dim[2] + (off1) * (input)->shape.dim[2] + (off2)])
#define RuntimeIsLoopBegin(idx, begin) (idx) == (begin)
#define RuntimeIsLoopEnd(idx, end) (idx) >= (end)

#define RUNTIME_GetInputShapeDimSize(ctx, inputIndex) \
    RuntimeGetInputShapeDimSize(&(startArgs)->inputTensorList[(inputIndex)])
#define RUNTIME_GetInputShapeDim(ctx, inputIndex, n) \
    RuntimeGetInputShapeDim(&(startArgs)->inputTensorList[(inputIndex)], (n))
#define RUNTIME_GetInputDataInt32Dim1(ctx, inputIndex, off0) \
    RuntimeGetInputDataInt32Dim1(&(startArgs)->inputTensorList[(inputIndex)], (off0))
#define RUNTIME_GetInputDataInt32Dim2(ctx, inputIndex, off0, off1) \
    RuntimeGetInputDataInt32Dim2(&(startArgs)->inputTensorList[(inputIndex)], (off0), (off1))
#define RUNTIME_GetInputDataInt32Dim3(ctx, inputIndex, off0, off1, off2) \
    RuntimeGetInputDataInt32Dim3(&(startArgs)->inputTensorList[(inputIndex)], (off0), (off1), (off2))
#define RUNTIME_IsLoopBegin(ctx, idx, begin) RuntimeIsLoopBegin((idx), (begin))
#define RUNTIME_IsLoopEnd(ctx, idx, end) RuntimeIsLoopEnd((idx), (end))

__always_inline
int64_t RUNTIME_GetViewValidShapeDim(void *ctx, int64_t validshape, int64_t viewOffset, int64_t viewshape) {
    (void)ctx;
    validshape -= viewOffset;
    if (validshape > viewshape)
        validshape = viewshape;
    else if (validshape < 0)
        validshape = 0;
    return validshape;
}

__always_inline
int64_t RUNTIME_Max(int64_t input1, int64_t input2) {
    if (input1 > input2)
        return input1;
    else
        return input2;
}

__always_inline
int64_t RUNTIME_Min(int64_t input1, int64_t input2) {
    if (input1 < input2)
        return input1;
    else
        return input2;
}
}  // namespace npu::tile_fwk
