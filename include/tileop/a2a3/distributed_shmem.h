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
 * \file distributed_shmem.h
 * \brief
 */

#ifndef __DISTRIBUTED_SHMEM__
#define __DISTRIBUTED_SHMEM__

#include "tileop_common.h"
#include "hccl_context.h"

#include <type_traits>

namespace TileOp::Distributed {
constexpr uint16_t COPY_BLOCK_BYTE_SIZE = 32;

enum class AtomicType {
    SET,
    ADD
};

struct CopyParams {
    uint16_t nBurst;
    uint16_t lenBurst;
    uint16_t srcStride;
    uint16_t dstStride;
};

template<typename T>
TILEOP void Conv2FP32(__ubuf__ float* dst, __ubuf__ T* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlocakStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr(std::is_same_v<T, half>) {
        vconv_f162fp32(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr(std::is_same_v<T, bfloat16_t>) {
        vconv_bf162fp32(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    }
}

template<typename T>
TILEOP void DeConvFP32(__ubuf__ T* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlocakStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr(std::is_same_v<T, half>) {
        vconv_f322f16(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr(std::is_same_v<T, bfloat16_t>) {
        vconv_f322bf32(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    }
}

template<typename T, uint16_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride>
TILEOP void CopyGmToGmCore(__gm__ T* target, __ubuf__ T* buffer, __gm__ T* source)
{
    copy_gm_to_ubuf(buffer, source, sid, nBurst, lenBurst, srcStride, dstStride);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(target, buffer, sid, nBurst, lenBurst, dstStride, srcStride);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template <typename T, uint32_t rowShape, uint32_t colShape, uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride>
TILEOP void CopyGmToGmBlock(__gm__ T* target, __ubuf__ T* buffer, __gm__ T* source) {
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    TileOp::UBCopyIn<T, rowShape, colShape, bufferStride, srcStride>(buffer, source);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TileOp::UBCopyOut<T, rowShape, colShape, dstStride, bufferStride>(target, buffer);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template <typename T, uint32_t colFullBlockCount, uint32_t bufferRowShape, uint32_t bufferColShape, uint32_t colTailShape, 
    uint32_t srcStride, uint32_t dstStride>
TILEOP void CopyGmToGmRow(__gm__ T* target, __ubuf__ T* buffer, __gm__ T* source) {
    uint32_t offset = 0;
    for(uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyGmToGmBlock<T, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride>(target + offset, buffer, source + offset);
    }
    if (colTailShape > 0) {
        CopyGmToGmBlock<T, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride>(target + offset, buffer, source + offset);
    }
}

template<typename T, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride>
TILEOP void CopyGmToGm(__gm__ T* target, __ubuf__ T* buffer, __gm__ T* source)
{
    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    for(uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, source += srcRowStride, target += dstRowStride) {
        CopyGmToGmRow<T, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride>(target, buffer, source);
    } 
    if (rowTailShape > 0) {
        CopyGmToGmRow<T, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride>(target, buffer, source);
    }
}

template<typename T, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride>
TILEOP void ShmemPut(__gm__ int32_t* dummy, __ubuf__ T* buffer, __gm__ T* nonShmemDataBaseAddr, __gm__ T* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    (void)dummy;
    __gm__ T* nonShmemDataAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ T* shmemDataAddr = shmemDataBaseAddr + shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
     CopyGmToGm<T, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride>(shmemDataAddr, buffer, nonShmemDataAddr);
}

template<int32_t value, AtomicType atomicType>
TILEOP void ShmemSignal(__ubuf__ int32_t* buffer, __gm__ int32_t* dummy, __gm__ int32_t* shmemSignalBaseAddr,
    uint32_t shmemSignalOffset0, uint32_t shmemSignalOffset1, uint32_t shmemSignalOffset2, uint32_t shmemSignalOffset3,
    uint32_t shmemSignalRawShape0, uint32_t shmemSignalRawShape1, uint32_t shmemSignalRawShape2, uint32_t shmemSignalRawShape3, __gm__ int64_t *hcclContext)
{
    (void)shmemSignalRawShape0;
    (void)dummy;
    __gm__ int32_t* shmemSignalAddr = shmemSignalBaseAddr + shmemSignalOffset1 * shmemSignalRawShape2 * shmemSignalRawShape3 + shmemSignalOffset2 * shmemSignalRawShape3 + shmemSignalOffset3;
    const uint16_t sid = 0;
    const uint16_t nBurst = 1;
    const uint16_t lenBurst = 1;
    const uint16_t srcStride = 0;
    const uint16_t dstStride = 0;
    buffer[0] = value;
    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_s32();
        set_atomic_add();
    }
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(shmemSignalAddr, buffer, sid, nBurst, lenBurst, dstStride, srcStride);
    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_none();
    }
}

template<typename T, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride>
TILEOP void ShmemGet(__gm__ T* nonShmemDataBaseAddr, __ubuf__ T* buffer, __gm__ int32_t* dummy, __gm__ T* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    (void)dummy;
    __gm__ T* nonShmemDataAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ T* shmemDataAddr = shmemDataBaseAddr + shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyGmToGm<T, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride>(nonShmemDataAddr, buffer, shmemDataAddr);
}

template<typename T, bool FP32Mode>
struct ShmemReduceProcess {};

// 类模板部分特化
template<typename T>
struct ShmemReduceProcess<T, true> {
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);

        copy_gm_to_ubuf(copyUb, x, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ T* src = copyUb;
        __ubuf__ float* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(dst, src, 1, 1, 1, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            src += offset;
            dst += offset;
        }
    }

    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ float* src = sumUb;
        __ubuf__ T* dst = copyUb;
        for (uint64_t i = 0; i < repeat; i++) {
            DeConvFP32<T>(dst, src, 1, 1, 1, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            src += offset;
            dst += offset;
        }

        copy_ubuf_to_gm(out, copyUb, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }

    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        __ubuf__ float* tempUb = sumUb + row * col;

        copy_gm_to_ubuf(copyUb, x, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ float* src = copyUb;
        __ubuf__ T* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(tempUb, src, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(dst, tempUb, dst, 1, 1, 1, 1, 8, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            src += offset;
            dst += offset;
        }
    }
};

template<typename T>
struct ShmemReduceProcess<T, false> {
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        copy_gm_to_ubuf(ubTensor, x, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    }

    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        copy_ubuf_to_gm(out, ubTensor, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }

    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* sumUb = ubTensor;
        __ubuf__ T* copyUb = ubTensor + row * col;

        copy_gm_to_ubuf(copyUb, x, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ T* src = copyUb;
        __ubuf__ T* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            vadd(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            src += offset;
            dst += offset;
        }
    }
};

template<typename T, bool FP32Mode, int64_t row, int64_t col>
TILEOP void ShmemReduce(__gm__ T* out, __ubuf__ T* ubTensor, __gm__ T* in, __gm__ T* shmData, __gm__ int32_t* dummy,
    int64_t rowOffset, int64_t colOffset, int64_t rowPerRank, int64_t colPerRank, __gm__ int64_t *hcclContext)
{
    // 暂时只支持二维的in和out
    __gm__ HcclCombinOpParam *winContext = (__gm__ HcclCombinOpParam *)(hcclContext[0]);    // 需要 hcclGroupIndex
    uint32_t localRankId = winContext->rankId;
    uint32_t rankSize = winContext->rankNum;

    // 先进行gm地址的偏移计算
    int64_t offset = rowOffset * colPerRank + colOffset;
    out += offset;
    in += offset;
    shmData += offset;

    uint16_t nBurst = (uint16_t)row;                                // 暂不考虑超过uint16大小的场景
    uint16_t lenBurst = (uint16_t)(col * sizeof(T) / 32);
    uint16_t stride = (uint16_t)((colPerRank - col) * sizeof(T) / 32);
    CopyParams copyInParams{nBurst, lenBurst, stride, 0};           // ub是连续的，gm是stride间隔的
    CopyParams copyOutParams{nBurst, lenBurst, 0, stride};

    __gm__ T* x;
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        // 根据rankId进行perRank偏移计算
        if (rankId == localRankId) {
            x = in + (uint64_t)rankId * rowPerRank * colPerRank;
        } else {
            x = shmData + (uint64_t)rankId * rowPerRank * colPerRank;
        }
        if (rankId == 0) {
            ShmemReduceProcess<T, FP32Mode>::ShmemReduceCopyIn(x, ubTensor, row, col, copyInParams);
        } else {
            ShmemReduceProcess<T, FP32Mode>::ShmemReduceAdd(x, ubTensor, row, col, copyInParams);
        }
    }
    ShmemReduceProcess<T, FP32Mode>::ShmemReduceCopyOut(out, ubTensor, row, col, copyOutParams);
}

} // namespace TileOp::Distributed
#endif