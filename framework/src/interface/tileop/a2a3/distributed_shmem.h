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
 * \file distributed_shmem.h
 * \brief
 */

#ifndef __DISTRIBUTED_SHMEM__
#define __DISTRIBUTED_SHMEM__

#include "tileop_common.h"
#include "hccl_context.h"

#include <type_traits>

namespace TileOp::Distributed {
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

TILEOP uint64_t GetVirtualAddrBist(uint64_t val, uint64_t start, uint64_t end)
{
    return (((val) >> (start)) & ((1UL << ((end) - (start) + 1UL)) - 1UL));
}

TILEOP uint64_t GetVirtaulAddrOffset(uint64_t val)
{
    constexpr uint64_t offsetStart = 0UL;
    constexpr uint64_t offsetEnd = 57UL;
    return GetVirtualAddrBist(val, offsetStart, offsetEnd);
}

TILEOP uint64_t GetVirtaulAddrGroupIndex(uint64_t val)
{
    constexpr uint64_t groupIndexStart = 58UL;
    constexpr uint64_t groupIndexEnd = 59UL;
    return GetVirtualAddrBist(val, groupIndexStart, groupIndexEnd);
}

TILEOP uint64_t GetVirtaulAddrMemType(uint64_t val)
{
    constexpr uint64_t memTypeStart = 60UL;
    constexpr uint64_t memTypeEnd = 61UL;
    return GetVirtualAddrBist(val, memTypeStart, memTypeEnd);
}

template<typename T>
TILEOP __gm__ T* MapVirtaulAddr(__gm__ int64_t *hcclContext, __gm__ T* vAddr, uint32_t dstRankId)
{
    auto groupIndex = GetVirtaulAddrGroupIndex((uint64_t)vAddr);
    auto offset = GetVirtaulAddrOffset((uint64_t)vAddr);
    return (__gm__ T*)(((__gm__ TileOp::HcclCombinOpParam *)hcclContext[groupIndex])->windowsIn[dstRankId] + offset);
}

template<typename T>
TILEOP void Conv2FP32(__ubuf__ float* dst, __ubuf__ T* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlocakStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr(std::is_same_v<T, half>) {
        vconv_f162f32(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr(std::is_same_v<T, bfloat16_t>) {
        vconv_bf162f32(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    }
}

template<typename T>
TILEOP void DeConvFP32(__ubuf__ T* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlocakStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr(std::is_same_v<T, half>) {
        vconv_f322f16(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr(std::is_same_v<T, bfloat16_t>) {
        vconv_f322bf16r(dst, src, repeat, dstBlockStride, srcBlocakStride, dstRepeatStride, srcRepeatStride);
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

template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlock(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source) {
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    if constexpr (std::is_same_v<TargetType, SourceType>) {
        TileOp::UBCopyIn<SourceType, rowShape, colShape, bufferStride, srcStride>(buffer, source);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        if constexpr (atomicType == AtomicType::ADD) {
            SetAttomicType<TargetType>();
            set_atomic_add();
        }
        TileOp::UBCopyOut<TargetType, rowShape, colShape, dstStride, bufferStride>(target, buffer);
    } else {
        uint64_t castAddr = AlignUp<uint64_t>(rowShape * colShape * sizeof(UBType), 32) / sizeof(UBType);
        __ubuf__ float* castUb = (__ubuf__ float*)(buffer + castAddr);
        uint64_t repeat = AlignUp<uint64_t>(rowShape * colShape * sizeof(float), 256) / 256;
        if constexpr (atomicType == AtomicType::ADD) {
            TileOp::UBCopyIn<SourceType, rowShape, colShape, bufferStride, srcStride>(buffer, source);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            Conv2FP32<UBType>(castUb, buffer, repeat, 1, 1, 8, 4);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            set_atomic_f32();
            set_atomic_add();
            TileOp::UBCopyOut<TargetType, rowShape, colShape, dstStride, bufferStride>(target, castUb);
        } else if constexpr (atomicType == AtomicType::SET) {
            TileOp::UBCopyIn<SourceType, rowShape, colShape, bufferStride, srcStride>(castUb, source);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            DeConvFP32<UBType>(buffer, castUb, repeat, 1, 1, 4, 8);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TileOp::UBCopyOut<TargetType, rowShape, colShape, dstStride, bufferStride>(target, buffer);
        }
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_none();
    }
}

template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGmBlock(__gm__ TargetType* target, __ubuf__ SourceType* source) {
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    if constexpr (atomicType == AtomicType::ADD) {
        SetAttomicType<TargetType>();
        set_atomic_add();
    }
    TileOp::UBCopyOut<TargetType, rowShape, colShape, dstStride, bufferStride>(target, source);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_none();
    }
}

template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToUbBlock(__ubuf__ TargetType* target, __gm__ SourceType* source) {
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    TileOp::UBCopyIn<SourceType, rowShape, colShape, bufferStride, srcStride>(target, source);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
}

template<typename TargetType, typename UBType, typename SourceType, uint32_t colFullBlockCount, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t colTailShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmRow(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source) {
    uint32_t offset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyGmToGmBlock<TargetType, UBType, SourceType, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, buffer, source + offset);
    }
    if (colTailShape > 0) {
        CopyGmToGmBlock<TargetType, UBType, SourceType, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, buffer, source + offset);
    }
}

template<typename TargetType, typename SourceType, uint32_t colFullBlockCount, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t colTailShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGmRow(__gm__ TargetType* target, __ubuf__ SourceType* source) {
    uint32_t offset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyUbToGmBlock<TargetType, SourceType, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, source + offset);
    }
    if (colTailShape > 0) {
        CopyUbToGmBlock<TargetType, SourceType, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, source + offset);
    }
}

template<typename TargetType, typename SourceType, uint32_t colFullBlockCount, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t colTailShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToUbRow(__ubuf__ TargetType* target, __gm__ SourceType* source) {
    uint32_t offset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyGmToUbBlock<TargetType, SourceType, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, source + offset);
    }
    if (colTailShape > 0) {
        CopyGmToUbBlock<TargetType, SourceType, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride, atomicType>(target + offset, source + offset);
    }
}

template<typename TargetType, typename UBType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGm(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source)
{
    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, source += srcRowStride, target += dstRowStride) {
        CopyGmToGmRow<TargetType, UBType, SourceType, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, buffer, source);
    }
    if (rowTailShape > 0) {
        CopyGmToGmRow<TargetType, UBType, SourceType, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, buffer, source);
    }
}

template<typename TargetType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGm(__gm__ TargetType* target, __ubuf__ SourceType* source)
{
    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, source += srcRowStride, target += dstRowStride) {
        CopyUbToGmRow<TargetType, SourceType, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, source);
    }
    if (rowTailShape > 0) {
        CopyUbToGmRow<TargetType, SourceType, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, source);
    }
}

template<typename TargetType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape, uint32_t bufferColShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToUb(__ubuf__ TargetType* target, __gm__ SourceType* source)
{
    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, source += srcRowStride, target += dstRowStride) {
        CopyGmToUbRow<TargetType, SourceType, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, source);
    }
    if (rowTailShape > 0) {
        CopyGmToUbRow<TargetType, SourceType, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(target, source);
    }
}

template<typename T>
TILEOP void ShmemClearSignal(__gm__ int32_t* shmemSignalRawBaseAddr, __ubuf__ int32_t* buffer, __gm__ int32_t* shmemSignalBaseAddr, __gm__ T* in,
    uint32_t shmemSignalOffset0, uint32_t shmemSignalOffset1, uint32_t shmemSignalOffset2, uint32_t shmemSignalOffset3,
    uint32_t shmemSignalRawShape0, uint32_t shmemSignalRawShape1, uint32_t shmemSignalRawShape2, uint32_t shmemSignalRawShape3, __gm__ int64_t *hcclContext)
{
    (void)shmemSignalRawBaseAddr;
    (void)hcclContext;
    (void)in;

    uint32_t byteSize = sizeof(int32_t) * shmemSignalRawShape3 * shmemSignalRawShape2 * shmemSignalRawShape1;

    constexpr int32_t src = 0;
    uint8_t repeat = byteSize / VECTOR_INSTRUCTION_BYTE_SIZE;
    constexpr uint16_t dstBlockStride = 1;
    constexpr uint16_t srcBlockStride = 0; // src 是个 scalar，srcBlockStride 不起作用，设置为 0 即可
    constexpr uint8_t dstRepeatStride = 8; // 每个 block 32B，每次拷贝 256B，dst 地址连续的话设置为 8
    constexpr uint8_t srcRepeatStride = 0; // src 是个 scalar，srcRepeatStride 不起作用，设置为 0 即可
    vector_dup(buffer, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    __gm__ int32_t* shmemSignalAddr = MapVirtaulAddr<int32_t>(hcclContext, shmemSignalBaseAddr, shmemSignalOffset0) + 
            shmemSignalOffset1 * shmemSignalRawShape2 * shmemSignalRawShape3 + shmemSignalOffset2 * shmemSignalRawShape3 + shmemSignalOffset3;
    constexpr uint16_t sid = 0;
    constexpr uint16_t nBurst = 1;
    uint16_t lenBurst = byteSize / COPY_BLOCK_BYTE_SIZE;
    constexpr uint16_t srcStride = 0;
    constexpr uint16_t dstStride = 0;

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

    copy_ubuf_to_gm(shmemSignalAddr, buffer, sid, nBurst, lenBurst, dstStride, srcStride);
}

template<typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPut(__ubuf__ NonShmemType* buffer, __gm__ NonShmemType* nonShmemDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    __gm__ NonShmemType* nonShmemDataAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtaulAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyGmToGm<ShmemType, NonShmemType, NonShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(shmemDataAddr, buffer, nonShmemDataAddr);
}

template<typename InShmemType, typename OutShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPut(__ubuf__ InShmemType* buffer, __gm__ InShmemType* inShmemDataBaseAddr, __gm__ OutShmemType* shmemDataBaseAddr,
    uint32_t inShmemDataOffset0, uint32_t inShmemDataOffset1, uint32_t inShmemDataOffset2, uint32_t inShmemDataOffset3,
    uint32_t inShmemDataRawShape0, uint32_t inShmemDataRawShape1, uint32_t inShmemDataRawShape2, uint32_t inShmemDataRawShape3,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)inShmemDataRawShape0;
    (void)shmemDataRawShape0;
    __gm__ InShmemType* inShmemDataAddr = MapVirtaulAddr<InShmemType>(hcclContext, inShmemDataBaseAddr, inShmemDataOffset0) + inShmemDataOffset1 * inShmemDataRawShape2 * inShmemDataRawShape3 +
        inShmemDataOffset2 * inShmemDataRawShape3 + inShmemDataOffset3;
    __gm__ OutShmemType* shmemDataAddr = MapVirtaulAddr<OutShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) + shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 +
        shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyGmToGm<OutShmemType, InShmemType, InShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(shmemDataAddr, buffer, inShmemDataAddr);
}

template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPutUb2Gm(__ubuf__ UBType* UBDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr, uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0,
    uint32_t UBDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    __ubuf__ UBType* UBDataAddr = UBDataBaseAddr + UBDataOffset0 * UBDataRawShape1 + UBDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtaulAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) + shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 +
        shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyUbToGm<ShmemType, UBType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(shmemDataAddr, UBDataAddr);
}

template<int64_t value, AtomicType atomicType>
TILEOP void ShmemSignal(__ubuf__ int32_t* buffer, __gm__ int32_t* shmemSignalBaseAddr,
    uint32_t shmemSignalOffset0, uint32_t shmemSignalOffset1, uint32_t shmemSignalOffset2, uint32_t shmemSignalOffset3,
    uint32_t shmemSignalRawShape0, uint32_t shmemSignalRawShape1, uint32_t shmemSignalRawShape2, uint32_t shmemSignalRawShape3, __gm__ int64_t *hcclContext)
{
    (void)shmemSignalRawShape0;
    __gm__ int32_t* shmemSignalAddr = MapVirtaulAddr<int32_t>(hcclContext, shmemSignalBaseAddr, shmemSignalOffset0) +
     shmemSignalOffset1 * shmemSignalRawShape2 * shmemSignalRawShape3 + shmemSignalOffset2 * shmemSignalRawShape3 + shmemSignalOffset3;
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

template<typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemGet(__gm__ NonShmemType* nonShmemDataBaseAddr, __ubuf__ NonShmemType* buffer, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    __gm__ NonShmemType* nonShmemDataAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtaulAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyGmToGm<NonShmemType, NonShmemType, ShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(nonShmemDataAddr, buffer, shmemDataAddr);
}

template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemGetGm2Ub(__ubuf__ UBType* UBDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0, uint32_t UBDataRawShape1,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    __ubuf__ UBType* UBDataAddr = UBDataBaseAddr + UBDataOffset0 * UBDataRawShape1 + UBDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtaulAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    CopyGmToUb<UBType, ShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(UBDataAddr, shmemDataAddr);
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
TILEOP void ShmemReduce(__gm__ T* out, __ubuf__ T* ubTensor, __gm__ T* in, __gm__ T* shmData,
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

template <typename T, uint32_t topK, uint16_t rowShape, uint16_t colShape, uint16_t paddedColShape>
TILEOP void ShmemMoeCombineSend(__gm__ int32_t* dummyOut, __ubuf__ T* dataBuffer, __ubuf__ int32_t* combineInfoBuffer,
    __ubuf__ int32_t* signalBuffer, __gm__ T* in, __gm__ int32_t* combineInfo, __gm__ T* shmemDataBaseAddr,
    __gm__ int32_t* shmemSignalBaseAddr, uint64_t inOffset0, uint64_t inOffset1, __gm__ int64_t* hcclContext)
{
    (void)dummyOut;
    (void)inOffset1;

    vector_dup(signalBuffer, 0, 1, 1, 1, 8, 0);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    signalBuffer[0] = 1;

    for (uint64_t row = inOffset0; row < inOffset0 + rowShape; row++) {
        TileOp::UBCopyIn<int32_t, 1, 3, 8, 3>(combineInfoBuffer, combineInfo + MOE_COMBINE_INFO_NUM * row);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        int32_t rankId = combineInfoBuffer[0];
        if (rankId == -1) {
            break;
        }
        int32_t tokenId = combineInfoBuffer[1];
        int32_t kOffset = combineInfoBuffer[2];

        __gm__ T* winDataAddr = MapVirtaulAddr<T>(hcclContext, shmemDataBaseAddr, rankId) +
            colShape * (topK * tokenId + kOffset);
        __gm__ int32_t* winSignalAddr = MapVirtaulAddr<int32_t>(hcclContext, shmemSignalBaseAddr, rankId) +
            MOE_COMBINE_SIGNAL_OFFSET * tokenId;

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        TileOp::UBCopyIn<T, 1, colShape, paddedColShape, colShape>(dataBuffer, in + colShape * row);

        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TileOp::UBCopyOut<T, 1, colShape, colShape, paddedColShape>(winDataAddr, dataBuffer);
        pipe_barrier(PIPE_MTE3);

        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        set_atomic_add();
        set_atomic_s32();
        copy_ubuf_to_gm(winSignalAddr, signalBuffer, 0, 1, 1, 0, 0);
        set_atomic_none();
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

TILEOP void ShmemMoeCombineWaitSignal(__gm__ int32_t* winSignalAddr, __ubuf__ int32_t* signalBuffer,
    int32_t expectedValue)
{
    do {
        copy_gm_to_ubuf(signalBuffer, winSignalAddr, 0, 1, 1, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    } while (signalBuffer[0] != expectedValue);
}

template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape>
TILEOP void ShmemMoeCombineComputeRoutedExperts(__ubuf__ T* out, __ubuf__ float* mulFp32Buffer,
    __ubuf__ float* sumFp32Buffer, __ubuf__ float* scale, __gm__ T* winDataAddr, uint8_t repeat)
{
    for (int kOffset = 0; kOffset < topK; kOffset++) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TileOp::UBCopyIn<T, 1, colShape, paddedColShape, colShape>(out, winDataAddr);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        vconv_bf162f32(mulFp32Buffer, out, repeat, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmuls(mulFp32Buffer, mulFp32Buffer, static_cast<float>(scale[kOffset]), repeat, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(sumFp32Buffer, mulFp32Buffer, sumFp32Buffer, repeat, 1, 1, 1, 8, 8, 8);
        winDataAddr += colShape;
    }
}

template <typename T, uint32_t topK, uint16_t colShape, uint16_t paddedColShape>
TILEOP void ShmemMoeCombineCompute(__ubuf__ T* out, __ubuf__ float* mulFp32Buffer, __ubuf__ float* sumFp32Buffer,
    __ubuf__ float* scale, __gm__ T* winDataAddr)
{
    uint8_t repeat = static_cast<uint8_t>(
        AlignUp<uint16_t>(sizeof(float) * paddedColShape, VECTOR_INSTRUCTION_BYTE_SIZE) / VECTOR_INSTRUCTION_BYTE_SIZE
    );

    vector_dup(sumFp32Buffer, 0.0f, repeat, 1, 1, 8, 8);

    ShmemMoeCombineComputeRoutedExperts<T, topK, colShape, paddedColShape>(out, mulFp32Buffer, sumFp32Buffer, scale,
        winDataAddr, repeat);

    pipe_barrier(PIPE_V);
    vconv_f322bf16a(out, sumFp32Buffer, repeat, 1, 1, 4, 8);
}

template <typename T, uint32_t topK, uint16_t rowShape, uint16_t colShape, uint16_t paddedColShape>
TILEOP void ShmemMoeCombineReceive(__gm__ T* out, __ubuf__ float* mulFp32Buffer, __ubuf__ float* sumFp32Buffer,
    __ubuf__ T* outBuffer, __gm__ int32_t* dummyIn, __ubuf__ float* scale, __gm__ T* shmemDataBaseAddr,
    __gm__ int32_t* shmemSignalBaseAddr, uint64_t shmemDataOffset0, uint64_t shmemDataOffset1,
    uint64_t shmemDataOffset2, uint64_t shmemDataOffset3, __gm__ int64_t* hcclContext)
{
    (void)dummyIn;
    (void)shmemDataOffset1;
    (void)shmemDataOffset3;

    uint64_t thisRankId = shmemDataOffset0;
    uint64_t rowOffset = shmemDataOffset2;

    for (uint64_t tokenId = rowOffset; tokenId < rowOffset + rowShape; tokenId++) {
        __gm__ int32_t* winSignalAddr = MapVirtaulAddr<int32_t>(hcclContext, shmemSignalBaseAddr, thisRankId) +
            MOE_COMBINE_SIGNAL_OFFSET * tokenId;
        __ubuf__ int32_t* signalBuffer = reinterpret_cast<__ubuf__ int32_t*>(outBuffer);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        ShmemMoeCombineWaitSignal(winSignalAddr, signalBuffer, topK);

        constexpr uint32_t scaleColShape = AlignUp<uint32_t>(sizeof(float) * topK, COPY_BLOCK_BYTE_SIZE) /
            sizeof(float);
        __gm__ T* winDataAddr = MapVirtaulAddr<T>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
            colShape * topK * tokenId;
        ShmemMoeCombineCompute<T, topK, colShape, paddedColShape>(outBuffer, mulFp32Buffer,
            sumFp32Buffer, scale + scaleColShape * tokenId, winDataAddr);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TileOp::UBCopyOut<T, 1, colShape, colShape, paddedColShape>(out + colShape * tokenId, outBuffer);
    }
}

} // namespace TileOp::Distributed
#endif