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
 * \file tileop_shmem.h
 * \brief Shmem (shared memory) tileops: clear/set, GM/UB copy, Put/Get/Signal, Reduce.
 */

#ifndef __DISTRIBUTED_SHMEM__
#define __DISTRIBUTED_SHMEM__

#include "common.h"
#include <type_traits>

#include "pto/pto-inst.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/type.hpp"

// ---------------------------------------------------------------------------
// Pipe sync macros
// ---------------------------------------------------------------------------
#define PIPE_SYNC(from, to) \
    do { \
        set_flag(from, to, EVENT_ID0); \
        wait_flag(from, to, EVENT_ID0); \
    } while(0)

#define PIPE_SYNC_V_MTE3()   PIPE_SYNC(PIPE_V, PIPE_MTE3)
#define PIPE_SYNC_MTE2_V()   PIPE_SYNC(PIPE_MTE2, PIPE_V)
#define PIPE_SYNC_MTE2_S()   PIPE_SYNC(PIPE_MTE2, PIPE_S)
#define PIPE_SYNC_MTE3_S()   PIPE_SYNC(PIPE_MTE3, PIPE_S)
#define PIPE_SYNC_S_MTE2()   PIPE_SYNC(PIPE_S, PIPE_MTE2)
#define PIPE_SYNC_S_MTE3()   PIPE_SYNC(PIPE_S, PIPE_MTE3)
#define PIPE_SYNC_V_S()      PIPE_SYNC(PIPE_V, PIPE_S)
#define PIPE_SYNC_MTE3_MTE2() PIPE_SYNC(PIPE_MTE3, PIPE_MTE2)

namespace TileOp::Distributed {

// ---------------------------------------------------------------------------
// Type conversion (UB): half/bf16 <-> float
// ---------------------------------------------------------------------------
template<typename T>
TILEOP void Conv2FP32(__ubuf__ float* dst, __ubuf__ T* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr(std::is_same_v<T, half>) {
        vconv_f162f32(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr(std::is_same_v<T, bfloat16_t>) {
        vconv_bf162f32(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
}

template<typename T>
TILEOP void DeConvFP32(__ubuf__ T* dst, __ubuf__ float* src, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
{
    if constexpr(std::is_same_v<T, half>) {
        vconv_f322f16(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    } else if constexpr(std::is_same_v<T, bfloat16_t>) {
        vconv_f322bf16r(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }
}

// ---------------------------------------------------------------------------
// Shmem tensor/tile type aliases
// ---------------------------------------------------------------------------
using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template<typename T, uint32_t RowShape, uint32_t ColShape>
using ShmemGlobalTensor = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template<typename T, uint32_t RowShape, uint32_t ColShape>
using ShmemUbTile = pto::Tile<pto::TileType::Vec, T, RowShape, ColShape, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>;

// ---------------------------------------------------------------------------
// Shmem clear / set
// ---------------------------------------------------------------------------
// Zero a shmem region; V→MTE3 sync ensures vector_dup completes before TSTORE.
template<typename T, uint32_t bufferEleNum, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2, uint32_t shmemTensorRawShape3>
TILEOP void ShmemClear(__ubuf__ T* buffer, __gm__ T* shmemTensorAddr)
{
    constexpr uint8_t repeat = sizeof(T) * bufferEleNum / VECTOR_INSTRUCTION_BYTE_SIZE;
    vector_dup(buffer, static_cast<T>(0), repeat, 1, 0, 8, 0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    constexpr uint32_t shmemTensorEleNum = shmemTensorRawShape1 * shmemTensorRawShape2 * shmemTensorRawShape3;
    constexpr uint32_t fullChunkCount = shmemTensorEleNum / bufferEleNum;

    ShmemUbTile<T, 1, bufferEleNum> ubTile(1, bufferEleNum);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(buffer));

    for (int32_t i = 0; i < fullChunkCount; i++) {
        __gm__ T* dstAddr = shmemTensorAddr + bufferEleNum * i;
        ShapeDyn shape(1, 1, 1, 1, bufferEleNum);
        StrideDyn strideDyn(1, 1, 1, bufferEleNum, 1);
        ShmemGlobalTensor<T, 1, bufferEleNum> gmTensor(dstAddr, shape, strideDyn);
        pto::TSTORE<decltype(ubTile), decltype(gmTensor), pto::AtomicType::AtomicNone>(gmTensor, ubTile);
    }

    constexpr uint32_t tailEleNum = shmemTensorEleNum % bufferEleNum;
    if constexpr (tailEleNum != 0) {
        __gm__ T* tailDstAddr = shmemTensorAddr + bufferEleNum * fullChunkCount;
        ShapeDyn tailShape(1, 1, 1, 1, tailEleNum);
        StrideDyn tailStrideDyn(1, 1, 1, tailEleNum, 1);
        ShmemGlobalTensor<T, 1, tailEleNum> tailGmTensor(tailDstAddr, tailShape, tailStrideDyn);
        ShmemUbTile<T, 1, tailEleNum> tailUbTile(1, tailEleNum);
        pto::TASSIGN(tailUbTile, reinterpret_cast<uintptr_t>(buffer));
        pto::TSTORE<decltype(tailUbTile), decltype(tailGmTensor), pto::AtomicType::AtomicNone>(tailGmTensor, tailUbTile);
    }
}

template<typename T, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2, uint32_t shmemTensorRawShape3,
    uint32_t bufferEleNum>
TILEOP void ShmemSet(__ubuf__ T* buffer, __gm__ T* shmemTensorBaseAddr, uint32_t shmemTensorOffset0,
    uint32_t shmemTensorOffset1, uint32_t shmemTensorOffset2, uint32_t shmemTensorOffset3, __gm__ int64_t *hcclContext)
{
    __gm__ T* shmemTensorAddr = MapVirtualAddr<T>(hcclContext, shmemTensorBaseAddr, shmemTensorOffset0) + 
        shmemTensorRawShape3 * shmemTensorRawShape2 * shmemTensorOffset1 + shmemTensorRawShape3 * shmemTensorOffset2 +
        shmemTensorOffset3;
    ShmemClear<T, bufferEleNum, shmemTensorRawShape1, shmemTensorRawShape2, shmemTensorRawShape3>(buffer, shmemTensorAddr);
}

template<typename T, uint32_t worldSize, uint32_t stride, uint32_t signalMaxTileNum,
    uint32_t bufferEleNum>
TILEOP void ShmemSet(__ubuf__ T* buffer, __gm__ T* shmemTensorBaseAddr, uint32_t shmemTensorOffset0,
    uint32_t shmemTensorOffset1, uint32_t shmemTensorOffset2, uint32_t shmemTensorOffset3, uint32_t shmemTensorOffset4, 
    uint32_t shmemTensorRawShape0, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2,
    uint32_t shmemTensorRawShape3, uint32_t shmemTensorRawShape4, uint32_t shmemTensorShape0, uint32_t shmemTensorShape1,
    uint32_t shmemTensorShape2, uint32_t shmemTensorShape3, uint32_t shmemTensorShape4, __gm__ int64_t *hcclContext)
{
    int32_t tileIndex = (shmemTensorOffset3 / shmemTensorShape3) *
        (shmemTensorRawShape4 / shmemTensorShape4 + (shmemTensorRawShape4 % shmemTensorShape4 == 0 ? 0 : 1)) +
        (shmemTensorOffset4 / shmemTensorShape4);
    int32_t rowTileNum = shmemTensorRawShape3 / shmemTensorShape3 + (shmemTensorRawShape3 % shmemTensorShape3 == 0 ? 0 : 1);
    int32_t colTileNum = shmemTensorRawShape4 / shmemTensorShape4 + (shmemTensorRawShape4 % shmemTensorShape4 == 0 ? 0 : 1);
    int32_t totalTileNum = rowTileNum * colTileNum;

    __gm__ T* shmemTensorAddr = MapVirtualAddr<T>(hcclContext, shmemTensorBaseAddr, shmemTensorOffset0) + 
        shmemTensorOffset0 * shmemTensorRawShape2 * totalTileNum * stride + (shmemTensorOffset2 * totalTileNum + tileIndex) * stride;

    ShmemClear<T, bufferEleNum, worldSize, signalMaxTileNum, stride>(buffer, shmemTensorAddr);
}

// ---------------------------------------------------------------------------
// Copy: GM↔GM (via UB, with optional type conversion and ping-pong)
// ---------------------------------------------------------------------------
template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlockSameType(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source,
    uint32_t eventId) {
    ShapeDyn shape(1, 1, 1, rowShape, colShape);
    StrideDyn srcStrideDyn(rowShape, rowShape, rowShape, srcStride, 1);
    StrideDyn dstStrideDyn(rowShape, rowShape, rowShape, dstStride, 1);
    ShmemGlobalTensor<SourceType, rowShape, colShape> srcGlobal(source, shape, srcStrideDyn);
    ShmemGlobalTensor<TargetType, rowShape, colShape> dstGlobal(target, shape, dstStrideDyn);
    ShmemUbTile<UBType, rowShape, colShape> ubTile(rowShape, colShape);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(buffer));
    pto::TLOAD(ubTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, eventId);
    wait_flag(PIPE_MTE2, PIPE_MTE3, eventId);
    if constexpr (atomicType == AtomicType::ADD) {
        pto::TSTORE<decltype(ubTile), decltype(dstGlobal), pto::AtomicType::AtomicAdd>(dstGlobal, ubTile);
    } else {
        pto::TSTORE<decltype(ubTile), decltype(dstGlobal), pto::AtomicType::AtomicNone>(dstGlobal, ubTile);
    }
}

template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlockConvert(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source,
    uint32_t eventId) {
    constexpr uint64_t copyLen = rowShape * AlignUp<uint64_t>(colShape * sizeof(UBType), 32) / sizeof(UBType);
    __ubuf__ float* castUb = (__ubuf__ float*)(buffer + copyLen);
    ShapeDyn shape(1, 1, 1, rowShape, colShape);
    StrideDyn srcStrideDyn(rowShape, rowShape, rowShape, srcStride, 1);
    StrideDyn dstStrideDyn(rowShape, rowShape, rowShape, dstStride, 1);
    ShmemGlobalTensor<SourceType, rowShape, colShape> srcGlobal(source, shape, srcStrideDyn);
    ShmemGlobalTensor<TargetType, rowShape, colShape> dstGlobal(target, shape, dstStrideDyn);
    if constexpr (atomicType == AtomicType::ADD) {
        ShmemUbTile<UBType, rowShape, colShape> srcTile(rowShape, colShape);
        ShmemUbTile<float, rowShape, colShape> dstTile(rowShape, colShape);
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(buffer));
        pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(castUb));
        pto::TLOAD(srcTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        pto::TSTORE<decltype(dstTile), decltype(dstGlobal), pto::AtomicType::AtomicAdd>(dstGlobal, dstTile);
    } else {
        ShmemUbTile<float, rowShape, colShape> srcTile(rowShape, colShape);
        ShmemUbTile<UBType, rowShape, colShape> dstTile(rowShape, colShape);
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(castUb));
        pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(buffer));
        pto::TLOAD(srcTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        pto::TSTORE<decltype(dstTile), decltype(dstGlobal), pto::AtomicType::AtomicNone>(dstGlobal, dstTile);
    }
}

// Single block GM→UB→GM. With conversion: buffer[0..copyLen-1]=UBType, buffer[copyLen..]=float.
template<typename TargetType, typename UBType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmBlock(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source,
    uint32_t eventId = EVENT_ID0) {
    (void)bufferStride;
    wait_flag(PIPE_MTE3, PIPE_S, eventId);
    set_flag(PIPE_S, PIPE_MTE2, eventId);
    wait_flag(PIPE_S, PIPE_MTE2, eventId);
    if constexpr (std::is_same_v<TargetType, SourceType>) {
        CopyGmToGmBlockSameType<TargetType, UBType, SourceType, rowShape, colShape, srcStride, dstStride, atomicType>(
            target, buffer, source, eventId);
    } else {
        CopyGmToGmBlockConvert<TargetType, UBType, SourceType, rowShape, colShape, srcStride, dstStride, atomicType>(
            target, buffer, source, eventId);
    }
    set_flag(PIPE_MTE3, PIPE_S, eventId);
}

// Row of blocks with ping-pong double buffering.
template<typename TargetType, typename UBType, typename SourceType, uint32_t colFullBlockCount, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t colTailShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGmRow(__gm__ TargetType* target, __ubuf__ UBType* bufferA, __ubuf__ UBType* bufferB, __gm__ SourceType* source, uint32_t eventId = EVENT_ID0) {
    uint32_t offset = 0;
    __ubuf__ UBType* useBuffer = eventId == EVENT_ID0 ? bufferA : bufferB;
    
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyGmToGmBlock<TargetType, UBType, SourceType, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride, atomicType>(
            target + offset, useBuffer, source + offset, eventId);
        eventId = eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0;
        useBuffer = eventId == EVENT_ID0 ? bufferA : bufferB;
    }
    if constexpr (colTailShape > 0) {
        CopyGmToGmBlock<TargetType, UBType, SourceType, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride, atomicType>(
            target + offset, useBuffer, source + offset, eventId);
    }
}

// Full tile copy with row/column chunking. Ping-pong layout: same type bufferA|bufferB; with conversion bufferA|castUbA|bufferB|castUbB.
template<typename TargetType, typename UBType, typename SourceType, uint32_t tileRowShape, uint32_t tileColShape, 
    uint32_t bufferRowShape, uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyGmToGm(__gm__ TargetType* target, __ubuf__ UBType* buffer, __gm__ SourceType* source)
{
    constexpr uint32_t rowFullBlockCount = tileRowShape / bufferRowShape;
    constexpr uint32_t colFullBlockCount = tileColShape / bufferColShape;
    constexpr uint32_t rowTailShape = tileRowShape % bufferRowShape;
    constexpr uint32_t colTailShape = tileColShape % bufferColShape;
    constexpr uint32_t srcRowStride = bufferRowShape * srcStride;
    constexpr uint32_t dstRowStride = bufferRowShape * dstStride;
    
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
    
    uint32_t eventId = EVENT_ID0;
    constexpr uint32_t colLoopNum = colFullBlockCount + (colTailShape > 0 ? 1 : 0);
    
    constexpr uint32_t copyLen = bufferRowShape * AlignUp<uint32_t>(bufferColShape * sizeof(UBType), 32) / sizeof(UBType);
    __ubuf__ UBType* bufferA = buffer;
    __ubuf__ UBType* bufferB = buffer + copyLen;
    
    if constexpr (!std::is_same_v<TargetType, SourceType>) {
        constexpr uint64_t castSize = AlignUp<uint64_t>(copyLen * sizeof(float), 256);
        bufferB = buffer + copyLen + castSize / sizeof(UBType);
    }
    
    __gm__ SourceType* srcPtr = source;
    __gm__ TargetType* dstPtr = target;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, srcPtr += srcRowStride, dstPtr += dstRowStride) {
        CopyGmToGmRow<TargetType, UBType, SourceType, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(
            dstPtr, bufferA, bufferB, srcPtr, eventId);
        if constexpr (colLoopNum % 2 == 1) {
            eventId = eventId == EVENT_ID0 ? EVENT_ID1 : EVENT_ID0;
        }
    }
    
    if constexpr (rowTailShape > 0) {
        CopyGmToGmRow<TargetType, UBType, SourceType, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(
            dstPtr, bufferA, bufferB, srcPtr, eventId);
    }
    
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
}

// ---------------------------------------------------------------------------
// Copy: UB → GM
// ---------------------------------------------------------------------------
template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t bufferStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGmBlock(__gm__ TargetType* target, __ubuf__ SourceType* source) {
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    
    ShapeDyn shape(1, 1, 1, rowShape, colShape);
    StrideDyn dstStrideDyn(rowShape, rowShape, rowShape, dstStride, 1);
    
    ShmemGlobalTensor<TargetType, rowShape, colShape> dstGlobal(target, shape, dstStrideDyn);
    ShmemUbTile<SourceType, rowShape, colShape> ubTile(rowShape, colShape);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(source));
    
    if constexpr (atomicType == AtomicType::ADD) {
        pto::TSTORE<decltype(ubTile), decltype(dstGlobal), pto::AtomicType::AtomicAdd>(dstGlobal, ubTile);
    } else {
        pto::TSTORE<decltype(ubTile), decltype(dstGlobal), pto::AtomicType::AtomicNone>(dstGlobal, ubTile);
    }
    
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template<typename TargetType, typename SourceType, uint32_t colFullBlockCount, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t colTailShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void CopyUbToGmRow(__gm__ TargetType* target, __ubuf__ SourceType* source) {
    uint32_t offset = 0;
    for (uint32_t colIndex = 0; colIndex < colFullBlockCount; ++colIndex, offset += bufferColShape) {
        CopyUbToGmBlock<TargetType, SourceType, bufferRowShape, bufferColShape, srcStride, bufferColShape, dstStride, atomicType>(
            target + offset, source + offset);
    }
    if constexpr (colTailShape > 0) {
        CopyUbToGmBlock<TargetType, SourceType, bufferRowShape, colTailShape, srcStride, bufferColShape, dstStride, atomicType>(
            target + offset, source + offset);
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
    
    __ubuf__ SourceType* srcPtr = source;
    __gm__ TargetType* dstPtr = target;
    for (uint32_t rowIndex = 0; rowIndex < rowFullBlockCount; ++rowIndex, srcPtr += srcRowStride, dstPtr += dstRowStride) {
        CopyUbToGmRow<TargetType, SourceType, colFullBlockCount, bufferRowShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(
            dstPtr, srcPtr);
    }
    if constexpr (rowTailShape > 0) {
        CopyUbToGmRow<TargetType, SourceType, colFullBlockCount, rowTailShape, bufferColShape, colTailShape, srcStride, dstStride, atomicType>(
            dstPtr, srcPtr);
    }
}

// ---------------------------------------------------------------------------
// Copy: GM → UB (single block, optional type conversion)
// ---------------------------------------------------------------------------
template<typename TargetType, typename SourceType, uint32_t rowShape, uint32_t colShape,
    uint32_t srcStride, uint32_t dstStride>
TILEOP void CopyGmToUbBlock(__ubuf__ TargetType* target, __ubuf__ TargetType* buffer, __gm__ SourceType* source) {
    if constexpr (std::is_same_v<TargetType, SourceType>) {
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        
        ShapeDyn shape(1, 1, 1, rowShape, colShape);
        StrideDyn srcStrideDyn(rowShape, rowShape, rowShape, srcStride, 1);
        
        ShmemGlobalTensor<SourceType, rowShape, colShape> srcGlobal(source, shape, srcStrideDyn);
        ShmemUbTile<TargetType, rowShape, colShape> ubTile(rowShape, colShape);
        pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(target));
        
        pto::TLOAD(ubTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    } else {
        __ubuf__ float* castUb = (__ubuf__ float*)buffer;
        
        ShapeDyn shape(1, 1, 1, rowShape, colShape);
        StrideDyn srcStrideDyn(rowShape, rowShape, rowShape, srcStride, 1);
        
        ShmemGlobalTensor<SourceType, rowShape, colShape> srcGlobal(source, shape, srcStrideDyn);
        ShmemUbTile<float, rowShape, colShape> srcTile(rowShape, colShape);
        ShmemUbTile<TargetType, rowShape, colShape> dstTile(rowShape, colShape);
        pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(castUb));
        pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(target));
        
        pto::TLOAD(srcTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        
        pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }
}

// ---------------------------------------------------------------------------
// Shmem Put / Get / Signal
// ---------------------------------------------------------------------------
// Put: local GM (or inShmem GM) → remote shmem GM.
template<typename NonShmemType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPut(__ubuf__ NonShmemType* buffer, __gm__ NonShmemType* nonShmemDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t nonShmemDataOffset0, uint32_t nonShmemDataOffset1, uint32_t nonShmemDataRawShape0,
    uint32_t nonShmemDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, uint32_t shmemGetTensorDataOffset, __gm__ int64_t *hcclContext)
{
    (void)nonShmemDataRawShape0;
    (void)shmemDataRawShape0;
    if (shmemGetTensorDataOffset != -1) {
        shmemDataOffset2 = shmemGetTensorDataOffset;
    }
    __gm__ NonShmemType* nonShmemDataAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    if constexpr (atomicType == AtomicType::ADD) {
        SetAttomicType<ShmemType>();
        set_atomic_add();
    }
    CopyGmToGm<ShmemType, NonShmemType, NonShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
        shmemDataAddr, buffer, nonShmemDataAddr);
    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_none();
    }
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
    
    __gm__ InShmemType* inShmemDataAddr = MapVirtualAddr<InShmemType>(hcclContext, inShmemDataBaseAddr, inShmemDataOffset0) +
        inShmemDataOffset1 * inShmemDataRawShape2 * inShmemDataRawShape3 + inShmemDataOffset2 * inShmemDataRawShape3 + inShmemDataOffset3;
    __gm__ OutShmemType* shmemDataAddr = MapVirtualAddr<OutShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    CopyGmToGm<OutShmemType, InShmemType, InShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
        shmemDataAddr, buffer, inShmemDataAddr);
}

// Put UB directly to remote shmem GM.
template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPutUb2Gm(__ubuf__ UBType* UBDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr, uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0,
    uint32_t UBDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    
    __ubuf__ UBType* UBDataAddr = UBDataBaseAddr + UBDataOffset0 * UBDataRawShape1 + UBDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    CopyUbToGm<ShmemType, UBType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(shmemDataAddr, UBDataAddr);
}

// Signal: write value to remote ranks; S→MTE3 sync so scalar write is visible to TSTORE.
template<int64_t value, int32_t stride, int32_t tileRowShape, int32_t tileColShape, AtomicType atomicType>
TILEOP void ShmemSignal(__ubuf__ int32_t* buffer, __gm__ int32_t* shmemSignalBaseAddr,
    uint32_t shmemSignalOffset0, uint32_t shmemSignalOffset1, uint32_t shmemSignalOffset2, uint32_t shmemSignalOffset3, uint32_t shmemSignalOffset4,
    uint32_t shmemSignalRawShape0, uint32_t shmemSignalRawShape1, uint32_t shmemSignalRawShape2, uint32_t shmemSignalRawShape3, uint32_t shmemSignalRawShape4,
    uint32_t shmemSignalShape0, uint32_t shmemSignalShape1, uint32_t shmemSignalShape2, uint32_t shmemSignalShape3, uint32_t shmemSignalShape4, __gm__ int64_t *hcclContext)
{
    (void)shmemSignalRawShape0;
    (void)shmemSignalRawShape1;
    (void)shmemSignalShape1;
    (void)shmemSignalShape2;
    
    int32_t tileCols = (static_cast<int32_t>(shmemSignalRawShape4) + tileColShape - 1) / tileColShape;
    int32_t tileRows = (static_cast<int32_t>(shmemSignalRawShape3) + tileRowShape - 1) / tileRowShape;
    int32_t tileRow = static_cast<int32_t>(shmemSignalOffset3) / tileRowShape;
    int32_t tileCol = static_cast<int32_t>(shmemSignalOffset4) / tileColShape;
    int32_t tileIndex = tileRow * tileCols + tileCol;
    int32_t totalTileNum = tileRows * tileCols;

    buffer[0] = static_cast<int32_t>(value);
    constexpr uint32_t signalColShape = 8;  // 8*4=32B alignment
    ShmemUbTile<int32_t, 1, signalColShape> signalTile(1, 1);
    pto::TASSIGN(signalTile, reinterpret_cast<uintptr_t>(buffer));
    
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    
    for (uint32_t rankId = shmemSignalOffset0; rankId < shmemSignalOffset0 + shmemSignalShape0; rankId++) {
        __gm__ int32_t* shmemSignalAddr = MapVirtualAddr<int32_t>(hcclContext, shmemSignalBaseAddr, rankId) +
            static_cast<int32_t>(shmemSignalOffset1) * static_cast<int32_t>(shmemSignalRawShape2) * totalTileNum * stride +
            (static_cast<int32_t>(shmemSignalOffset2) * totalTileNum + tileIndex) * stride;
        ShapeDyn signalShape(1, 1, 1, 1, 1);
        StrideDyn signalStride(1, 1, 1, 1, 1);
        ShmemGlobalTensor<int32_t, 1, signalColShape> signalGlobal(shmemSignalAddr, signalShape, signalStride);
        
        if constexpr (atomicType == AtomicType::ADD) {
            pto::TSTORE<decltype(signalTile), decltype(signalGlobal), pto::AtomicType::AtomicAdd>(signalGlobal, signalTile);
        } else {
            pto::TSTORE<decltype(signalTile), decltype(signalGlobal), pto::AtomicType::AtomicNone>(signalGlobal, signalTile);
        }
    }
}

// Get: remote shmem GM → local GM.
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
    __gm__ ShmemType* shmemDataAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    CopyGmToGm<NonShmemType, NonShmemType, ShmemType, tileRowShape, tileColShape, bufferRowShape, bufferColShape, srcStride, dstStride, atomicType>(
        nonShmemDataAddr, buffer, shmemDataAddr);
}

// Get: remote shmem GM → UB (single block, optional type conversion).
template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemGetGm2Ub(__ubuf__ UBType* UBDataBaseAddr, __ubuf__ UBType* buffer, __gm__ ShmemType* shmemDataBaseAddr,
    uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0, uint32_t UBDataRawShape1,
    uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)tileRowShape;
    (void)tileColShape;
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    
    __ubuf__ UBType* UBDataAddr = UBDataBaseAddr + UBDataOffset0 * UBDataRawShape1 + UBDataOffset1;
    __gm__ ShmemType* shmemDataAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    CopyGmToUbBlock<UBType, ShmemType, bufferRowShape, bufferColShape, srcStride, dstStride>(UBDataAddr, buffer, shmemDataAddr);
}

// ---------------------------------------------------------------------------
// Shmem Reduce (GM load/store + UB reduce; FP32 or native type)
// ---------------------------------------------------------------------------
template<typename T, int64_t rowShape, int64_t colShape>
TILEOP void ReduceTLoad(__gm__ T* gmAddr, __ubuf__ T* ubAddr, CopyParams params)
{
    copy_gm_to_ubuf(ubAddr, gmAddr, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
}

template<typename T, int64_t rowShape, int64_t colShape>
TILEOP void ReduceTStore(__gm__ T* gmAddr, __ubuf__ T* ubAddr, CopyParams params)
{
    copy_ubuf_to_gm(gmAddr, ubAddr, 0, params.nBurst, params.lenBurst, params.srcStride, params.dstStride);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}

template<typename T, bool FP32Mode>
struct ShmemReduceProcess {};

constexpr uint64_t REDUCE_CHUNK_BYTES = 256;
constexpr uint64_t REDUCE_FP32_REPEAT_STRIDE = REDUCE_CHUNK_BYTES / sizeof(float);  // 64

template<typename T>
struct ShmemReduceProcess<T, true> {
    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        ReduceTLoad<T, rowShape, colShape>(x, copyUb, params);
        const uint64_t repeat = static_cast<uint64_t>(row) * static_cast<uint64_t>(col) * sizeof(float) / REDUCE_CHUNK_BYTES;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(sumUb + i * REDUCE_FP32_REPEAT_STRIDE, copyUb + i * REDUCE_FP32_REPEAT_STRIDE, 1, 1, 1, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        }
    }

    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        const uint64_t repeat = static_cast<uint64_t>(row) * static_cast<uint64_t>(col) * sizeof(float) / REDUCE_CHUNK_BYTES;
        for (uint64_t i = 0; i < repeat; i++) {
            DeConvFP32<T>(copyUb + i * REDUCE_FP32_REPEAT_STRIDE, sumUb + i * REDUCE_FP32_REPEAT_STRIDE, 1, 1, 1, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        }
        ReduceTStore<T, rowShape, colShape>(out, copyUb, params);
    }

    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        __ubuf__ float* tempUb = sumUb + row * col;
        ReduceTLoad<T, rowShape, colShape>(x, copyUb, params);
        const uint64_t repeat = static_cast<uint64_t>(row) * static_cast<uint64_t>(col) * sizeof(float) / REDUCE_CHUNK_BYTES;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(tempUb + i * REDUCE_FP32_REPEAT_STRIDE, copyUb + i * REDUCE_FP32_REPEAT_STRIDE, 1, 1, 1, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            vadd(sumUb + i * REDUCE_FP32_REPEAT_STRIDE, tempUb + i * REDUCE_FP32_REPEAT_STRIDE, sumUb + i * REDUCE_FP32_REPEAT_STRIDE, 1, 1, 1, 1, 8, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        }
    }
};

template<typename T>
struct ShmemReduceProcess<T, false> {
    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        ReduceTLoad<T, rowShape, colShape>(x, ubTensor, params);
    }

    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        ReduceTStore<T, rowShape, colShape>(out, ubTensor, params);
    }

    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* sumUb = ubTensor;
        __ubuf__ T* copyUb = ubTensor + row * col;
        ReduceTLoad<T, rowShape, colShape>(x, copyUb, params);
        const uint64_t repeat = static_cast<uint64_t>(row) * static_cast<uint64_t>(col) * sizeof(T) / REDUCE_CHUNK_BYTES;
        const uint64_t stride = REDUCE_CHUNK_BYTES / sizeof(T);
        for (uint64_t i = 0; i < repeat; i++) {
            vadd(sumUb + i * stride, copyUb + i * stride, sumUb + i * stride, 1, 1, 1, 1, 8, 8, 8);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        }
    }
};

template<typename T, bool FP32Mode, int64_t row, int64_t col>
TILEOP void ShmemReduce(__gm__ T* out, __ubuf__ T* ubTensor, __gm__ T* in, __gm__ T* shmData,
    int64_t rowOffset, int64_t colOffset, int64_t rowPerRank, int64_t colPerRank, __gm__ int64_t *hcclContext)
{
    // 暂时只支持二维的in和out
    __gm__ CommContext *winContext = (__gm__ CommContext *)(hcclContext[0]);    // 需要 hcclGroupIndex
    uint32_t localRankId = winContext->rankId;
    uint32_t rankSize = winContext->rankNum;
    int64_t offset = rowOffset * colPerRank + colOffset;
    out += offset;
    in += offset;
    shmData += offset;

    uint16_t nBurst = (uint16_t)row;
    uint16_t lenBurst = (uint16_t)(col * sizeof(T) / 32);
    uint16_t stride = (uint16_t)((colPerRank - col) * sizeof(T) / 32);
    CopyParams copyInParams{nBurst, lenBurst, stride, 0};
    CopyParams copyOutParams{nBurst, lenBurst, 0, stride};

    __gm__ T* x;
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        if (rankId == localRankId) {
            x = in + (uint64_t)rankId * rowPerRank * colPerRank;
        } else {
            x = shmData + (uint64_t)rankId * rowPerRank * colPerRank;
        }
        if (rankId == 0) {
            ShmemReduceProcess<T, FP32Mode>::template ShmemReduceCopyIn<row, col>(x, ubTensor, row, col, copyInParams);
        } else {
            ShmemReduceProcess<T, FP32Mode>::template ShmemReduceAdd<row, col>(x, ubTensor, row, col, copyInParams);
        }
    }
    ShmemReduceProcess<T, FP32Mode>::template ShmemReduceCopyOut<row, col>(out, ubTensor, row, col, copyOutParams);
}

} // namespace TileOp::Distributed
#endif