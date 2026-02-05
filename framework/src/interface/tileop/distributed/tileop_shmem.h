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
 * \file mte.h
 * \brief
*/

#ifndef __DISTRIBUTED_SHMEM__
#define __DISTRIBUTED_SHMEM__

#include "common.h"
#include "hccl_context.h"

#include <type_traits>

#include "pto/pto-inst.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/type.hpp"

// Pipe synchronization macros
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

// Type aliases for Shmem operations
using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template<typename T, uint32_t RowShape, uint32_t ColShape>
using ShmemGlobalTensor = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template<typename T, uint32_t RowShape, uint32_t ColShape>
using ShmemUbTile = pto::Tile<pto::TileType::Vec, T, RowShape, ColShape, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC>;

template<typename T, uint32_t bufferEleNum, uint32_t shmemTensorRawShape1, uint32_t shmemTensorRawShape2, uint32_t shmemTensorRawShape3>
TILEOP void ShmemClear(__ubuf__ T* buffer, __gm__ T* shmemTensorAddr)
{
    constexpr uint8_t repeat = sizeof(T) * bufferEleNum / VECTOR_INSTRUCTION_BYTE_SIZE;
    vector_dup(buffer, static_cast<T>(0), repeat, 1, 0, 8, 0);
    PIPE_SYNC_V_MTE3();

    constexpr uint32_t shmemTensorEleNum = shmemTensorRawShape1 * shmemTensorRawShape2 * shmemTensorRawShape3;
    constexpr uint32_t fullChunkCount = shmemTensorEleNum / bufferEleNum;

    // Use pto instructions to copy buffer to GM
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

// Operation type for ShmemCopyCore
enum class ShmemOp { PUT, GET };

// Core function for ShmemPut and ShmemGet operations with row chunking
template<ShmemOp op, typename SrcType, typename DstType, uint32_t tileRowShape, uint32_t tileColShape,
    uint32_t bufferRowShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType, typename BufferType>
TILEOP void ShmemCopyCore(__ubuf__ BufferType* buffer, __gm__ SrcType* srcAddr, __gm__ DstType* dstAddr) {
    // Calculate chunk parameters - directly use bufferRowShape for chunking
    constexpr uint32_t chunkRowShape = tileRowShape < bufferRowShape ? tileRowShape : bufferRowShape;
    constexpr uint32_t rowChunkCount = (tileRowShape + chunkRowShape - 1) / chunkRowShape;
    constexpr bool needTypeConversion = !std::is_same_v<SrcType, DstType>;
    
    // Calculate buffer offset for type conversion
    constexpr uint32_t bufferLen = bufferRowShape * AlignUp<uint32_t>(tileColShape * sizeof(BufferType), 32) / sizeof(BufferType);
    using ConvertedBufferType = std::conditional_t<op == ShmemOp::PUT, __ubuf__ DstType*, __ubuf__ SrcType*>;
    auto convertedBuffer = reinterpret_cast<ConvertedBufferType>(buffer + (needTypeConversion ? bufferLen : 0));
    
    __gm__ SrcType* currentSrcAddr = srcAddr;
    __gm__ DstType* currentDstAddr = dstAddr;
    
    for (uint32_t chunkIdx = 0; chunkIdx < rowChunkCount; ++chunkIdx) {
        uint32_t remainingRows = tileRowShape - chunkIdx * chunkRowShape;
        if (remainingRows == 0) break;
        uint32_t currentChunkRows = (remainingRows < chunkRowShape) ? remainingRows : chunkRowShape;
        
        // Wait for previous chunk completion when doing type conversion
        if constexpr (needTypeConversion) {
            if (chunkIdx > 0) {
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                PIPE_SYNC_S_MTE2();
            }
        }
        
        // Create shape and stride descriptors
        ShapeDyn shape(1, 1, 1, currentChunkRows, tileColShape);
        StrideDyn srcStrideDyn(currentChunkRows, currentChunkRows, currentChunkRows, srcStride, 1);
        StrideDyn dstStrideDyn(currentChunkRows, currentChunkRows, currentChunkRows, dstStride, 1);
        
        ShmemGlobalTensor<SrcType, chunkRowShape, tileColShape> srcGlobal(currentSrcAddr, shape, srcStrideDyn);
        ShmemGlobalTensor<DstType, chunkRowShape, tileColShape> dstGlobal(currentDstAddr, shape, dstStrideDyn);
        
        if constexpr (!needTypeConversion) {
            // Same type: use TPUT/TGET directly
            ShmemUbTile<BufferType, chunkRowShape, tileColShape> ubTile(currentChunkRows, tileColShape);
            pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(buffer));
            
            if constexpr (op == ShmemOp::PUT) {
                if constexpr (atomicType == AtomicType::ADD) {
                    pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstGlobal, srcGlobal, ubTile);
                } else {
                    pto::comm::TPUT<pto::AtomicType::AtomicNone>(dstGlobal, srcGlobal, ubTile);
                }
            } else {
                pto::comm::TGET(dstGlobal, srcGlobal, ubTile);
            }
        } else {
            // Different types: need TLOAD + TCVT + TSTORE
            ShmemUbTile<SrcType, chunkRowShape, tileColShape> srcTile(currentChunkRows, tileColShape);
            ShmemUbTile<DstType, chunkRowShape, tileColShape> dstTile(currentChunkRows, tileColShape);
            
            if constexpr (op == ShmemOp::PUT) {
                pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(buffer));
                pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(convertedBuffer));
            } else {
                pto::TASSIGN(srcTile, reinterpret_cast<uintptr_t>(convertedBuffer));
                pto::TASSIGN(dstTile, reinterpret_cast<uintptr_t>(buffer));
            }
            
            // Load source data
            pto::TLOAD(srcTile, srcGlobal);
            PIPE_SYNC_MTE2_V();
            
            // Type conversion
            pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
            PIPE_SYNC_V_MTE3();
            
            // Store converted data
            if constexpr (atomicType == AtomicType::ADD) {
                pto::TSTORE<decltype(dstTile), decltype(dstGlobal), pto::AtomicType::AtomicAdd>(dstGlobal, dstTile);
            } else {
                pto::TSTORE<decltype(dstTile), decltype(dstGlobal), pto::AtomicType::AtomicNone>(dstGlobal, dstTile);
            }
            PIPE_SYNC_MTE3_MTE2();
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
        }
        
        currentSrcAddr += currentChunkRows * srcStride;
        currentDstAddr += currentChunkRows * dstStride;
    }
    
    // Final synchronization for type conversion path
    if constexpr (needTypeConversion) {
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
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
    
    __gm__ NonShmemType* srcAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ ShmemType* dstAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    ShmemCopyCore<ShmemOp::PUT, NonShmemType, ShmemType, tileRowShape, tileColShape, bufferRowShape, srcStride, dstStride, atomicType>(
        buffer, srcAddr, dstAddr);
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
    
    __gm__ InShmemType* srcAddr = MapVirtualAddr<InShmemType>(hcclContext, inShmemDataBaseAddr, inShmemDataOffset0) +
        inShmemDataOffset1 * inShmemDataRawShape2 * inShmemDataRawShape3 + inShmemDataOffset2 * inShmemDataRawShape3 + inShmemDataOffset3;
    __gm__ OutShmemType* dstAddr = MapVirtualAddr<OutShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    ShmemCopyCore<ShmemOp::PUT, InShmemType, OutShmemType, tileRowShape, tileColShape, bufferRowShape, srcStride, dstStride, atomicType>(
        buffer, srcAddr, dstAddr);
}

template<typename UBType, typename ShmemType, uint32_t tileRowShape, uint32_t tileColShape, uint32_t bufferRowShape,
    uint32_t bufferColShape, uint32_t srcStride, uint32_t dstStride, AtomicType atomicType>
TILEOP void ShmemPutUb2Gm(__ubuf__ UBType* UBDataBaseAddr, __gm__ ShmemType* shmemDataBaseAddr, uint32_t UBDataOffset0, uint32_t UBDataOffset1, uint32_t UBDataRawShape0,
    uint32_t UBDataRawShape1, uint32_t shmemDataOffset0, uint32_t shmemDataOffset1, uint32_t shmemDataOffset2, uint32_t shmemDataOffset3,
    uint32_t shmemDataRawShape0, uint32_t shmemDataRawShape1, uint32_t shmemDataRawShape2, uint32_t shmemDataRawShape3, __gm__ int64_t *hcclContext)
{
    (void)UBDataRawShape0;
    (void)shmemDataRawShape0;
    (void)bufferColShape;
    
    __ubuf__ UBType* ubAddr = UBDataBaseAddr + UBDataOffset0 * UBDataRawShape1 + UBDataOffset1;
    __gm__ ShmemType* gmAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    ShapeDyn shape(1, 1, 1, tileRowShape, tileColShape);
    StrideDyn strideDyn(tileRowShape, tileRowShape, tileRowShape, dstStride, 1);
    ShmemGlobalTensor<ShmemType, tileRowShape, tileColShape> gmTensor(gmAddr, shape, strideDyn);
    ShmemUbTile<UBType, tileRowShape, tileColShape> ubTile(tileRowShape, tileColShape);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(ubAddr));
    
    PIPE_SYNC_S_MTE3();
    
    if constexpr (atomicType == AtomicType::ADD) {
        pto::TSTORE<decltype(ubTile), decltype(gmTensor), pto::AtomicType::AtomicAdd>(gmTensor, ubTile);
    } else {
        pto::TSTORE<decltype(ubTile), decltype(gmTensor), pto::AtomicType::AtomicNone>(gmTensor, ubTile);
    }
    
    PIPE_SYNC_MTE3_S();
}

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

    // Set signal value in buffer
    buffer[0] = static_cast<int32_t>(value);
    
    // Use 1x8 tile shape to meet 32-byte alignment requirement (8 * sizeof(int32_t) = 32)
    constexpr uint32_t signalColShape = 8;
    ShmemUbTile<int32_t, 1, signalColShape> signalTile(1, 1);
    pto::TASSIGN(signalTile, reinterpret_cast<uintptr_t>(buffer));
    
    // Synchronize before writing
    PIPE_SYNC_S_MTE3();
    
    for (uint32_t rankId = shmemSignalOffset0; rankId < shmemSignalOffset0 + shmemSignalShape0; rankId++) {
        __gm__ int32_t* shmemSignalAddr = MapVirtualAddr<int32_t>(hcclContext, shmemSignalBaseAddr, rankId) +
            static_cast<int32_t>(shmemSignalOffset1) * static_cast<int32_t>(shmemSignalRawShape2) * totalTileNum * stride +
            (static_cast<int32_t>(shmemSignalOffset2) * totalTileNum + tileIndex) * stride;
        
        // Create GlobalTensor for destination
        ShapeDyn signalShape(1, 1, 1, 1, 1);
        StrideDyn signalStride(1, 1, 1, 1, 1);
        ShmemGlobalTensor<int32_t, 1, signalColShape> signalGlobal(shmemSignalAddr, signalShape, signalStride);
        
        // Store signal with atomic operation
        if constexpr (atomicType == AtomicType::ADD) {
            pto::TSTORE<decltype(signalTile), decltype(signalGlobal), pto::AtomicType::AtomicAdd>(signalGlobal, signalTile);
        } else {
            pto::TSTORE<decltype(signalTile), decltype(signalGlobal), pto::AtomicType::AtomicNone>(signalGlobal, signalTile);
        }
    }
    
    // Synchronize after writing
    PIPE_SYNC_MTE3_S();
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
    
    __gm__ NonShmemType* dstAddr = nonShmemDataBaseAddr + nonShmemDataOffset0 * nonShmemDataRawShape1 + nonShmemDataOffset1;
    __gm__ ShmemType* srcAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    ShmemCopyCore<ShmemOp::GET, ShmemType, NonShmemType, tileRowShape, tileColShape, bufferRowShape, srcStride, dstStride, atomicType>(
        buffer, srcAddr, dstAddr);
}

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
    (void)buffer;
    
    __ubuf__ UBType* ubAddr = UBDataBaseAddr + UBDataOffset0 * UBDataRawShape1 + UBDataOffset1;
    __gm__ ShmemType* gmAddr = MapVirtualAddr<ShmemType>(hcclContext, shmemDataBaseAddr, shmemDataOffset0) +
        shmemDataOffset1 * shmemDataRawShape2 * shmemDataRawShape3 + shmemDataOffset2 * shmemDataRawShape3 + shmemDataOffset3;
    
    ShapeDyn shape(1, 1, 1, bufferRowShape, bufferColShape);
    StrideDyn strideDyn(bufferRowShape, bufferRowShape, bufferRowShape, srcStride, 1);
    ShmemGlobalTensor<ShmemType, bufferRowShape, bufferColShape> gmTensor(gmAddr, shape, strideDyn);
    ShmemUbTile<UBType, bufferRowShape, bufferColShape> ubTile(bufferRowShape, bufferColShape);
    pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(ubAddr));
    
    PIPE_SYNC_S_MTE2();
    
    pto::TLOAD(ubTile, gmTensor);
    
    PIPE_SYNC_MTE2_S();
}

template<typename T, bool FP32Mode>
struct ShmemReduceProcess {};

// 类模板部分特化
template<typename T>
struct ShmemReduceProcess<T, true> {
    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);

        ShapeDyn shape(1, 1, 1, rowShape, colShape);
        StrideDyn srcStrideDyn(rowShape, rowShape, rowShape, params.srcStride + colShape, 1);
        ShmemGlobalTensor<T, rowShape, colShape> gmTensor(x, shape, srcStrideDyn);
        ShmemUbTile<T, rowShape, colShape> ubTile(rowShape, colShape);
        pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(copyUb));

        PIPE_SYNC_S_MTE2();
        pto::TLOAD(ubTile, gmTensor);
        PIPE_SYNC_MTE2_S();

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ T* src = copyUb;
        __ubuf__ float* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(dst, src, 1, 1, 1, 8, 8);
            PIPE_SYNC_V_S();
            src += offset;
            dst += offset;
        }
    }

    template<int64_t rowShape, int64_t colShape>
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
            PIPE_SYNC_V_S();
            src += offset;
            dst += offset;
        }

        ShapeDyn shapeOut(1, 1, 1, rowShape, colShape);
        StrideDyn dstStrideDyn(rowShape, rowShape, rowShape, params.dstStride + colShape, 1);
        ShmemGlobalTensor<T, rowShape, colShape> gmTensorOut(out, shapeOut, dstStrideDyn);
        ShmemUbTile<T, rowShape, colShape> ubTileOut(rowShape, colShape);
        pto::TASSIGN(ubTileOut, reinterpret_cast<uintptr_t>(copyUb));

        PIPE_SYNC_S_MTE3();
        pto::TSTORE<decltype(ubTileOut), decltype(gmTensorOut), pto::AtomicType::AtomicNone>(gmTensorOut, ubTileOut);
        PIPE_SYNC_MTE3_S();
    }

    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* copyUb = ubTensor;
        __ubuf__ float* sumUb = (__ubuf__ float*)(ubTensor + row * col);
        __ubuf__ float* tempUb = sumUb + row * col;

        ShapeDyn shapeAdd(1, 1, 1, rowShape, colShape);
        StrideDyn srcStrideDynAdd(rowShape, rowShape, rowShape, params.srcStride + colShape, 1);
        ShmemGlobalTensor<T, rowShape, colShape> gmTensorAdd(x, shapeAdd, srcStrideDynAdd);
        ShmemUbTile<T, rowShape, colShape> ubTileAdd(rowShape, colShape);
        pto::TASSIGN(ubTileAdd, reinterpret_cast<uintptr_t>(copyUb));

        PIPE_SYNC_S_MTE2();
        pto::TLOAD(ubTileAdd, gmTensorAdd);
        PIPE_SYNC_MTE2_S();

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ float* src = copyUb;
        __ubuf__ T* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            Conv2FP32<T>(tempUb, src, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(dst, tempUb, dst, 1, 1, 1, 1, 8, 8, 8);
            PIPE_SYNC_V_S();
            src += offset;
            dst += offset;
        }
    }
};

template<typename T>
struct ShmemReduceProcess<T, false> {
    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceCopyIn(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        ShapeDyn shape(1, 1, 1, rowShape, colShape);
        StrideDyn srcStrideDyn(rowShape, rowShape, rowShape, params.srcStride + colShape, 1);
        ShmemGlobalTensor<T, rowShape, colShape> gmTensor(x, shape, srcStrideDyn);
        ShmemUbTile<T, rowShape, colShape> ubTile(rowShape, colShape);
        pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(ubTensor));

        PIPE_SYNC_S_MTE2();
        pto::TLOAD(ubTile, gmTensor);
        PIPE_SYNC_MTE2_S();
    }

    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceCopyOut(__gm__ T* out, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        ShapeDyn shape(1, 1, 1, rowShape, colShape);
        StrideDyn dstStrideDyn(rowShape, rowShape, rowShape, params.dstStride + colShape, 1);
        ShmemGlobalTensor<T, rowShape, colShape> gmTensor(out, shape, dstStrideDyn);
        ShmemUbTile<T, rowShape, colShape> ubTile(rowShape, colShape);
        pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(ubTensor));

        PIPE_SYNC_S_MTE3();
        pto::TSTORE<decltype(ubTile), decltype(gmTensor), pto::AtomicType::AtomicNone>(gmTensor, ubTile);
        PIPE_SYNC_MTE3_S();
    }

    template<int64_t rowShape, int64_t colShape>
    TILEOP void ShmemReduceAdd(__gm__ T* x, __ubuf__ T* ubTensor, int64_t row, int64_t col, CopyParams params)
    {
        __ubuf__ T* sumUb = ubTensor;
        __ubuf__ T* copyUb = ubTensor + row * col;

        ShapeDyn shape(1, 1, 1, rowShape, colShape);
        StrideDyn srcStrideDyn(rowShape, rowShape, rowShape, params.srcStride + colShape, 1);
        ShmemGlobalTensor<T, rowShape, colShape> gmTensor(x, shape, srcStrideDyn);
        ShmemUbTile<T, rowShape, colShape> ubTile(rowShape, colShape);
        pto::TASSIGN(ubTile, reinterpret_cast<uintptr_t>(copyUb));

        PIPE_SYNC_S_MTE2();
        pto::TLOAD(ubTile, gmTensor);
        PIPE_SYNC_MTE2_S();

        uint64_t repeat = row * col * sizeof(float) / 256;
        uint64_t offset = 256 / sizeof(float);
        __ubuf__ T* src = copyUb;
        __ubuf__ T* dst = sumUb;
        for (uint64_t i = 0; i < repeat; i++) {
            vadd(dst, src, dst, 1, 1, 1, 1, 8, 8, 8);
            PIPE_SYNC_V_S();
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
            ShmemReduceProcess<T, FP32Mode>::template ShmemReduceCopyIn<row, col>(x, ubTensor, row, col, copyInParams);
        } else {
            ShmemReduceProcess<T, FP32Mode>::template ShmemReduceAdd<row, col>(x, ubTensor, row, col, copyInParams);
        }
    }
    ShmemReduceProcess<T, FP32Mode>::template ShmemReduceCopyOut<row, col>(out, ubTensor, row, col, copyOutParams);
}

} // namespace TileOp::Distributed
#endif