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

template<typename T, uint16_t rowShape, uint16_t colShape>
TILEOP void CopyGmToGm(__gm__ T* target, __ubuf__ T* buffer, __gm__ T* source)
{
    const uint16_t sid = 0;
    const uint16_t nBurst = rowShape;
    const uint16_t lenBurst = colShape * sizeof(T) / COPY_BLOCK_BYTE_SIZE;
    const uint16_t srcStride = 0;
    const uint16_t dstStride = 0;
    CopyGmToGmCore<T, sid, nBurst, lenBurst, srcStride, dstStride>(target, buffer, source);
}

template<typename T, uint16_t rowShape, uint16_t colShape>
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
    CopyGmToGm<T, rowShape, colShape>(shmemDataAddr, buffer, nonShmemDataAddr);
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

template<typename T, uint16_t rowShape, uint16_t colShape>
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
    CopyGmToGm<T, rowShape, colShape>(nonShmemDataAddr, buffer, shmemDataAddr);
}
} // namespace TileOp::Distributed
#endif