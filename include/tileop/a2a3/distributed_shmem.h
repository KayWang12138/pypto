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
TILEOP void ShmemPut(__gm__ T* dummy, __ubuf__ T* buffer, __gm__ T* nonShmemData, __gm__ T* shmemData,
    __gm__ int64_t *hcclContext)
{
    (void)dummy;
    CopyGmToGm<T, rowShape, colShape>(shmemData, buffer, nonShmemData);
}

template<typename T, int32_t value, AtomicType atomicType>
TILEOP void ShmemSignal(__gm__ int32_t* shmemSignal, __ubuf__ int32_t* buffer, __gm__ T* dummy,
    __gm__ int64_t *hcclContext)
{
    (void)dummy;
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
    copy_ubuf_to_gm(shmemSignal, buffer, sid, nBurst, lenBurst, dstStride, srcStride);
    if constexpr (atomicType == AtomicType::ADD) {
        set_atomic_none();
    }
}

template<typename T, uint16_t rowShape, uint16_t colShape>
TILEOP void ShmemGet(__gm__ T* nonShmemData, __ubuf__ T* buffer, __gm__ T* dummy, __gm__ T* shmemData,
    __gm__ int64_t *hcclContext)
{
    (void)dummy;
    CopyGmToGm<T, rowShape, colShape>(nonShmemData, buffer, shmemData);
}
} // namespace TileOp::Distributed
#endif