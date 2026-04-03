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
 * \file comm_context.h
 * \brief
 */

#ifndef __COMM_CONTEXT__
#define __COMM_CONTEXT__

#include <type_traits>

namespace TileOp {
struct CommContext {
    uint64_t rankId = 0;    // 当前卡rankId
    uint64_t rankNum = 0;
    int64_t startIndex = 0; // 每个win区起始Index
    int64_t statusIndex = -1;
    int64_t debugIndex = -1;
    uint64_t winDataSize = 0; // 每个win区大小
    uint64_t winStatusSize = 0;
    uint64_t winDebugSize = 0;
    uint64_t totalWinNum = 0;
    uint64_t
        winAddr[0]; // size大小rankNum*3，内存排布windata[0~rankNum-1], winStatus[0~rankNum-1], winDebug[0~rankNum-1]
};

namespace Distributed {
constexpr uint64_t OFFSET_BITS = 42UL;
constexpr uint64_t TILE_NUM_BITS = 12UL;
constexpr uint64_t GROUP_BITS = 2UL;
constexpr uint64_t MEMTYPE_BITS = 2UL;

constexpr uint64_t TILE_NUM_SHIFT = OFFSET_BITS;
constexpr uint64_t GROUP_SHIFT = TILE_NUM_SHIFT + TILE_NUM_BITS;
constexpr uint64_t MEMTYPE_SHIFT = GROUP_SHIFT + GROUP_BITS;
constexpr uint64_t FILL_SHIFT = MEMTYPE_SHIFT + MEMTYPE_BITS;

constexpr uint64_t OFFSET_MASK = (1UL << OFFSET_BITS) - 1UL;
constexpr uint64_t GROUP_MASK = (1UL << GROUP_BITS) - 1UL;
constexpr uint64_t MEMTYPE_MASK = (1UL << MEMTYPE_BITS) - 1UL;
constexpr uint64_t TILE_NUM_MASK = (1UL << TILE_NUM_BITS) - 1UL;

#define ENCODE_SHMEM_ADDR(offset, maxTileNum, groupIndex, memType)                                                \
    ((offset) | ((maxTileNum) << TILE_NUM_SHIFT) | ((groupIndex) << GROUP_SHIFT) | ((memType) << MEMTYPE_SHIFT) | \
     (1UL << FILL_SHIFT))

#define DECODE_SHMEM_ADDR_OFFSET(val) ((val)&OFFSET_MASK)

#define DECODE_SHMEM_ADDR_MAX_TILE_NUM(val) (((val) >> TILE_NUM_SHIFT) & TILE_NUM_MASK)

#define DECODE_SHMEM_ADDR_GROUP_INDEX(val) (((val) >> GROUP_SHIFT) & GROUP_MASK)

#define DECODE_SHMEM_ADDR_MEMTYPE(val) (((val) >> MEMTYPE_SHIFT) & MEMTYPE_MASK)
} // namespace Distributed
} // namespace TileOp

#endif
