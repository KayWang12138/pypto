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
 * \file index_outcast.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H
#define TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T, typename T2, unsigned src0OriShape1, unsigned src1OriShape1, unsigned src0rawShape1, unsigned cacheMode, unsigned blockSize>
TILEOP void TIndexoutcastBase(__gm__ T *dst, __ubuf__ T *src0, __ubuf__ T2 *src1, unsigned GmShape1) {
    for (auto i = 0; i < src1OriShape1; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        T2 curValue = *(reinterpret_cast<__ubuf__ T2 *>(src1 + i));
        if constexpr (cacheMode == 1) { // PA_NZ
            T2 blockCount = curValue / blockSize;
            T2 index = curValue % blockSize;

            __gm__ T *new_dst = dst + blockCount * blockSize * GmShape1 + index * 32 / sizeof(T);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

            // === 替换为 MCOPY（使用 Tile）===
            pto::Tile<pto::TileType::Vec, T, 1, src0OriShape1> src_tile;
            pto::Tile<pto::TileType::Vec, T, 1, src0OriShape1> dst_tile;
            pto::TASSIGN(src_tile, reinterpret_cast<uint64_t>(src0 + i * src0OriShape1));
            pto::TASSIGN(dst_tile, reinterpret_cast<uint64_t>(new_dst));
            pto::MCOPY(dst_tile, src_tile);
#ifdef __DAV_V220
            pipe_barrier(PIPE_MTE3);
#endif

        } else {
            __gm__ T *new_dst = dst + curValue * GmShape1;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

            // === 替换为 MCOPY ===
            pto::Tile<pto::TileType::Vec, T, 1, src0OriShape1> src_tile;
            pto::Tile<pto::TileType::Vec, T, 1, src0OriShape1> dst_tile;
            pto::TASSIGN(src_tile, reinterpret_cast<uint64_t>(src0 + i * src0rawShape1));
            pto::TASSIGN(dst_tile, reinterpret_cast<uint64_t>(new_dst));
            pto::MCOPY(dst_tile, src_tile);
#ifdef __DAV_V220
            pipe_barrier(PIPE_MTE3);
#endif
        }
    }
}

template <typename T, typename T2>
TILEOP void DynTIndexoutcast(__gm__ T *dst, __ubuf__ T *src, __ubuf__ T2 *index, unsigned src1OriShape0,
    unsigned src1OriShape1, unsigned src1rawShape1, unsigned src0OriShape3, unsigned src0rawShape1, unsigned src0rawShape3) {
    unsigned b = src1OriShape0;
    unsigned s1 = src1OriShape1;
    unsigned s1_32aligned = src1rawShape1;
    unsigned nd = src0OriShape3;
    unsigned nd_32aligned = src0rawShape3;

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    __gm__ T *curDst = dst;
    __ubuf__ T2 *dstIdx = index;
    __ubuf__ T *curSrc = src;
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < s1; ++j) {
            curDst = dst + *dstIdx * nd;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

            // === 替换为 MCOPY ===
            pto::Tile<pto::TileType::Vec, T, 1, nd> src_tile;
            pto::Tile<pto::TileType::Vec, T, 1, nd> dst_tile;
            pto::TASSIGN(src_tile, reinterpret_cast<uint64_t>(curSrc));
            pto::TASSIGN(dst_tile, reinterpret_cast<uint64_t>(curDst));
            pto::MCOPY(dst_tile, src_tile);
#ifdef __DAV_V220
            pipe_barrier(PIPE_MTE3);
#endif

            curSrc += nd_32aligned;
            dstIdx++;
        }
        curSrc += (src0rawShape1 - s1) * nd_32aligned;
        dstIdx += s1_32aligned - s1;
    }
}

// 主入口：无裸指令，保持不变
template <typename T, typename T2, unsigned src0OriShape0, unsigned src0OriShape1, unsigned src0OriShape3,
    unsigned src0rawShape1, unsigned src0rawShape2, unsigned src0rawShape3, unsigned src1OriShape0,
    unsigned src1OriShape1, unsigned src1rawShape3, unsigned cacheMode, unsigned blockSize>
TILEOP void DynTIndexoutcast(__gm__ T *dst, __ubuf__ T *src0, __ubuf__ T2 *src1, unsigned GmShape0, unsigned GmShape1, 
    unsigned GmShape2, unsigned GmShape3, unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3) {
    if (src0OriShape0 == 0 || src0OriShape1 == 0 || src0OriShape3 == 0 || src1OriShape0 == 0 || src1OriShape1 == 0) {
        return;
    }
    if (cacheMode == 2) {
        DynTIndexoutcast<T, T2>(dst, src0, src1, src1OriShape0, src1OriShape1, src1rawShape3,
            src0OriShape3, src0rawShape1, src0rawShape3);
        return;
    }
    dst += CalcLinearOffset(GmShape1, GmShape2, GmShape3, Offset0, Offset1, Offset2, Offset3);

    static_assert(src0OriShape1 == 1, "src0OriShape1 now only support 1");

    int alignTS2TS3 = src0rawShape2 * src0rawShape3;
    int alignSrc1 = src1rawShape3;
    for (int i = 0; i < src0OriShape0; ++i) {
        for (int j = 0; j < src0OriShape1; ++j) {
            TIndexoutcastBase<T, T2, src0OriShape3,
                src1OriShape1, src0rawShape3, cacheMode, blockSize>(dst, src0, src1, GmShape3);
        }
        src0 += alignTS2TS3;
        src1 += alignSrc1;
        dst += GmShape2 * GmShape3;
    }
}