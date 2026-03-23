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
 * \file random.h
 * \brief Random number generator implementation
 */

#ifndef TILEOP_TILE_OPERATOR_RANDOM__H
#define TILEOP_TILE_OPERATOR_RANDOM__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

using RandomKey = uint64_t;
using RandomCounter = uint32_t[4];

template <uint16_t Rounds = 10, typename TileData>
TILEOP void TRandom(TileData dstTile, RandomKey &randomKey, RandomCounter &randomCounter);

template <typename LastUse, typename DstTile>
TILEOP void RandomImpl(DstTile dstTile, uint64_t key, uint64_t *counter, uint16_t rounds) {
    RandomKey randomKey = key;
    RandomCounter randomCounter = {static_cast<uint32_t>(counter[0] & 0xFFFFFFFF),
                                    static_cast<uint32_t>(counter[0] >> 32),
                                    static_cast<uint32_t>(counter[1] & 0xFFFFFFFF),
                                    static_cast<uint32_t>(counter[1] >> 32)};
    
    if (rounds == 7) {
        TRandom<7>(dstTile, randomKey, randomCounter);
    } else {
        TRandom<10>(dstTile, randomKey, randomCounter);
    }
    
    counter[0] = static_cast<uint64_t>(randomCounter[0]) | (static_cast<uint64_t>(randomCounter[1]) << 32);
    counter[1] = static_cast<uint64_t>(randomCounter[2]) | (static_cast<uint64_t>(randomCounter[3]) << 32);
}

template <typename T>
TILEOP void SkipCounter(T *counter, uint64_t offset) {
    uint32_t offsetLo = static_cast<uint32_t>(offset);
    uint32_t offsetHi = static_cast<uint32_t>(offset >> 32);
    
    uint32_t c0 = static_cast<uint32_t>(counter[0] & 0xFFFFFFFF);
    uint32_t c1 = static_cast<uint32_t>(counter[0] >> 32);
    uint32_t c2 = static_cast<uint32_t>(counter[1] & 0xFFFFFFFF);
    uint32_t c3 = static_cast<uint32_t>(counter[1] >> 32);
    
    c0 += offsetLo;
    if (c0 < offsetLo) {
        ++offsetHi;
    }
    c1 += offsetHi;
    if (c1 < offsetHi) {
        if (++c2 == 0) {
            ++c3;
        }
    }
    
    counter[0] = static_cast<uint64_t>(c0) | (static_cast<uint64_t>(c1) << 32);
    counter[1] = static_cast<uint64_t>(c2) | (static_cast<uint64_t>(c3) << 32);
}

#define OP_TILE_OP_RANDOM TRandomOp
template <typename LastUse = LastUse2Dim<0, 0>, typename T0>
TILEOP void TRandomOp(T0 dst, uint64_t key, uint64_t *counter, uint16_t rounds, int64_t tileOffset) {
    constexpr size_t expectSize = 1;
    const auto dstLayout = dst.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();

    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 0, expectSize>();

    using DstTile = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    DstTile dstTile(1, dstShape0);

    uint64_t tileCounter[2] = {counter[0], counter[1]};
    SkipCounter(tileCounter, static_cast<uint64_t>(tileOffset));
    
    RandomImpl<LastUse, DstTile>(dstTile, key, tileCounter, rounds);
}

#endif
