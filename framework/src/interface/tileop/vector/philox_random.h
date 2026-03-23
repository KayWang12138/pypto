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
 * \file philox_random.h
 * \brief Philox random number generator implementation
 */

#ifndef TILEOP_TILE_OPERATOR_PHILOX_RANDOM__H
#define TILEOP_TILE_OPERATOR_PHILOX_RANDOM__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

using PhiloxKey = uint32_t[2];
using PhiloxCounter = uint32_t[4];

template <uint16_t Rounds = 10, typename TileData>
TILEOP void TPhiloxRandom(TileData dstTile, PhiloxKey &philoxKey, PhiloxCounter &philoxCounter);

template <typename LastUse, typename DstTile>
TILEOP void PhiloxRandomImpl(DstTile dstTile, uint32_t *key, uint32_t *counter, uint16_t rounds) {
    PhiloxKey philoxKey = {key[0], key[1]};
    PhiloxCounter philoxCounter = {counter[0], counter[1], counter[2], counter[3]};
    
    if (rounds == 7) {
        TPhiloxRandom<7>(dstTile, philoxKey, philoxCounter);
    } else {
        TPhiloxRandom<10>(dstTile, philoxKey, philoxCounter);
    }
    
    counter[0] = philoxCounter[0];
    counter[1] = philoxCounter[1];
    counter[2] = philoxCounter[2];
    counter[3] = philoxCounter[3];
}

#define OP_TILE_OP_PHILOX_RANDOM TPhiloxRandomOp
template <typename LastUse = LastUse2Dim<0, 0>, typename T0>
TILEOP void TPhiloxRandomOp(T0 dst, uint32_t *key, uint32_t *counter, uint16_t rounds) {
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    constexpr auto dstTypeSize = sizeof(typename T0::Type);

    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();

    using DstTile = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    DstTile dstTile(dstShape3, dstShape4);

    uint64_t totalElements = dstShape0 * dstShape1 * dstShape2 * dstShape3 * dstShape4;
    uint64_t elementsPerTile = dstShape3 * dstShape4;
    
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                
                PhiloxRandomImpl<LastUse, DstTile>(dstTile, key, counter, rounds);
            }
        }
    }
}

#endif
