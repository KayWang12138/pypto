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

#define OP_TILE_OP_RANDOM TRandom
template <typename TDst>
TILEOP void TRandom(TDst dst, uint64_t key, uint64_t *counter, uint16_t rounds, int64_t tileOffset) {
    constexpr auto shapeSize = Std::tuple_size<typename TDst::Shape>::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename TDst::TileShape>::type::value;

    constexpr size_t expectSize = 1;
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<0, expectSize>();

    uint64_t tileCounter[2] = {counter[0], counter[1]};
    SkipCounter(tileCounter, static_cast<uint64_t>(tileOffset));

    using TileDst = pto::Tile<pto::TileType::Vec, typename TDst::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    TileDst dstTile(1, shape0);
    pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());

    pto::TRandomKey randomKey = {static_cast<uint32_t>(key & 0xFFFFFFFF),
                                  static_cast<uint32_t>(key >> 32)};
    pto::TRandomCounter randomCounter = {static_cast<uint32_t>(tileCounter[0] & 0xFFFFFFFF),
                                          static_cast<uint32_t>(tileCounter[0] >> 32),
                                          static_cast<uint32_t>(tileCounter[1] & 0xFFFFFFFF),
                                          static_cast<uint32_t>(tileCounter[1] >> 32)};

    if (rounds == 7) {
        pto::TRandom<7>(dstTile, randomKey, randomCounter);
    } else {
        pto::TRandom<10>(dstTile, randomKey, randomCounter);
    }
}

#endif
