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
 * \file uniform.h
 * \brief Uniform random number generator implementation
 */

#ifndef TILEOP_TILE_OPERATOR_UNIFORM__H
#define TILEOP_TILE_OPERATOR_UNIFORM__H

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

#define OP_TILE_OP_UNIFORM TUniform
template <typename T, int Size>
TILEOP void TUniform(__ubuf__ T* dst, uint64_t key, uint64_t counter0, uint64_t counter1, uint16_t rounds, int64_t tileOffset) {
    constexpr int tileW = (Size + 7) / 8 * 8;

    uint64_t tileCounter[2] = {counter0, counter1};
    // SkipCounter(tileCounter, static_cast<uint64_t>(tileOffset));

    // 使用临时buffer存储uint32结果
    __ubuf__ uint32_t uint32Buffer[Size];
    
    using TileUint32 = pto::Tile<pto::TileType::Vec, uint32_t, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    TileUint32 uint32Tile(1, Size);
    pto::TASSIGN(uint32Tile, (uint64_t)uint32Buffer);

    pto::TRandomKey uniformKey = {static_cast<uint32_t>(key & 0xFFFFFFFF),
                                   static_cast<uint32_t>(key >> 32)};
    pto::TRandomCounter uniformCounter = {static_cast<uint32_t>(tileCounter[0] & 0xFFFFFFFF),
                                           static_cast<uint32_t>(tileCounter[0] >> 32),
                                           static_cast<uint32_t>(tileCounter[1] & 0xFFFFFFFF),
                                           static_cast<uint32_t>(tileCounter[1] >> 32)};

    if (rounds == 7) {
        pto::TRandom<7, TileUint32>(uint32Tile.data(), uniformKey, uniformCounter, 1, Size);
    } else {
        pto::TRandom<10, TileUint32>(uint32Tile.data(), uniformKey, uniformCounter, 1, Size);
    }

    // 将uint32转换为float
    // 逻辑：const uint32_t man = x & 0x7fffffu; // 23 bit mantissa
    //       const uint32_t exp = static_cast<uint32_t>(127);  // 7 bit exp
    //       const uint32_t val = (exp << 23) | man;
    //       float result; memcpy(&result, &val, sizeof(val));
    //       return result - 1.0f;
    
    // 1. 提取23位尾数：man = x & 0x7fffff
    pto::TANDS(uint32Tile, uint32Tile, 0x7fffff);
    
    // 2. 构造float的位模式：val = (127 << 23) | man
    //    先将exp左移23位：127 << 23 = 0x3f800000
    pto::TORS(uint32Tile, uint32Tile, 0x3f800000);
    
    // 3. 将uint32的位模式重新解释为float
    //    使用reinterpret_cast将uint32 buffer的地址转换为float指针
    __ubuf__ float* floatBuffer = reinterpret_cast<__ubuf__ float*>(uint32Buffer);
    
    using TileFloat = pto::Tile<pto::TileType::Vec, float, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    TileFloat floatTile(1, Size);
    pto::TASSIGN(floatTile, (uint64_t)floatBuffer);
    
    // 4. 减去1.0：result - 1.0f
    pto::TSUBS(floatTile, floatTile, 1.0f);
    
    // 5. 将结果写回dst
    TileFloat dstTile(1, Size);
    pto::TASSIGN(dstTile, (uint64_t)dst);
    pto::TMOV(dstTile, floatTile);
}

#endif
