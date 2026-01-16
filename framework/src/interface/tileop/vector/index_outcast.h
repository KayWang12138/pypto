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
#ifndef TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H
#define TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <unsigned cacheMode, unsigned blockSize, typename T0, typename T1, typename T2>
TILEOP void TIndexOutcast(T0 dst, T1 src, T2 src1)
{
    constexpr auto expectSize = 5; // 所有张量均为 5D

    // src: data [1, B, S, 1, D]
    const auto uLayout = src.GetLayout();
    auto uShape1 = uLayout.template GetShapeDim<1, expectSize>(); // B
    auto uShape2 = uLayout.template GetShapeDim<2, expectSize>(); // S
    auto uShape4 = uLayout.template GetShapeDim<4, expectSize>(); // D
    auto uStride1 = uLayout.template GetStrideDim<1, expectSize>();
    auto uStride2 = uLayout.template GetStrideDim<2, expectSize>();
    auto uStride4 = uLayout.template GetStrideDim<4, expectSize>(); // should be 1

    // src1: indices [1, 1, 1, B, S]
    const auto iLayout = src1.GetLayout();
    auto iShape3 = iLayout.template GetShapeDim<3, expectSize>(); // B
    auto iShape4 = iLayout.template GetShapeDim<4, expectSize>(); // S
    auto iStride3 = iLayout.template GetStrideDim<3, expectSize>(); // stride of B
    auto iStride4 = iLayout.template GetStrideDim<4, expectSize>(); // should be 1

    // dst: [1, N, K, 1, D]
    const auto dLayout = dst.GetLayout();
    auto dShape1 = dLayout.template GetShapeDim<1, expectSize>(); // N
    auto dShape2 = dLayout.template GetShapeDim<2, expectSize>(); // K  32/128
    auto dShape4 = dLayout.template GetShapeDim<4, expectSize>(); // D
    auto dStride1 = dLayout.template GetStrideDim<1, expectSize>();
    auto dStride4 = dLayout.template GetStrideDim<4, expectSize>(); // should be 1

    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using IdxDtype = typename T2::Type;

    // Raw tile-aligned shapes from src
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>(); // dim3 = 1
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>(); // D_32aligned
    constexpr auto src0rawShape4 = dstTileW;
    constexpr auto src0rawShape2 = TileOp::GetTensorTileShapeDim<T1, 2, 5>(); // S_32aligned

    if (uShape1 == 0 || uShape2 == 0 || uShape4 == 0 || iShape3 == 0 || iShape4 == 0) {
        return;
    }

    if constexpr (cacheMode == 2) {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);

        __ubuf__ SrcDtype* srcBase = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
        __ubuf__ IdxDtype* idxBase = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
        __gm__ DstDtype* dstBase   = reinterpret_cast<__gm__ DstDtype*>(dst.GetAddr());

        unsigned B = iShape3;
        unsigned S = iShape4;
        unsigned D = uShape4;

        constexpr unsigned S_32aligned = src0rawShape2;
        constexpr unsigned D_32aligned = src0rawShape4;

        auto dStride2 = dLayout.template GetStrideDim<2, expectSize>(); 

        for (unsigned b = 0; b < B; ++b) {
            for (unsigned s = 0; s < S; ++s) {
                IdxDtype idx_val_raw = idxBase[b * iStride3 + s * iStride4];
                unsigned idx_val = static_cast<unsigned>(idx_val_raw);
                if (idx_val >= dShape1 * dShape2) continue;

                uint64_t srcOffset = b * (S_32aligned * D_32aligned) + s * D_32aligned;

                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, dstTileH, dstTileW,
                                               pto::BLayout::RowMajor, -1, -1>;
                SrcTileDefine srcTile(1, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcBase + srcOffset));

                __gm__ DstDtype* scatterAddr = dstBase + idx_val * dStride2;

                using DstGlobalType = pto::GlobalTensor<
                    DstDtype,
                    pto::Shape<1, 1, 1, 1, -1>,
                    pto::Stride<0, 0, 0, 0, 1>
                >;
                DstGlobalType dstGlobal(
                    scatterAddr,
                    pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(D)),
                    pto::Stride<0, 0, 0, 0, 1>(0, 0, 0, 0, 1)
                );
                pto::TSTORE(dstGlobal, srcTile);
            }
            idxBase += (S_32aligned - S);
            srcBase += (S_32aligned - S) * D_32aligned;
        }
        return;
    }

    unsigned B = uShape1;   // = iShape3
    unsigned S = uShape2;   // = iShape4
    unsigned D = uShape4;   // logical D

    constexpr unsigned S_32aligned = src0rawShape2;
    constexpr unsigned D_32aligned = src0rawShape4;

    __ubuf__ SrcDtype* src0_base = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
    __ubuf__ IdxDtype* src1_base = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
    __gm__ DstDtype* dst_base   = reinterpret_cast<__gm__ DstDtype*>(dst.GetAddr());

    for (unsigned b = 0; b < B; ++b) {
        __ubuf__ SrcDtype* cur_src0 = src0_base + b * (S_32aligned * D_32aligned);
        __ubuf__ IdxDtype* cur_src1 = src1_base + b * S_32aligned;

        for (unsigned s = 0; s < S; ++s) {
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);

            IdxDtype curValue = cur_src1[s];

            if constexpr (cacheMode == 1) {
                // PA_NZ mode: index = blockCount * blockSize + offset_in_block
                auto blockCount = static_cast<unsigned>(curValue) / blockSize;
                auto index_in_block = static_cast<unsigned>(curValue) % blockSize;
                unsigned global_row = blockCount * blockSize + index_in_block;
                __gm__ DstDtype* new_dst = dst_base + global_row * dStride1;

                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, dstTileH, dstTileW,
                                               pto::BLayout::RowMajor, -1, -1>;
                SrcTileDefine srcTile(1, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(cur_src0 + s * D_32aligned));

                using ScatterShape  = pto::Shape<1, 1, 1, 1, -1>;
                using ScatterStride = pto::Stride<0, 0, 0, 0, -1>;

                auto dstGlobal = pto::GlobalTensor<DstDtype, ScatterShape, ScatterStride>(
                    new_dst,
                    pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(D)),
                    pto::Stride<0, 0, 0, 0, -1>(0, 0, 0, 0, 1)
                );
                pto::TSTORE(dstGlobal, srcTile);
            } else {
                // Normal scatter: curValue is logical row id
                unsigned row_id = static_cast<unsigned>(curValue);
                __gm__ DstDtype* new_dst = dst_base + row_id * dStride1;

                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, dstTileH, dstTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                SrcTileDefine srcTile(1, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(cur_src0 + s * D_32aligned));

                using ScatterShape  = pto::Shape<1, 1, 1, 1, -1>;
                using ScatterStride = pto::Stride<0, 0, 0, 0, -1>;

                auto dstGlobal = pto::GlobalTensor<DstDtype, ScatterShape, ScatterStride>(
                    new_dst,
                    pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(D)),
                    pto::Stride<0, 0, 0, 0, -1>(0, 0, 0, 0, 1)
                );
                pto::TSTORE(dstGlobal, srcTile);
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H