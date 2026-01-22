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
 	  * \file index_outcast.h
 	  * \brief
*/
#ifndef TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H
#define TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <unsigned cacheMode, unsigned blockSize, typename T0, typename T1, typename T2>
TILEOP void TIndexOutcast(T0 dst, T1 src, T2 src1)
{
    constexpr auto expectSize = 5;

    const auto uLayout = src.GetLayout();
    auto uShape1 = uLayout.template GetShapeDim<1, expectSize>();
    auto uShape2 = uLayout.template GetShapeDim<2, expectSize>();
    auto uShape4 = uLayout.template GetShapeDim<4, expectSize>();

    const auto iLayout = src1.GetLayout();
    auto iShape3 = iLayout.template GetShapeDim<3, expectSize>();
    auto iShape4 = iLayout.template GetShapeDim<4, expectSize>();

    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using IdxDtype = typename T2::Type;

    constexpr auto src0rawShape1 = TileOp::GetTensorTileShapeDim<T1, 2, 5>();
    constexpr auto dstTileH      = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto dstTileW      = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto s1_32aligned  = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    constexpr auto nd_32aligned  = dstTileW;

    if (uShape1 == 0 || uShape2 == 0 || uShape4 == 0 || iShape3 == 0 || iShape4 == 0) {
        return;
    }

    if constexpr (cacheMode == 2) {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);

        __ubuf__ SrcDtype* srcBase = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
        __ubuf__ IdxDtype* idxBase = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
        __gm__   DstDtype* dstBase = reinterpret_cast<__gm__   DstDtype*>(dst.GetAddr());

        __ubuf__ SrcDtype* curSrc = srcBase;
        __ubuf__ IdxDtype* dstIdx = idxBase;

        unsigned B = iShape3;
        unsigned S = iShape4;
        unsigned D = uShape4;

        for (unsigned b = 0; b < B; ++b) {
            for (unsigned s = 0; s < S; ++s) {
                unsigned targetRow = static_cast<unsigned>(*dstIdx);
                __gm__ DstDtype* curDst = dstBase + targetRow * D;

                using SrcTileDefine = pto::Tile<
                    pto::TileType::Vec,
                    SrcDtype,
                    dstTileH,
                    dstTileW,
                    pto::BLayout::RowMajor,
                    -1, -1
                >;
                SrcTileDefine srcTile(dstTileH, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(curSrc));

                using DstGlobal = pto::GlobalTensor<
                    DstDtype,
                    pto::Shape<1, 1, 1, 1, -1>,
                    pto::Stride<0, 0, 0, 0, 1>
                >;
                DstGlobal dstGlobal(
                    curDst,
                    pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(D)),
                    pto::Stride<0, 0, 0, 0, 1>(0, 0, 0, 0, 1)
                );
                pto::TSTORE(dstGlobal, srcTile);

                curSrc += nd_32aligned;
                dstIdx++;
            }
            curSrc += (src0rawShape1 - S) * nd_32aligned;
            dstIdx += (s1_32aligned - S);
        }
        return;
    }

    unsigned B = iShape3;
    unsigned S = iShape4;
    unsigned D = uShape4;

    constexpr unsigned S_32aligned = s1_32aligned;
    constexpr unsigned D_32aligned = nd_32aligned;

    __ubuf__ SrcDtype* src0_base = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
    __ubuf__ IdxDtype* src1_base = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
    __gm__   DstDtype* dst_base  = reinterpret_cast<__gm__   DstDtype*>(dst.GetAddr());

    for (unsigned b = 0; b < B; ++b) {
        __ubuf__ SrcDtype* cur_src0 = src0_base + b * (S_32aligned * D_32aligned);
        __ubuf__ IdxDtype* cur_src1 = src1_base + b * S_32aligned;

        for (unsigned s = 0; s < S; ++s) {
            IdxDtype curValue = cur_src1[s];
            __gm__ DstDtype* curDst = nullptr;

            if constexpr (cacheMode == 1) {
                auto blockCount = static_cast<unsigned>(curValue) / blockSize;
                auto index_in_block = static_cast<unsigned>(curValue) % blockSize;
                unsigned byte_offset = blockCount * blockSize * D + index_in_block * 32 / sizeof(DstDtype);
                curDst = dst_base + byte_offset;
            } else {
                unsigned row_id = static_cast<unsigned>(curValue);
                curDst = dst_base + row_id * D;
            }

            using SrcTileDefine = pto::Tile<
                pto::TileType::Vec,
                SrcDtype,
                dstTileH,
                dstTileW,
                pto::BLayout::RowMajor,
                -1, -1
            >;
            SrcTileDefine srcTile(1, D);
            pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(cur_src0 + s * D_32aligned));

            using DstGlobalType = pto::GlobalTensor<
                DstDtype,
                pto::Shape<1, 1, 1, 1, -1>,
                pto::Stride<0, 0, 0, 0, 1>
            >;
            DstGlobalType dstGlobal(
                curDst,
                pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(D)),
                pto::Stride<0, 0, 0, 0, 1>(0, 0, 0, 0, 1)
            );
            pto::TSTORE(dstGlobal, srcTile);
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H