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

template <unsigned cacheMode, unsigned blockSize, typename T0, typename T1, typename T2,typename C>
TILEOP void TIndexOutcast(T0 dst, T1 src, T2 src1,C coordinate)
{
    constexpr auto expectSize = 5;
    const auto uLayout = src.GetLayout();
    auto uShape1 = uLayout.template GetShapeDim<1, expectSize>();
    auto uShape2 = uLayout.template GetShapeDim<2, expectSize>();
    auto uShape4 = uLayout.template GetShapeDim<4, expectSize>();

    const auto iLayout = src1.GetLayout();
    auto iShape3 = iLayout.template GetShapeDim<3, expectSize>();
    auto iShape4 = iLayout.template GetShapeDim<4, expectSize>();

    const auto dLayout = dst.GetLayout();
    auto GmShape0 = dLayout.template GetShapeDim<1, expectSize>();
    auto GmShape1 = dLayout.template GetShapeDim<2, expectSize>();
    auto GmShape2 = dLayout.template GetShapeDim<3, expectSize>();
    auto GmShape3 = dLayout.template GetShapeDim<4, expectSize>();
    
    auto Offset = dLayout.template GetGmOffset<C, expectSize>(coordinate);
    
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

        unsigned b = iShape3;
        unsigned s = iShape4;
        unsigned dim = uShape4;

        for (unsigned i = 0; i < b; ++i) {
            for (unsigned j = 0; j < s; ++j) {
                unsigned targetRow = static_cast<unsigned>(*dstIdx);
                __gm__ DstDtype* curDst = dstBase + targetRow * dim;

                using SrcTileDefine = pto::Tile<
                    pto::TileType::Vec,
                    SrcDtype,
                    dstTileH,
                    dstTileW,
                    pto::BLayout::RowMajor,
                    -1, -1
                >;
                SrcTileDefine srcTile(dstTileH, dim);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(curSrc));

                using DstGlobal = pto::GlobalTensor<
                    DstDtype,
                    pto::Shape<1, 1, 1, 1, -1>,
                    pto::Stride<0, 0, 0, 0, 1>
                >;
                DstGlobal dstGlobal(
                    curDst,
                    pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(dim)),
                    pto::Stride<0, 0, 0, 0, 1>(0, 0, 0, 0, 1)
                );
                pto::TSTORE(dstGlobal, srcTile);

                curSrc += nd_32aligned;
                dstIdx++;
            }
            curSrc += (src0rawShape1 - s) * nd_32aligned;
            dstIdx += (s1_32aligned - s);
        }
        return;
    }

    auto alignTS2TS3 = dstTileH * dstTileW;
    auto alignSrc1 = s1_32aligned;
    __ubuf__ SrcDtype* src0_base = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
    __ubuf__ IdxDtype* src1_base = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
    __gm__ DstDtype* dst_base = reinterpret_cast<__gm__ DstDtype*>(dst.GetAddr());
    dst_base += Offset;

    for (int i = 0; i < uShape1; ++i) {
        for (int j = 0; j < uShape2; ++j) {
            for (auto k = 0; k < iShape4; ++k) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);

                auto curValue = *(reinterpret_cast<__ubuf__ IdxDtype*>(src1_base + k));

                __ubuf__ SrcDtype* src_ptr = src0_base + k * nd_32aligned;
                
                if constexpr (cacheMode == 1) {
                    auto blockCount = curValue / blockSize;
                    auto index = curValue % blockSize;
                    __gm__ DstDtype* new_dst = dst_base + blockCount * blockSize * GmShape3 + index * 32 / sizeof(DstDtype);
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

                    using SrcTileDefine = pto::Tile<
                        pto::TileType::Vec,
                        SrcDtype,
                        dstTileH,
                        dstTileW,
                        pto::BLayout::RowMajor,
                        -1, -1
                    >;
                    SrcTileDefine srcTile(dstTileH, dstTileW);
                    pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(src_ptr));

                    using DstGlobalType = pto::GlobalTensor<
                        DstDtype,
                        pto::Shape<1, 1, 1, 1, -1>,
                        pto::Stride<0, 0, 0, 0, 1>
                    >;
                    DstGlobalType dstGlobal(
                        new_dst,
                        pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(dstTileW)),
                        pto::Stride<0, 0, 0, 0, 1>(0, 0, 0, 0, 1)
                    );
                    pto::TSTORE(dstGlobal, srcTile);

                } else {
                    __gm__ DstDtype* new_dst = dst_base + static_cast<unsigned>(curValue) * GmShape3;
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

                    using SrcTileDefine = pto::Tile<
                        pto::TileType::Vec,
                        SrcDtype,
                        dstTileH,
                        dstTileW,
                        pto::BLayout::RowMajor,
                        -1, -1
                    >;
                    SrcTileDefine srcTile(dstTileH, dstTileW);
                    pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(src_ptr));

                    using DstGlobalType = pto::GlobalTensor<
                        DstDtype,
                        pto::Shape<1, 1, 1, 1, -1>,
                        pto::Stride<0, 0, 0, 0, 1>
                    >;
                    DstGlobalType dstGlobal(
                        new_dst,
                        pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(dstTileW)),
                        pto::Stride<0, 0, 0, 0, 1>(0, 0, 0, 0, 1)
                    );
                    pto::TSTORE(dstGlobal, srcTile);
                }
            }
            src0_base += alignTS2TS3;
            src1_base += alignSrc1;
            dst_base += GmShape2 * GmShape3;
        }
    }
}
#endif // TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H