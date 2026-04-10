/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scene_specific_impl.h
 * \brief 特殊场景实现（如TGatherInL1等非通用功能）
 */

#ifndef TILEOP_TILE_OPERATOR_SCENE_SPECIFIC_IMPL__H
#define TILEOP_TILE_OPERATOR_SCENE_SPECIFIC_IMPL__H

#include "common_impl.h"

template <
    int64_t blockSize, typename globalData, typename tileData, typename DstT, typename SrcT, typename BlockT,
    typename OffsetT, bool isStatic>
INLINE void GatherExecute(
    DstT dst, SrcT src, BlockT block, OffsetT offset, uint64_t offsetsStartOffset, uint64_t srcColumnStartOffset,
    uint64_t GMBlockTableOffset, int64_t srcShape0, int64_t srcShape1, int64_t srcStride0, int64_t srcStride1,
    int64_t dstShape0, int64_t dstShape1, int64_t srcCol, int64_t loop, int64_t c0Size)
{
    for (int64_t i = 0; i < loop; i++) {
        uint64_t gatherOffset = offset.GetAddr()[i + offsetsStartOffset];
        gatherOffset = CalaOffset2PageAttention<uint64_t, typename BlockT::Type, blockSize>(
            block.GetAddr() + GMBlockTableOffset, gatherOffset);
        globalData src0Global(
            (__gm__ typename SrcT::Type*)(src.GetAddr() + gatherOffset * srcCol + srcColumnStartOffset),
            pto::Shape<1, 1, 1, -1, -1>(srcShape0, srcShape1), pto::Stride<1, 1, 1, -1, -1>(srcStride0, srcStride1));

        tileData dstL1;
        if constexpr (!isStatic) {
            dstL1 = tileData(dstShape0, dstShape1);
        }

        pto::TASSIGN(dstL1, (uint64_t)((__cbuf__ typename DstT::Type*)dst.GetAddr() + i * c0Size));
        pto::TLOAD(dstL1, src0Global);
    }
}

template <int64_t blockSize, typename DstT, typename SrcT, typename BlockT, typename OffsetT, typename globalData>
INLINE void TGatherInL1Static(
    DstT dst, SrcT src, BlockT block, OffsetT offset, int64_t srcShape1, int64_t srcStride0, int64_t srcStride1,
    uint64_t srcColumnStartOffset, uint64_t offsetsStartOffset, uint64_t GMBlockTableOffset)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename DstT::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename DstT::Type);
    constexpr int64_t staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstT::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename DstT::TileShape>::type::value;
    constexpr int64_t staticDstShape0 = TileOp::GetTensorShapeDim<DstT, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticDstShape1 = TileOp::GetTensorShapeDim<DstT, shapeSize - 1>();

    if constexpr (staticDstShape1 % c0Size > 0) {
        using tileData = pto::Tile<
            pto::TileType::Mat, typename DstT::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, staticDstShape0,
            staticDstShape1, pto::SLayout::RowMajor>;
        GatherExecute<blockSize, globalData, tileData, DstT, SrcT, BlockT, OffsetT, true>(
            dst, src, block, offset, offsetsStartOffset, srcColumnStartOffset, GMBlockTableOffset, 1, staticL1W,
            srcStride0, srcStride1, staticDstShape0, staticDstShape1, srcShape1, staticDstShape0, c0Size);
    } else {
        using tileData = pto::Tile<
            pto::TileType::Mat, typename DstT::Type, staticL1W / c0Size, staticL1H * c0Size, pto::BLayout::RowMajor,
            staticDstShape1 / c0Size, c0Size>;
        GatherExecute<blockSize, globalData, tileData, DstT, SrcT, BlockT, OffsetT, true>(
            dst, src, block, offset, offsetsStartOffset, srcColumnStartOffset, GMBlockTableOffset,
            staticDstShape1 / c0Size, c0Size, c0Size, 1, staticDstShape1 / c0Size, c0Size, srcShape1, staticDstShape0,
            c0Size);
    }
}

template <int64_t blockSize, typename DstT, typename SrcT, typename BlockT, typename OffsetT, typename globalData>
INLINE void TGatherInL1Dynamic(
    DstT dst, SrcT src, BlockT block, OffsetT offset, int64_t srcShape1, int64_t srcStride0, int64_t srcStride1,
    uint64_t srcColumnStartOffset, uint64_t offsetsStartOffset, uint64_t GMBlockTableOffset)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename DstT::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename DstT::Type);
    constexpr int64_t staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename DstT::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename DstT::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);

    if (dstShape1 % c0Size > 0) {
        using tileData = pto::Tile<
            pto::TileType::Mat, typename DstT::Type, staticL1H, staticL1W, pto::BLayout::ColMajor, -1, -1,
            pto::SLayout::RowMajor>;
        GatherExecute<blockSize, globalData, tileData, DstT, SrcT, BlockT, OffsetT, false>(
            dst, src, block, offset, offsetsStartOffset, srcColumnStartOffset, GMBlockTableOffset, 1, staticL1W,
            srcStride0, srcStride1, dstShape0, dstShape1, srcShape1, dstShape0, c0Size);
    } else {
        using tileData = pto::Tile<
            pto::TileType::Mat, typename DstT::Type, staticL1W / c0Size, staticL1H * c0Size, pto::BLayout::RowMajor, -1,
            -1>;
        // 这里需要采用性能更高的ND2ND的搬运方式。
        // GM上将(1, dstShape1)的数据，转变为(dstShape1 / c0Size, c0Size)。
        // L1上将(staticL1W / c0Size, staticL1H / 16, 16, c0Size)的NZ数据，对每c0Size列做展平为一行，得到(staticL1W /
        // c0Size, staticL1H * c0Size)的ND格式。 那么L1上有效数据就是(dstShape1 / c0Size, c0Size)。
        GatherExecute<blockSize, globalData, tileData, DstT, SrcT, BlockT, OffsetT, false>(
            dst, src, block, offset, offsetsStartOffset, srcColumnStartOffset, GMBlockTableOffset, dstShape1 / c0Size,
            c0Size, c0Size, 1, dstShape1 / c0Size, c0Size, srcShape1, dstShape0, c0Size);
    }
}

#endif // TILEOP_TILE_OPERATOR_SCENE_SPECIFIC_IMPL__H