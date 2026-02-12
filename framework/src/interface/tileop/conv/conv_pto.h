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
 * \file conv_pto.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_CONV_PTO__H
#define TILEOP_TILE_OPERATOR_CONV_PTO__H
#include <limits.h>
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace TileOp {

constexpr int16_t SHAPE_DIM4 = 4;
constexpr int16_t SHAPE_DIM5 = 5;
constexpr int16_t CONV_IDX_0 = 0;
constexpr int16_t CONV_IDX_1 = 1;
constexpr int16_t CONV_IDX_2 = 2;
constexpr int16_t CONV_IDX_3 = 3;
constexpr int16_t CONV_IDX_4 = 4;

struct ShapeInfo {
    int64_t shape0 = 0;
    int64_t shape1 = 0;
    int64_t shape2 = 0;
    int64_t shape3 = 0;
    int64_t shape4 = 0;
};

struct OffsetInfo {
    int64_t offset0 = 0;
    int64_t offset1 = 0;
    int64_t offset2 = 0;
    int64_t offset3 = 0;
    int64_t offset4 = 0;
};

template <int16_t idx, typename U>
INLINE int64_t GetConvShape(const U &tileTensor) {
    static_assert(idx < SHAPE_DIM5, "Idx should be less than 5");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetShapeDim<idx>();
}

template <int16_t idx, typename U>
INLINE int64_t GetConvStride(const U &tileTensor) {
    static_assert(idx < SHAPE_DIM4, "Idx should be less than 4");
    const auto tileLayout = tileTensor.GetLayout();
    return tileLayout.template GetStrideDim<idx>();
}

/**
 * Calculate load GM offset for input/weight with NCHW format.
 * shapeInfo: input -> [orgCi, orgHi, orgWi], weight -> [kh, kw]
 * shapeInfo: input -> [  0  ,   1  ,   2  ], weight -> [0 , 1 ]
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_d_offset, weight -> src_d_offset
 * offset3: input -> src_h_offset, weight -> 0
 * offset4: input -> src_w_offset, weight -> 0
 * isInput: true -> input, false -> weight
 */
INLINE int64_t CalLoadOffsetNCHW(const ShapeInfo &shapeInfo, const OffsetInfo &offsetInfo, const bool &isInput) {
    if (isInput) {
        int64_t inputOneBatchSize = shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2;
        int64_t cinOffset = offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2;
        int64_t offsetH = offsetInfo.offset3 < 0 ? 0 : offsetInfo.offset3;
        offsetH = offsetInfo.offset3 > shapeInfo.shape1 ? shapeInfo.shape1 : offsetInfo.offset3;
        int64_t offsetW = offsetInfo.offset4 < 0 ? 0 : offsetInfo.offset4;
        offsetW = offsetInfo.offset4 > shapeInfo.shape2 ? shapeInfo.shape2 : offsetInfo.offset4;
        return offsetInfo.offset0 * inputOneBatchSize + cinOffset + offsetH * shapeInfo.shape2 + offsetW;
    } else {
        return offsetInfo.offset0 + offsetInfo.offset1 * shapeInfo.shape0 * shapeInfo.shape1;
    }
}

/**
 * Calculate store GM offset with NZ -> NC1HWC0 format.
 * shapeInfo: [cout, hout, wout]
 * shapeInfo: [  0 ,  1  ,  2  ]
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_d_offset
 * offset3: dst_h_offset
 * offset4: dst_w_offset
 */
INLINE int64_t CalStoreOffsetNCHW(const ShapeInfo &shapeInfo, const OffsetInfo &offsetInfo) {
    int64_t outputOneBatchSize = shapeInfo.shape0 * shapeInfo.shape1 * shapeInfo.shape2;
    int64_t coutOffset = offsetInfo.offset1 * shapeInfo.shape1 * shapeInfo.shape2;
    return offsetInfo.offset0 * outputOneBatchSize + coutOffset + offsetInfo.offset3 * shapeInfo.shape2 +
           offsetInfo.offset4;
}

/**
 * Copy input data from DDR to L1 with DN2NZ, input NCHW -> NC1HWC0, weight NCHW -> FZ.
 * dst: input -> AL1(NC1HWC0), weight -> BL1(FZ)
 * src: input -> GM(NCHW), weigh -> GM(NCHW)
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_d_offset, weight -> src_d_offset
 * offset3: input -> src_h_offset, weight -> 0
 * offset4: input -> src_w_offset, weight -> 0
 * isInput: true -> input, false -> weight
 */
template <typename T, typename U>
INLINE void TLoadConvDN2NZ(T &dst, U &src, const OffsetInfo &offsetInfo, const bool &isInput) {
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t srcN = GetConvShape<CONV_IDX_0>(src);
    int64_t srcC = GetConvShape<CONV_IDX_1>(src);
    int64_t srcH = GetConvShape<CONV_IDX_2>(src);
    int64_t srcW = GetConvShape<CONV_IDX_3>(src);
    int64_t dstShape0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
    int64_t srcStrideN = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStrideC = GetConvStride<CONV_IDX_1>(src);
    int64_t srcStrideH = GetConvStride<CONV_IDX_2>(src);
    int64_t srcStrideW = GetConvStride<CONV_IDX_3>(src);
    constexpr auto stcDstShape0 = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto stcDstShape1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto stcDstShape2 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto stcDstShape3 = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    ShapeInfo shapeInfo;
    if (isInput) {
        shapeInfo = {srcC, srcH, srcW};
    } else {
        shapeInfo = {srcH, srcW};
    }
    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    int64_t gmOffset = CalLoadOffsetNCHW(shapeInfo, offsetInfo, isInput);
    globalData srcGlobal((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        shapeDim4(srcN, srcC, srcH, srcW),
        strideDim4(srcStrideN, srcStrideC, srcStrideH, srcStrideW));
    int64_t tileShape0 = isInput ? dstShape0 : 1;
    int64_t tileShape1 = isInput ? dstShape1 : dstShape0;
    int64_t tileShape2 = isInput ? dstShape2 : dstShape1;
    int64_t tileShape3 = isInput ? dstShape3 : dstShape2;
    if (isInput) {
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * stcDstShape3 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, T, bufferSize, pto::Layout::NC1HWC0,
            pto::ConvTileShape<-1, -1, -1, -1, c0Size>>;
        tileData dstL1(tileShape0, tileShape1, tileShape2, tileShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    } else {
        constexpr auto bufferSize = stcDstShape0 * stcDstShape1 * stcDstShape2 * BLOCK_ALIGN_BYTE;
        using tileData = pto::ConvTile<pto::TileType::Mat, T, bufferSize, pto::Layout::FRACTAL_Z,
            pto::ConvTileShape<-1, -1, -1, -1, c0Size>>;
        tileData dstL1(tileShape0, tileShape1, tileShape2, tileShape3);
        pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
        pto::TLOAD(dstL1, srcGlobal);
    }
    return;
}

// Copy data from DDR to L1
template <CopyInMode mode, typename T, typename U>
TILEOP void TLoadConv(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3, const int64_t &offset4, const bool &isInput) {
    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoadConv Error]: Src format shoulde be GM and Dst format shoulde be L1");
    OffsetInfo offsetInfo = {offset0, offset1, offset2, offset3, offset4};
    if constexpr (mode == CopyInMode::ND2NZ) {
    } else if constexpr (mode == CopyInMode::DN2NZ) {
        TLoadConvDN2NZ(dst, src, offsetInfo, isInput);
    } else if constexpr (mode == CopyInMode::NZ2NZ) {
    }
    return;
}

/**
 * Copy data from L0C to DDR with NZ -> NCHW format.
 * dst: GM(NCHW)
 * src: l0c(NZ)
 * offset0: dst_n_offset
 * offset1: dst_c_offset
 * offset2: dst_d_offset
 * offset3: dst_h_offset
 * offset4: dst_w_offset
 */
template <typename T, typename U>
INLINE void TStoreConvNZ2DN(T &dst, U &src, const OffsetInfo &offsetInfo, const int64_t &realM, const int64_t &realN) {
    constexpr auto shapeSize = Std::tuple_size<typename U::Shape>::value;
    constexpr auto srcM = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr auto srcN = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_3>(dst);

    ShapeInfo shapeInfo{dstC, dstH, dstW};
    int64_t gmOffset = CalStoreOffsetNCHW(shapeInfo, offsetInfo);
    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    globalData dstGlobal((__gm__ typename T::Type *)(dst.GetAddr() + gmOffset),
        shapeDim4(dstN, dstC, dstH, dstW),
        strideDim4(dstStrideN, dstStrideC, dstStrideH, dstStrideW));
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcN, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Normal>;
    tileData srcL0C(realM, realN);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TSTORE(dstGlobal, srcL0C);
    return;
}

// Copy data from L0C to DDR
template <CopyOutMode mode, typename T, typename U>
TILEOP void TStoreConv(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3, const int64_t &offset4, const int64_t &realM, const int64_t &realN) {
    constexpr auto srcShapeSize = Std::tuple_size<typename U::Shape>::value;
    static_assert(srcShapeSize == SHAPE_DIM2, "L0C shape size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::GM && U::FORMAT == Hardware::L0C,
        "[TStoreConv Error]: Src format shoulde be L0C and Dst format shoulde be GM");
    OffsetInfo offsetInfo = {offset0, offset1, offset2, offset3, offset4};
    if constexpr (mode == CopyOutMode::NZ2ND) {
    } else if constexpr (mode == CopyOutMode::NZ2DN) {
        TStoreConvNZ2DN(dst, src, offsetInfo, realM, realN);
    } else if constexpr (mode == CopyOutMode::NZ2NZ) {
        // TStoreConvNZ2NZ(dst, src, offset0, offset1, offset2, offset3, offset4, nBL1, woL1, nL0, woL0);
    }
    return;
}

template <typename T, typename U>
TILEOP void TLoad3D(T &dst, U &src, const int64_t &mPos, const int64_t &kPos, 
                    const int64_t &padLeft, const int64_t &padRight, const int64_t &padTop, const int64_t &padBottom, const int64_t &padValue, 
                    const int64_t &filterH, const int64_t &filterW, const int64_t &dilationH, const int64_t &dilationW, 
                    const int64_t &strideH, const int64_t &strideW) {
    constexpr auto staticN = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto staticC1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto staticH = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto staticW = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    constexpr auto staticC0 = Std::tuple_element<CONV_IDX_4, typename U::TileShape>::type::value;
    constexpr auto bufferSize = staticN * staticC1 * staticH * staticW * staticC0;
    // 待pto-isa的ConvTileShape数据类型改为int64之后，需要移除类型转换
    int n = static_cast<int>(GetConvShape<CONV_IDX_0>(src));
    int c1 = static_cast<int>(GetConvShape<CONV_IDX_1>(src));
    int h = static_cast<int>(GetConvShape<CONV_IDX_2>(src));
    int w = static_cast<int>(GetConvShape<CONV_IDX_3>(src));
    int c0 = static_cast<int>(GetConvShape<CONV_IDX_4>(src));
    using srcTensor = pto::ConvTile<pto::TileType::Mat, typename U::Type, bufferSize, pto::Layout::NC1HWC0, pto::ConvTileShape<-1, -1, -1, -1, -1>>;
    srcTensor l1(n, c1, h, w, c0);
    l1.SetFmapH(static_cast<uint16_t>(h));
    l1.SetFmapW(static_cast<uint16_t>(w));
    uint8_t values[4] = {static_cast<uint8_t>(padLeft), static_cast<uint8_t>(padRight), static_cast<uint8_t>(padTop), static_cast<uint8_t>(padBottom)};
    l1.SetPadListArray(values);
    l1.SetFilterH(filterH);
    l1.SetFilterW(filterW);
    l1.SetDilationH(dilationH);
    l1.SetDilationW(dilationW);
    l1.SetStrideH(strideH);
    l1.SetStrideW(strideW);
    l1.SetPadValue(padValue);
    l1.SetChannelSize(c1 * c0);

    constexpr auto staticML0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto staticKL0 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    int64_t mL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t kL0 = GetConvShape<CONV_IDX_1>(dst);
    using dstTensor = pto::TileLeft<typename T::Type, staticML0, staticKL0, -1, -1>;
    dstTensor l0(mL0, kL0);

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TSETFMATRIX(l1);
    pto::TIMG2COL(l0, l1, mPos, kPos);
}

template <typename T, typename U>
TILEOP void TLoad2D(T &dst, U &src, const int64_t &indexRow, const int64_t &indexCol) {
    constexpr auto staticC1HW = Std::tuple_element<CONV_IDX_0, typename U::TileShape>::type::value;
    constexpr auto staticN1 = Std::tuple_element<CONV_IDX_1, typename U::TileShape>::type::value;
    constexpr auto staticN0 = Std::tuple_element<CONV_IDX_2, typename U::TileShape>::type::value;
    constexpr auto staticC0 = Std::tuple_element<CONV_IDX_3, typename U::TileShape>::type::value;
    constexpr auto bufferSize = staticC1HW * staticN1 * staticN0 * staticC0;
    // 待pto-isa的ConvTileShape数据类型改为int64之后，需要移除类型转换
    int c1hw = static_cast<int>(GetConvShape<CONV_IDX_0>(src));
    int n1 = static_cast<int>(GetConvShape<CONV_IDX_1>(src));
    int n0 = static_cast<int>(GetConvShape<CONV_IDX_2>(src));
    int c0 = static_cast<int>(GetConvShape<CONV_IDX_3>(src));
    using srcTensor = pto::ConvTile<pto::TileType::Mat, typename U::Type, bufferSize, pto::Layout::FRACTAL_Z, pto::ConvTileShape<-1, -1, -1, -1>>;
    srcTensor l1(c1hw, n1, n0, c0);

    constexpr auto staticKL0 = Std::tuple_element<CONV_IDX_0, typename T::TileShape>::type::value;
    constexpr auto staticNL0 = Std::tuple_element<CONV_IDX_1, typename T::TileShape>::type::value;
    int64_t kL0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t nL0 = GetConvShape<CONV_IDX_1>(dst);
    using dstTensor = pto::TileRight<typename T::Type, staticKL0, staticNL0, -1, -1>;
    dstTensor l0(kL0, nL0);

    pto::TASSIGN(l1, static_cast<uint64_t>(src.GetAddr()));
    pto::TASSIGN(l0, static_cast<uint64_t>(dst.GetAddr()));
    pto::TEXTRACT<dstTensor, srcTensor>(l0, l1, indexRow, indexCol);
}
} // namespace TileOp
#endif // TILEOP_TILE_OPERATOR_CONV_PTO__H