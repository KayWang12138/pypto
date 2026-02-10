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
#include "utils/layout.h"
#include "utils/tile_tensor.h"

namespace TileOp {

constexpr int16_t SHAPE_DIM2 = 2;
constexpr int16_t SHAPE_DIM4 = 4;
constexpr int16_t SHAPE_DIM5 = 5;
constexpr int16_t CONV_IDX_0 = 0;
constexpr int16_t CONV_IDX_1 = 1;
constexpr int16_t CONV_IDX_2 = 2;
constexpr int16_t CONV_IDX_3 = 3;
constexpr int16_t CONV_IDX_4 = 4;

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
 * shapeInfo: input -> [orgHi, orgWi, orgCi, khxkw, kAL1], weight -> [orgCi, khxkw, kBL1]
 * shapeInfo: input -> [  0  ,   1  ,   2  ,   3  ,  4  ], weight -> [  0  ,   1  ,  2  ]
 * offset0: input -> batchOffset, weight -> nBL1Iter
 * offset1: input -> kAL1Iter,  weight -> kBL1Iter
 * offset2: input -> hiIdx, weight -> nBL1
 * offset3: input -> wiIdx, weight -> groups
 * inputFlag: true -> input, false -> weight
 */
INLINE int64_t CalLoadOffsetNHWC(const std::vector<int64_t> &srcShape, const int64_t &offset0, const int64_t &offset1,
    const int64_t &offset2, const int64_t &offset3, const bool &inputFlag) {
    if (inputFlag) {
        int64_t inputOneBatchSize = shapeInfo[CONV_IDX_0] * shapeInfo[CONV_IDX_1] * shapeInfo[CONV_IDX_2];
        int64_t cinAInCore = shapeInfo[CONV_IDX_4] / shapeInfo[CONV_IDX_3];
        return offset0 * inputOneBatchSize + offset2 * shapeInfo[CONV_IDX_1] * shapeInfo[CONV_IDX_2] +
               offset3 * shapeInfo[CONV_IDX_2] + offset1 * cinAInCore;
    } else {
        int64_t cinBInCore = shapeInfo[CONV_IDX_2] / shapeInfo[CONV_IDX_1];
        int64_t coutOffset = shapeInfo[CONV_IDX_0] / offset3 * shapeInfo[CONV_IDX_1];
        return offset1 * cinBInCore + offset0 * offset2 * coutOffset;
    }
}

/**
 * Calculate load GM offset for input/weight with NCHW format.
 * shapeInfo: input -> [orgCi, orgHi, orgWi], weight -> [kh, kw]
 * shapeInfo: input -> [  0  ,   1  ,   2  ], weight -> [0 , 1 ]
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_h_offset, weight -> src_d_offset
 * offset3: input -> src_w_offset, weight -> 0
 * offset4: input -> src_d_offset, weight -> 0
 * isInput: 1 -> input, 0 -> weight
 */
INLINE int64_t CalLoadOffsetNCHW(const std::vector<int64_t> &shapeInfo, const int64_t &offset0, const int64_t &offset1,
    const int64_t &offset2, const int64_t &offset3, const int64_t &offset4, const uint8_t &isInput) {
    if (isInput) {
        int64_t inputOneBatchSize = shapeInfo[CONV_IDX_0] * shapeInfo[CONV_IDX_1] * shapeInfo[CONV_IDX_2];
        int64_t cinOffset = offset1 * shapeInfo[CONV_IDX_1] * shapeInfo[CONV_IDX_2];
        offset2 = offset2 < 0 ? 0 : offset2;
        offset2 = offset2 > shapeInfo[CONV_IDX_1] ? shapeInfo[CONV_IDX_1] : offset2;
        offset3 = offset3 < 0 ? 0 : offset3;
        offset3 = offset3 > shapeInfo[CONV_IDX_2] ? shapeInfo[CONV_IDX_2] : offset3;
        return offset0 * inputOneBatchSize + offset1 * cinOffset + offset2 * shapeInfo[CONV_IDX_2] + offset3;
    } else {
        return offset0 + offset1 * shapeInfo[CONV_IDX_0] * shapeInfo[CONV_IDX_1];
    }
}

/**
 * Calculate store GM offset with NZ -> NC1HWC0 format.
 * shapeInfo: [cout, hout, wout]
 * shapeInfo: [  0 ,  1  ,  2  ]
 * offset0: batchOffset
 * offset1: nL1Offset
 * offset2: hL1OutOffset
 * offset3: wL1OutOffset
 * offset4: nL0Offset
 * offset5: hL0Offset
 * offset6: wL0Offset
 */
INLINE int64_t CalStoreOffsetNCHW(const std::vector<int64_t> &shapeInfo, const int64_t &offset0, const int64_t &offset1,
    const int64_t &offset2, const int64_t &offset3, const int64_t &offset4, const int64_t &offset5,
    const int64_t &offset6) {
    int64_t outOneBatchSize = shapeInfo[CONV_IDX_0] * shapeInfo[CONV_IDX_1] * shapeInfo[CONV_IDX_2];
    int64_t hoOffset = offset2 + offset5;
    int64_t woOffset = offset3 + offset6;
    int64_t coOffset = offset1 + offset4;
    return offset0 * outOneBatchSize + coOffset * shapeInfo[CONV_IDX_1] * shapeInfo[CONV_IDX_2] +
           hoOffset * shapeInfo[CONV_IDX_2] + woOffset;
}

/**
 * Calculate store GM offset with NZ -> NC1HWC0 format.
 * shapeInfo: [hout, wout, cout0, nBL1, woL1, nL0, woL0]
 * shapeInfo: [  0 ,  1  ,   2  ,  3  ,  4  ,  5 ,  6  ]
 * offset0: nBL1Iter
 * offset1: hoL1Iter
 * offset2: woL1Iter
 * offset3: nL0Iter
 * offset4: woL0Iter
 */
INLINE int64_t CalStoreOffset5hd(const std::vector<int64_t> &shapeInfo, const int64_t &offset0, const int64_t &offset1,
    const int64_t &offset2, const int64_t &offset3, const int64_t &offset4) {
    int64_t coutOffset = offset0 * shapeInfo[CONV_IDX_3] + offset3 * shapeInfo[CONV_IDX_5];
    int64_t woutOffset = offset2 * shapeInfo[CONV_IDX_4] + offset4 * shapeInfo[CONV_IDX_6];
    return coutOffset * shapeInfo[CONV_IDX_0] * shapeInfo[CONV_IDX_1] +
           offset2 * shapeInfo[CONV_IDX_1] * shapeInfo[CONV_IDX_2] +
           woutOffset * shapeInfo[CONV_IDX_2];
}

/**
 * Copy input data from DDR to L1 with ND2NZ, input NHWC -> NC1HWC0, weight NHWC -> FZ.
 * dst: input -> AL1(NC1HWC0), weight -> BL1(FZ)
 * src: input -> GM(NHWC), weigh -> GM(NHWC)
 * offset0: input -> batchOffset, weight -> nBL1Iter
 * offset1: input -> kAL1Iter,  weight -> kBL1Iter
 * offset2: input -> hiIdx, weight -> nBL1
 * offset3: input -> wiIdx, weight -> groups
 * khxkw: kernelH * kernelW
 * kL1: input -> kAL1, weight -> kBL1
 */
template <typename T, typename U>
INLINE void TLoadConvND2NZ( T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3, const int64_t &khxkw, const int64_t &kL1) {
    constexpr auto dstShapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(dstShapeSize == SHAPE_DIM4 || dstShapeSize == SHAPE_DIM5, "Dst shape size should be 4 or 5 Dim");
    bool inputFlag = dstShapeSize == SHAPE_DIM5;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t srcN = GetConvShape<CONV_IDX_0>(src);
    int64_t srcH = GetConvShape<CONV_IDX_1>(src);
    int64_t srcW = GetConvShape<CONV_IDX_2>(src);
    int64_t srcC = GetConvShape<CONV_IDX_3>(src);
    int64_t dstShape0 = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstShape1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstShape2 = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstShape3 = GetConvShape<CONV_IDX_3>(dst);
    int64_t srcStrideN = GetConvStride<CONV_IDX_0>(src);
    int64_t srcStrideH = GetConvStride<CONV_IDX_1>(src);
    int64_t srcStrideW = GetConvStride<CONV_IDX_2>(src);
    int64_t srcStrideC = GetConvStride<CONV_IDX_3>(src);
    int64_t bufferSize = inputFlag ? dstShape0 * dstShape1 * dstShape2 * dstShape3 * BLOCK_ALIGN_BYTE :
                                     dstShape0 * dstShape1 * dstShape2 * BLOCK_ALIGN_BYTE;
    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim4, strideDim4, pto::Layout::NHWC>;
    std::vector<int64_t> shapeInfo = inputFlag ? {srcH, srcW, srcC, khxkw, kL1} : {srcC, khxkw, kL1};
    int64_t gmOffset = CalLoadOffsetNHWC(shapeInfo, offset0, offset1, offset2, offset3, inputFlag);
    globalData srcGlobal((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        shapeDim4(srcN, srcH, srcW, srcC),
        strideDim4(srcStrideN, srcStrideH, srcStrideW, srcStrideC));
    pto::Layout layout = inputFlag ? pto::Layout::NC1HWC0 : pto::Layout::FRACTAL_Z;
    using tileData = pto::ConvTile<pto::TileType::Mat, T, bufferSize, layout,
        pto::ConvTileShape<-1, -1, -1, -1, c0Size>>;
    int64_t tileShape0 = inputFlag ? dstShape0 : 1;
    int64_t tileShape1 = inputFlag ? dstShape1 : dstShape0;
    int64_t tileShape2 = inputFlag ? dstShape2 : dstShape1;
    int64_t tileShape3 = inputFlag ? dstShape3 : dstShape2;
    tileData dstL1(tileShape0, tileShape1, tileShape2, tileShape3);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, srcGlobal);
    return;
}

/**
 * Copy input data from DDR to L1 with DN2NZ, input NCHW -> NC1HWC0, weight NCHW -> FZ.
 * dst: input -> AL1(NC1HWC0), weight -> BL1(FZ)
 * src: input -> GM(NCHW), weigh -> GM(NCHW)
 * offset0: input -> src_n_offset, weight -> src_n_offset
 * offset1: input -> src_c_offset,  weight -> src_c_offset
 * offset2: input -> src_h_offset, weight -> src_d_offset
 * offset3: input -> src_w_offset, weight -> 0
 * offset4: input -> src_d_offset, weight -> 0
 * isInput: 1 -> input, 0 -> weight
 */
template <typename T, typename U>
INLINE void TLoadConvDN2NZ( T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3, const int64_t &offset4, const uint8_t &isInput) {
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
    int64_t bufferSize = inputFlag ? dstShape0 * dstShape1 * dstShape2 * dstShape3 * BLOCK_ALIGN_BYTE :
                                     dstShape0 * dstShape1 * dstShape2 * BLOCK_ALIGN_BYTE;
    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    std::vector<int64_t> shapeInfo = inputFlag ? {srcC, srcH, srcW} : {srcH, srcW};
    int64_t gmOffset = CalLoadOffsetNCHW(shapeInfo, offset0, offset1, offset2, offset3, offset4, isInput);
    globalData srcGlobal((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        shapeDim4(srcN, srcC, srcH, srcW),
        strideDim4(srcStrideN, srcStrideC, srcStrideH, srcStrideW));
    pto::Layout layout = inputFlag ? pto::Layout::NC1HWC0 : pto::Layout::FRACTAL_Z;
    using tileData = pto::ConvTile<pto::TileType::Mat, T, bufferSize, layout,
        pto::ConvTileShape<-1, -1, -1, -1, c0Size>>;
    int64_t tileShape0 = inputFlag ? dstShape0 : 1;
    int64_t tileShape1 = inputFlag ? dstShape1 : dstShape0;
    int64_t tileShape2 = inputFlag ? dstShape2 : dstShape1;
    int64_t tileShape3 = inputFlag ? dstShape3 : dstShape2;
    tileData dstL1(tileShape0, tileShape1, tileShape2, tileShape3);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, srcGlobal);
    return;
}

/**
 * Copy input data from DDR to L1 with NC1HWC0 -> NC1HWC0 format.
 * dst: AL1(NC1HWC0)
 * src: GM(NC1HWC0)
 * offset0: cin1Idx
 * offset1: hiIdx
 * offset2: wiIdx
 * offset3: reserved
 */
template <typename T, typename U>
INLINE void TLoadInput5HD(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM5, "Shape Size should be 5 Dim");
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t srcN = GetConvShape<CONV_IDX_0>(src);
    int64_t srcC1 = GetConvShape<CONV_IDX_1>(src);
    int64_t srcH = GetConvShape<CONV_IDX_2>(src);
    int64_t srcW = GetConvShape<CONV_IDX_3>(src);
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);
    int64_t bufferSize = dstN * dstC1 * dstH * dstW * BLOCK_ALIGN_BYTE;
    using shapeDim5 = pto::Shape<-1, -1, -1, -1, c0Size>;
    using strideDim5 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDim5, strideDim5, pto::Layout::NC1HWC0>;
    int64_t gmOffset = (offset0 * srcH * srcW + offset1 * srcW + offset2) * c0Size;
    globalData srcGlobal((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        shapeDim5(srcN, srcC1, srcH, srcW),
        strideDim5(srcC1 * srcH * srcW * c0Size, srcH * srcW * c0Size, srcW * c0Size));
    using tileData = pto::ConvTile<pto::TileType::Mat, T, bufferSize, pto::Layout::NC1HWC0,
        pto::ConvTileShape<-1, -1, -1, -1, c0Size>>;
    tileData dstL1(dstN, dstC1, dstH, dstW);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, srcGlobal);
    return;
}

/**
 * Copy weight data from DDR to L1 with FZ -> FZ format.
 * dst: AL1(FZ -> [C1HW, N1, N0, C0])
 * src: GM(FZ -> [C1HW, N1, N0, C0])
 * offset0: nBL1Iter
 * offset1: kBL1Iter
 * offset2: nBL1
 * offset3: kBL1
 */
template <typename T, typename U>
INLINE void TLoadWeightFZ(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3) {
    constexpr auto shapeSize = Std::tuple_size<typename T::Shape>::value;
    static_assert(shapeSize == SHAPE_DIM4, "Shape Size should be 4 Dim");
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t srcC1HW = GetConvShape<CONV_IDX_0>(src);
    int64_t srcN1 = GetConvShape<CONV_IDX_1>(src);
    int64_t dstC1HW = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstN1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t bufferSize = dstC1HW * dstN1 * BLOCK_CUBE_M_N * BLOCK_ALIGN_BYTE;
    using shapeDimFZ = pto::Shape<-1, -1, BLOCK_CUBE_M_N, c0Size>;
    using strideDimFZ = pto::Stride<-1, BLOCK_CUBE_M_N * c0Size, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename U::Type, shapeDimFZ, strideDimFZ, pto::Layout::FRACTAL_Z>;
    int64_t gmOffset = offset0 * offset2 * c0Size + offset1 * offset3 * srcN1 * BLOCK_CUBE_M_N;
    globalData srcGlobal((__gm__ typename U::Type *)(src.GetAddr() + gmOffset),
        shapeDimFZ(srcC1HW, srcN1),
        strideDimFZ(srcN1 * BLOCK_CUBE_M_N * c0Size));
    using tileData = pto::ConvTile<pto::TileType::Mat, T, bufferSize, pto::Layout::FRACTAL_Z,
        pto::ConvTileShape<1, -1, -1, -1, c0Size>>;
    tileData dstL1(dstC1HW, dstN1, BLOCK_CUBE_M_N);
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());
    pto::TLOAD(dstL1, srcGlobal);
    return;
}

// Copy data from DDR to L1 with NC1HWC0 -> NC1HWC0 format || FZ -> FZ format
template <typename T, typename U>
INLINE void TLoadConvNZ2NZ(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3) {
    if (U::layout == pto::Layout::NC1HWC0) {
        TLoadInput5HD(dst, src, offset0, offset1, offset2, offset3);
    } else if (U::layout == pto::Layout::FRACTAL_Z) {
        TLoadWeightFZ(dst, src, offset0, offset1, offset2, offset3);
    }
    return;
}

// Copy data from DDR to L1
template <CopyInMode mode, typename T, typename U>
TILEOP void TLoadConv(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3, const int64_t &offset4, const int64_t &kL1, const uint8_t &isInput) {
    static_assert(T::FORMAT == Hardware::L1 && U::FORMAT == Hardware::GM,
        "[TLoadConv Error]: Src format shoulde be GM and Dst format shoulde be L1");
    if constexpr (mode == CopyInMode::ND2NZ) {
        TLoadConvND2NZ(dst, src, offset0, offset1, offset2, offset3, offset4, kL1, isInput);
    } else if constexpr (mode == CopyInMode::DN2NZ) {
        TLoadConvDN2NZ(dst, src, offset0, offset1, offset2, offset3, offset4, isInput);
    } else if constexpr (mode == CopyInMode::NZ2NZ) {
        TLoadConvNZ2NZ(dst, src, offset0, offset1, offset2, offset3);
    }
    return;
}

/**
 * Copy data from L0C to DDR with NZ -> NCHW format.
 * dst: GM(NCHW)
 * src: l0c(NZ)
 * offset0: batchOffset
 * offset1: nL1Offset
 * offset2: hL1OutOffset
 * offset3: wL1OutOffset
 * offset4: nL0Offset
 * offset5: hL0Offset
 * offset6: wL0Offset
 */
template <typename T, typename U>
INLINE void TStoreConvNZ2DN(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3, const int64_t &offset4, const int64_t &offset5, const int64_t &offset6) {
    int64_t srcM = GetConvShape<CONV_IDX_0>(src);
    int64_t srcN = GetConvShape<CONV_IDX_1>(src);
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);
    int64_t dstStrideN = GetConvStride<CONV_IDX_0>(dst);
    int64_t dstStrideC = GetConvStride<CONV_IDX_1>(dst);
    int64_t dstStrideH = GetConvStride<CONV_IDX_2>(dst);
    int64_t dstStrideW = GetConvStride<CONV_IDX_3>(dst);

    std::vector<int64_t> shapeInfo{dstC, dstH, dstW};
    int64_t gmOffset = CalStoreOffsetNCHW(shapeInfo, offset0, offset1, offset2, offset3, offset4, offset5, offset6);
    using shapeDim4 = pto::Shape<1, -1, -1, -1, -1>;
    using strideDim4 = pto::Stride<1, -1, -1, -1, -1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim4, strideDim4, pto::Layout::NCHW>;
    globalData dstGlobal((__gm__ typename T::Type *)(dst.GetAddr() + gmOffset),
        shapeDim4(dstN, dstC, dstH, dstW),
        strideDim4(dstStrideN, dstStrideC, dstStrideH, dstStrideW));
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcM, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Normal>;
    tileData srcL0C(srcM, srcN);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TSTORE(dstGlobal, srcL0C);
    return;
}

/**
 * Copy data from L0C to DDR with NZ -> NC1HWC0 format.
 * dst: GM(NC1HWC0)
 * src: l0c(NZ)
 * offset0: nBL1Iter
 * offset1: hoL1Iter
 * offset2: woL1Iter
 * offset3: nBL0Iter
 * offset4: woL0Iter
 */
template <typename T, typename U>
INLINE void TStoreConvNZ2NZ(
    T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2, const int64_t &offset3,
    const int64_t &offset4, const int64_t &nBL1, const int64_t &woL1, const int64_t &nL0, const int64_t &woL0) {
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename T::Type);
    int64_t srcM = GetConvShape<CONV_IDX_0>(src);
    int64_t srcN = GetConvShape<CONV_IDX_1>(src);
    int64_t dstN = GetConvShape<CONV_IDX_0>(dst);
    int64_t dstC1 = GetConvShape<CONV_IDX_1>(dst);
    int64_t dstH = GetConvShape<CONV_IDX_2>(dst);
    int64_t dstW = GetConvShape<CONV_IDX_3>(dst);

    std::vector<int64_t> shapeInfo{dstH, dstW, c0Size, nBL1, woL1, nL0, woL0};
    int64_t gmOffset = CalStoreOffset5hd(shapeInfo, offset0, offset1, offset2, offset3, offset4);
    using shapeDim5 = pto::Shape<-1, -1, -1, -1, c0Size>;
    using strideDim5 = pto::Stride<-1, -1, -1, c0Size, 1>;
    using globalData = pto::GlobalTensor<typename T::Type, shapeDim5, strideDim5, pto::Layout::NC1HWC0>;
    globalData dstGlobal((__gm__ typename T::Type *)(dst.GetAddr() + gmOffset),
        shapeDim5(dstN, dstC1, dstH, dstW),
        strideDim5(dstC1 * dstH * dstW * c0Size, dstH * dstW * c0Size, dstW * c0Size));
    using tileData = pto::Tile<pto::TileType::Acc, typename U::Type, srcM, srcM, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor, pto::TileConfig::fractalCSize, pto::PadValue::Null, pto::CompactMode::Normal>;
    tileData srcL0C(srcM, srcN);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TSTORE(dstGlobal, srcL0C);
    return;
}

// Copy data from L0C to DDR
template <CopyOutMode mode, typename T, typename U>
TILEOP void TStoreConv(T &dst, U &src, const int64_t &offset0, const int64_t &offset1, const int64_t &offset2,
    const int64_t &offset3, const int64_t &offset4, const int64_t &offset5, const int64_t &offset6) {
    constexpr auto srcShapeSize = Std::tuple_size<typename U::Shape>::value;
    static_assert(srcShapeSize == SHAPE_DIM2, "L0C shape size should be 2 Dim");
    static_assert(T::FORMAT == Hardware::GM && U::FORMAT == Hardware::L0C,
        "[TStoreConv Error]: Src format shoulde be L0C and Dst format shoulde be GM");
    if constexpr (mode == CopyOutMode::NZ2ND) {
        return;
    } else if constexpr (mode == CopyOutMode::NZ2DN) {
        TStoreConvNZ2DN(dst, src, offset0, offset1, offset2, offset3, offset4, offset5, offset6);
    } else if constexpr (mode == CopyOutMode::NZ2NZ) {
        // TStoreConvNZ2NZ(dst, src, offset0, offset1, offset2, offset3, offset4, nBL1, woL1, nL0, woL0);
        return;
    }
    return;
}

} // namespace TileOp
#endif // TILEOP_TILE_OPERATOR_CONV_PTO__H