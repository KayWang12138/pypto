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
 * \file extract_common.h
 * \brief 通用数据提取内部实现
 */

#ifndef TILEOP_TILE_OPERATOR_EXTRACT_COMMON__H
#define TILEOP_TILE_OPERATOR_EXTRACT_COMMON__H

#include "common_impl.h"


// Copy data from L0C to L1(Extract)
template <typename config, typename l1Data, typename l0cData, typename V>
INLINE void TExtractL0CToL1(
    l1Data& dstL1, l0cData& srcL0C, V& fixbuf, uint16_t l0cOffset0, uint16_t l0cOffset1, uint64_t scaleValue = 0)
{
    if constexpr (
        std::is_same<typename l0cData::DType, int32_t>::value && std::is_same<typename l1Data::DType, half>::value) {
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode =
                (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu;
            pto::TEXTRACT<l1Data, l0cData, relu_mode>(dstL1, srcL0C, scaleValue, l0cOffset0, l0cOffset1);
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, (uint64_t)fixbuf.GetAddr());
            pto::TEXTRACT_FP<
                l1Data, l0cData, decltype(scaleData),
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstL1, srcL0C, scaleData, l0cOffset0, l0cOffset1);
        }
    } else {
        pto::TEXTRACT<
            l1Data, l0cData, config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstL1, srcL0C, l0cOffset0, l0cOffset1);
    }
}

// Copy data from L0C to L1(Insert)
template <typename config, typename l1Data, typename l0cData, typename V>
INLINE void TInsertL0CToL1(
    l1Data& dstL1, l0cData& srcL0C, V& fixbuf, uint16_t l1Offset0, uint16_t l1Offset1, uint64_t scaleValue = 0)
{
    if constexpr (
        std::is_same<typename l0cData::DType, int32_t>::value && std::is_same<typename l1Data::DType, half>::value) {
        if (scaleValue != 0) {
            constexpr pto::ReluPreMode relu_mode =
                (config::kReluMode == 0) ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu;
            pto::TINSERT<l1Data, l0cData, relu_mode>(dstL1, srcL0C, scaleValue, l1Offset0, l1Offset1);
        } else {
            auto scaleData = CreateScaleTileData(fixbuf);
            pto::TASSIGN(scaleData, (uint64_t)fixbuf.GetAddr());
            pto::TINSERT_FP<
                l1Data, l0cData, decltype(scaleData),
                config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
                dstL1, srcL0C, scaleData, l1Offset0, l1Offset1);
        }
    } else {
        pto::TINSERT<l1Data, l0cData, config::kReluMode == 0 ? pto::ReluPreMode::NoRelu : pto::ReluPreMode::NormalRelu>(
            dstL1, srcL0C, l1Offset0, l1Offset1);
    }
}

template <typename config, typename T, typename U, typename V>
INLINE void TExtractL0CToL1Static(
    T& dst, U& src, V& fixbuf, uint16_t l1Offset0, uint16_t l1Offset1, uint16_t l0cOffset0, uint16_t l0cOffset1,
    uint64_t scaleValue)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t staticDstShape0 = TileOp::GetTensorShapeDim<T, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticDstShape1 = TileOp::GetTensorShapeDim<T, shapeSize - 1>();
    constexpr int64_t staticSrcShape0 = TileOp::GetTensorShapeDim<U, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticSrcShape1 = TileOp::GetTensorShapeDim<U, shapeSize - 1>();
    constexpr int64_t tileL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t tileL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t tileL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t tileL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    using L1Tile = pto::Tile<
        pto::TileType::Mat, typename T::Type, tileL1H, tileL1W,
        config::kMode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, staticDstShape0,
        staticDstShape1, config::kMode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;

    using L0CTile = pto::Tile<
        pto::TileType::Acc, typename U::Type, tileL0CH, tileL0CW, pto::BLayout::ColMajor, staticSrcShape0,
        staticSrcShape1, pto::SLayout::RowMajor>;

    L1Tile dstL1;
    L0CTile srcL0C;
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());

    if constexpr (staticDstShape0 < staticSrcShape0 || staticDstShape1 < staticSrcShape1) {
        TExtractL0CToL1<config, L1Tile, L0CTile, V>(dstL1, srcL0C, fixbuf, l0cOffset0, l0cOffset1, scaleValue);
    } else {
        TInsertL0CToL1<config, L1Tile, L0CTile, V>(dstL1, srcL0C, fixbuf, l1Offset0, l1Offset1, scaleValue);
    }
}

template <typename config, typename T, typename U, typename V>
INLINE void TExtractL0CToL1Dynamic(
    T& dst, U& src, V& fixbuf, uint16_t l1Offset0, uint16_t l1Offset1, uint16_t l0cOffset0, uint16_t l0cOffset1,
    uint64_t scaleValue)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t tileL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t tileL1W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t tileL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t tileL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;

    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);

    using L1Tile = pto::Tile<
        pto::TileType::Mat, typename T::Type, tileL1H, tileL1W,
        config::kMode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        config::kMode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;

    using L0CTile = pto::Tile<
        pto::TileType::Acc, typename U::Type, tileL0CH, tileL0CW, pto::BLayout::ColMajor, -1, -1,
        pto::SLayout::RowMajor>;

    L1Tile dstL1(dstShape0, dstShape1);
    L0CTile srcL0C(srcShape0, srcShape1);
    pto::TASSIGN(srcL0C, (uint64_t)src.GetAddr());
    pto::TASSIGN(dstL1, (uint64_t)dst.GetAddr());

    if (dstShape0 < srcShape0 || dstShape1 < srcShape1) {
        TExtractL0CToL1<config, L1Tile, L0CTile, V>(dstL1, srcL0C, fixbuf, l0cOffset0, l0cOffset1, scaleValue);
    } else {
        TInsertL0CToL1<config, L1Tile, L0CTile, V>(dstL1, srcL0C, fixbuf, l1Offset0, l1Offset1, scaleValue);
    }
}

template <bool isTrans, typename T, typename U>
INLINE void TExtractL1ToL0Static(T& dst, U& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticL0H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticL0W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t staticDstShape0 = TileOp::GetTensorShapeDim<T, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticDstShape1 = TileOp::GetTensorShapeDim<T, shapeSize - 1>();
    constexpr int64_t staticSrcShape0 = TileOp::GetTensorShapeDim<U, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticSrcShape1 = TileOp::GetTensorShapeDim<U, shapeSize - 1>();
    constexpr int64_t srcValidH = isTrans ? staticSrcShape1 : staticSrcShape0;
    constexpr int64_t srcValidW = isTrans ? staticSrcShape0 : staticSrcShape1;
    using tileL1Tensor = pto::Tile<
        pto::TileType::Mat, typename U::Type, isTrans ? staticL1W : staticL1H, isTrans ? staticL1H : staticL1W,
        isTrans ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, srcValidH, srcValidW,
        isTrans ? pto::SLayout::ColMajor : pto::SLayout::RowMajor>;

    using tileL0Tensor = std::conditional_t<
        T::FORMAT == Hardware::L0A,
        pto::TileLeftCompact<typename T::Type, staticL0H, staticL0W, staticDstShape0, staticDstShape1>,
        pto::TileRightCompact<typename T::Type, staticL0H, staticL0W, staticDstShape0, staticDstShape1>>;
    tileL1Tensor l1Tile;
    tileL0Tensor l0Tile;
    if constexpr (std::is_same<typename tileL0Tensor::DType, float>::value && T::FORMAT == Hardware::L0A) {
        l0Tile.SetKAligned(true);
    }
    pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
    pto::TASSIGN(l0Tile, (uint64_t)dst.GetAddr());
    pto::TEXTRACT(l0Tile, l1Tile, isTrans ? offset1 : offset0, isTrans ? offset0 : offset1);
}

template <bool isTrans, typename T, typename U>
INLINE void TExtractL1ToL0Dynamic(T& dst, U& src, const int64_t& offset0, const int64_t& offset1)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t staticL1H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticL0H = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticL0W = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    using tileL1Tensor = pto::Tile<
        pto::TileType::Mat, typename U::Type, isTrans ? staticL1W : staticL1H, isTrans ? staticL1H : staticL1W,
        isTrans ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        isTrans ? pto::SLayout::ColMajor : pto::SLayout::RowMajor>;
    using tileL0Tensor = std::conditional_t<
        T::FORMAT == Hardware::L0A, pto::TileLeftCompact<typename T::Type, staticL0H, staticL0W, -1, -1>,
        pto::TileRightCompact<typename T::Type, staticL0H, staticL0W, -1, -1>>;
    tileL1Tensor l1Tile(srcShape0, srcShape1);
    tileL0Tensor l0Tile(dstShape0, dstShape1);
    if (std::is_same<typename tileL0Tensor::DType, float>::value && T::FORMAT == Hardware::L0A) {
        l0Tile.SetKAligned(true);
    }
    pto::TASSIGN(l1Tile, (uint64_t)src.GetAddr());
    pto::TASSIGN(l0Tile, (uint64_t)dst.GetAddr());
    pto::TEXTRACT(l0Tile, l1Tile, isTrans ? offset1 : offset0, isTrans ? offset0 : offset1);
}

template <bool isTrans, typename T, typename U>
INLINE void TExtractL1ToL0(T& dst, U& src, const int64_t& offset0, const int64_t& offset1)
{
    if constexpr (TileOp::IsConstContinous<T, U>()) {
        TExtractL1ToL0Static<isTrans, T, U>(dst, src, offset0, offset1);
    } else {
        TExtractL1ToL0Dynamic<isTrans, T, U>(dst, src, offset0, offset1);
    }
}

template <bool isTrans, typename T, typename U>
INLINE void TExtractL1ToBTOrFB(T& dst, U& src)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t staticL1W = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticL0BW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;

    if constexpr (TileOp::IsConstContinous<T, U>()) {
        constexpr int64_t staticNL1 = TileOp::GetTensorShapeDim<U, 1>();
        constexpr int64_t staticNL0 = TileOp::GetTensorShapeDim<T, 1>();
        using tileL1Tensor =
            pto::Tile<pto::TileType::Mat, typename U::Type, 1, staticL1W, pto::BLayout::RowMajor, 1, staticNL1>;
        using tileBiasOrFbTensor = pto::Tile<
            T::FORMAT == Hardware::BIAS ? pto::TileType::Bias : pto::TileType::Scaling, typename T::Type, 1, staticL0BW,
            pto::BLayout::RowMajor, 1, staticNL0>;
        tileL1Tensor l1Tensor;
        tileBiasOrFbTensor biasOrFbTensor;
        pto::TASSIGN<tileL1Tensor>(l1Tensor, (uint64_t)src.GetAddr());
        pto::TASSIGN<tileBiasOrFbTensor>(biasOrFbTensor, (uint64_t)dst.GetAddr());
        pto::TMOV(biasOrFbTensor, l1Tensor);
    } else {
        int64_t nL1 = GetShape<1>(src);
        int64_t nL0 = GetShape<1>(dst);
        using tileL1Tensor =
            pto::Tile<pto::TileType::Mat, typename U::Type, 1, staticL1W, pto::BLayout::RowMajor, -1, -1>;
        using tileBiasOrFbTensor = pto::Tile<
            T::FORMAT == Hardware::BIAS ? pto::TileType::Bias : pto::TileType::Scaling, typename T::Type, 1, staticL0BW,
            pto::BLayout::RowMajor, -1, -1>;

        tileL1Tensor l1Tensor(1, nL1);
        tileBiasOrFbTensor biasOrFbTensor(1, nL0);
        pto::TASSIGN<tileL1Tensor>(l1Tensor, (uint64_t)src.GetAddr());
        pto::TASSIGN<tileBiasOrFbTensor>(biasOrFbTensor, (uint64_t)dst.GetAddr());
        pto::TMOV(biasOrFbTensor, l1Tensor);
    }
}

template <CopyOutMode mode, typename Coord, typename T, typename U>
INLINE void TExtractL0CToUBStatic(T& dst, U& src, const Coord& coord, int16_t subblockId)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();
    constexpr int64_t staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticUBW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t staticL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticSrcShape0 = TileOp::GetTensorShapeDim<U, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticSrcShape1 = TileOp::GetTensorShapeDim<U, shapeSize - 1>();
    constexpr int64_t staticDstShape0 = TileOp::GetTensorShapeDim<T, shapeSize - SHAPE_DIM2>();
    constexpr int64_t staticDstShape1 = TileOp::GetTensorShapeDim<T, shapeSize - 1>();
    int64_t l0cOffset = CalNZOffset(staticSrcShape0, staticSrcShape1, offset0, offset1, c0Size);

    using tileUBTensor = pto::Tile<
        pto::TileType::Vec, typename T::Type, staticUBH, staticUBW,
        mode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, staticDstShape0, staticDstShape1,
        mode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using tileL0CTensor = pto::TileAcc<typename U::Type, staticL0CH, staticL0CW, staticSrcShape0, staticSrcShape1>;

    tileUBTensor UBTile;
    tileL0CTensor l0cTile;
    pto::TASSIGN(UBTile, (uint64_t)dst.GetAddr());
    pto::TASSIGN(l0cTile, (uint64_t)src.GetAddr() + l0cOffset);
    if (subblockId == 0) {
        pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec0>(UBTile, l0cTile);
    } else {
        pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec1>(UBTile, l0cTile);
    }
}

template <CopyOutMode mode, typename Coord, typename T, typename U>
INLINE void TExtractL0CToUBDynamic(T& dst, U& src, const Coord& coord, int16_t subblockId)
{
    constexpr uint16_t shapeSize = Std::tuple_size<typename T::Shape>::value;
    constexpr int64_t c0Size = BLOCK_ALIGN_BYTE / sizeof(typename U::Type);
    int64_t offset0 = coord.GetValue();
    int64_t offset1 = static_cast<const Std::tuple<size_t>&>(coord).GetValue();
    constexpr int64_t staticUBH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticUBW = Std::tuple_element<shapeSize - 1, typename T::TileShape>::type::value;
    constexpr int64_t staticL0CH = Std::tuple_element<shapeSize - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL0CW = Std::tuple_element<shapeSize - 1, typename U::TileShape>::type::value;
    int64_t srcShape0 = GetShape<0>(src);
    int64_t srcShape1 = GetShape<1>(src);
    int64_t dstShape0 = GetShape<0>(dst);
    int64_t dstShape1 = GetShape<1>(dst);
    int64_t l0cOffset = CalNZOffset(srcShape0, srcShape1, offset0, offset1, c0Size);

    using tileUBTensor = pto::Tile<
        pto::TileType::Vec, typename T::Type, staticUBH, staticUBW,
        mode == CopyOutMode::NZ2ND ? pto::BLayout::RowMajor : pto::BLayout::ColMajor, -1, -1,
        mode == CopyOutMode::NZ2ND ? pto::SLayout::NoneBox : pto::SLayout::RowMajor>;
    using tileL0CTensor = pto::TileAcc<typename U::Type, staticL0CH, staticL0CW, -1, -1>;

    tileUBTensor UBTile(dstShape0, dstShape1);
    tileL0CTensor l0cTile(srcShape0, srcShape1);
    pto::TASSIGN(UBTile, (uint64_t)dst.GetAddr());
    pto::TASSIGN(l0cTile, (uint64_t)src.GetAddr() + l0cOffset);
    if (subblockId == 0) {
        pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec0>(UBTile, l0cTile);
    } else {
        pto::TMOV<tileUBTensor, tileL0CTensor, pto::AccToVecMode::SingleModeVec1>(UBTile, l0cTile);
    }
}

#endif // TILEOP_TILE_OPERATOR_EXTRACT_COMMON__H