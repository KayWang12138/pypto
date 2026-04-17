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
 * \file matmul_mx.h
 * \brief MX矩阵乘核心内部实现（仅A5架构）
 */

#ifndef TILEOP_TILE_OPERATOR_ARCH35_MATMUL_MX__H
#define TILEOP_TILE_OPERATOR_ARCH35_MATMUL_MX__H

#include "../utools.h"

template <bool isZeroC, typename T0, typename T1, typename T2, typename T3, typename T4>
INLINE void MatmulMXImpl(T0& c, T1& a, T2& aScale, T3& b, T4& bScale)
{
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeAScale = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto shapeSizeBScale = Std::tuple_size<typename T4::Shape>::value;

    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename T1::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename T1::TileShape>::type::value;
    constexpr auto staticL0AScaleW =
        Std::tuple_element<shapeSizeAScale - SHAPE_DIM2, typename T2::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename T3::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename T3::TileShape>::type::value;
    constexpr auto staticL0BScaleH =
        Std::tuple_element<shapeSizeBScale - SHAPE_DIM3, typename T4::TileShape>::type::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T0::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T0::TileShape>::type::value;

    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validN = GetShape<1>(b);
    int64_t validScaleK = GetShape<1>(aScale) * SHAPE_DIM2;
    if (validM == 0 || validK == 0 || validN == 0 || validScaleK == 0) {
        return;
    }
    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0AScaleTensor = pto::TileLeftScale<typename T1::Type, staticL0AH, staticL0AScaleW * SHAPE_DIM2, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename T3::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0BScaleTensor = pto::TileRightScale<typename T3::Type, staticL0BScaleH * SHAPE_DIM2, staticL0BW, -1, -1>;

    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    tileL0ATensor l0a(validM, validK);
    tileL0AScaleTensor l0aScale(validM, validScaleK);
    tileL0BTensor l0b(validK, validN);
    tileL0BScaleTensor l0bScale(validScaleK, validN);
    tileL0CTensor l0c(validM, validN);

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0aScale, (uint64_t)aScale.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0bScale, (uint64_t)bScale.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());

    if constexpr (!isZeroC) {
        pto::TMATMUL_MX(l0c, l0a, l0aScale, l0b, l0bScale);
    } else {
        pto::TMATMUL_MX(l0c, l0c, l0a, l0aScale, l0b, l0bScale);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
INLINE void MatmulMXImpl(T0& c, T1& a, T2& aScale, T3& b, T4& bScale, T5& bias)
{
    constexpr auto shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto shapeSizeAScale = Std::tuple_size<typename T2::Shape>::value;
    constexpr auto shapeSizeB = Std::tuple_size<typename T3::Shape>::value;
    constexpr auto shapeSizeBScale = Std::tuple_size<typename T4::Shape>::value;
    constexpr auto staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T0::TileShape>::type::value;
    constexpr auto staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T0::TileShape>::type::value;
    constexpr auto staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename T1::TileShape>::type::value;
    constexpr auto staticL0AW = Std::tuple_element<shapeSizeA - 1, typename T1::TileShape>::type::value;
    constexpr auto staticL0AScaleW =
        Std::tuple_element<shapeSizeAScale - SHAPE_DIM2, typename T2::TileShape>::type::value;
    constexpr auto staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename T3::TileShape>::type::value;
    constexpr auto staticL0BW = Std::tuple_element<shapeSizeB - 1, typename T3::TileShape>::type::value;
    constexpr auto staticL0BScaleH =
        Std::tuple_element<shapeSizeBScale - SHAPE_DIM3, typename T4::TileShape>::type::value;

    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validScaleK = GetShape<1>(aScale) * SHAPE_DIM2;
    int64_t validN = GetShape<1>(b);
    if (validM == 0 || validK == 0 || validN == 0 || validScaleK == 0) {
        return;
    }
    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW, -1, -1>;
    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0AScaleTensor = pto::TileLeftScale<typename T1::Type, staticL0AH, staticL0AScaleW * SHAPE_DIM2, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename T3::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0BScaleTensor = pto::TileRightScale<typename T3::Type, staticL0BScaleH * SHAPE_DIM2, staticL0BW, -1, -1>;
    using tileBiasTensor =
        pto::Tile<pto::TileType::Bias, typename T5::Type, 1, staticL0BW, pto::BLayout::RowMajor, -1, -1>;

    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    tileL0ATensor l0a(validM, validK);
    tileL0AScaleTensor l0aScale(validM, validScaleK);
    tileL0BTensor l0b(validK, validN);
    tileL0BScaleTensor l0bScale(validScaleK, validN);
    tileL0CTensor l0c(validM, validN);
    tileBiasTensor biasT(1, validM);

    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0aScale, (uint64_t)aScale.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0bScale, (uint64_t)bScale.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());
    pto::TASSIGN(biasT, (uint64_t)bias.GetAddr());
    pto::TMATMUL_MX(l0c, l0a, l0aScale, l0b, l0bScale, biasT);
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_MATMUL_MX__H