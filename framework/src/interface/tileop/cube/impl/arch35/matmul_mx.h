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

#include "../common_impl.h"

template <
    bool isZeroC, bool hasBias, typename tileL0ATensor, typename tileL0AScaleTensor, typename tileL0BTensor,
    typename tileL0BScaleTensor, typename tileL0CTensor, typename T0, typename T1, typename T2, typename T3,
    typename T4, typename T5 = void*, typename tileBiasTensor = void*>
INLINE void ExecuteMatmulMXPipeline(
    tileL0ATensor& l0a, tileL0AScaleTensor& l0aScale, tileL0BTensor& l0b, tileL0BScaleTensor& l0bScale,
    tileL0CTensor& l0c, T0& c, T1& a, T2& aScale, T3& b, T4& bScale, T5 bias, tileBiasTensor biasT)
{
    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0aScale, (uint64_t)aScale.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0bScale, (uint64_t)bScale.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());

    if constexpr (hasBias) {
        static_assert(!std::is_same_v<T5, std::nullptr_t> && !std::is_same_v<tileBiasTensor, void*>);
        pto::TASSIGN(biasT, (uint64_t)bias.GetAddr());
        pto::TMATMUL_MX(l0c, l0a, l0aScale, l0b, l0bScale, biasT);
    } else {
        if constexpr (!isZeroC) {
            pto::TMATMUL_MX(l0c, l0a, l0aScale, l0b, l0bScale);
        } else {
            pto::TMATMUL_MX(l0c, l0c, l0a, l0aScale, l0b, l0bScale);
        }
    }
}

template <bool isZeroC, bool hasBias, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
INLINE void MatmulMXStatic(T0& c, T1& a, T2& aScale, T3& b, T4& bScale, T5 bias = nullptr)
{
    constexpr int64_t shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr int64_t shapeSizeB = Std::tuple_size<typename T3::Shape>::value;
    constexpr int64_t shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    static_assert(
        shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[MatmulMX ERROR]: Shape dim size shoulde be 2");

    constexpr int64_t staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename T1::TileShape>::type::value;
    constexpr int64_t staticL0AW = Std::tuple_element<shapeSizeA - 1, typename T1::TileShape>::type::value;
    constexpr int64_t staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename T3::TileShape>::type::value;
    constexpr int64_t staticL0BW = Std::tuple_element<shapeSizeB - 1, typename T3::TileShape>::type::value;
    constexpr int64_t staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T0::TileShape>::type::value;
    constexpr int64_t staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T0::TileShape>::type::value;

    constexpr int64_t staticValidM = TileOp::GetTensorShapeDim<T1, shapeSizeA - SHAPE_DIM2>();
    constexpr int64_t staticValidK = TileOp::GetTensorShapeDim<T1, shapeSizeA - 1>();
    constexpr int64_t staticValidN = TileOp::GetTensorShapeDim<T3, shapeSizeB - 1>();
    if constexpr (staticValidM == 0 || staticValidK == 0 || staticValidN == 0) {
        return;
    }

    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW, staticValidM, staticValidK>;
    using tileL0AScaleTensor = pto::TileLeftScaleCompact<
        typename T2::Type, staticL0AH, staticL0AW * SHAPE_DIM2, staticValidM, staticValidK * SHAPE_DIM2>;
    using tileL0BTensor = pto::TileRight<typename T3::Type, staticL0BH, staticL0BW, staticValidK, staticValidN>;
    using tileL0BScaleTensor = pto::TileRightScaleCompact<
        typename T4::Type, staticL0BH * SHAPE_DIM2, staticL0BW, staticValidK * SHAPE_DIM2, staticValidN>;
    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW, staticValidM, staticValidN>;

    tileL0ATensor l0a;
    tileL0AScaleTensor l0aScale;
    tileL0BTensor l0b;
    tileL0BScaleTensor l0bScale;
    tileL0CTensor l0c;

    if constexpr (hasBias) {
        using tileBiasTensor =
            pto::Tile<pto::TileType::Bias, typename T5::Type, 1, staticL0CW, pto::BLayout::RowMajor, 1, staticValidN>;
        tileBiasTensor biasT;
        ExecuteMatmulMXPipeline<isZeroC, hasBias>(
            l0a, l0aScale, l0b, l0bScale, l0c, c, a, aScale, b, bScale, bias, biasT);
    } else {
        ExecuteMatmulMXPipeline<isZeroC, hasBias>(
            l0a, l0aScale, l0b, l0bScale, l0c, c, a, aScale, b, bScale, nullptr, nullptr);
    }
}

template <bool isZeroC, bool hasBias, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
INLINE void MatmulMXDynamic(T0& c, T1& a, T2& aScale, T3& b, T4& bScale, T5 bias = nullptr)
{
    constexpr int64_t shapeSizeA = Std::tuple_size<typename T1::Shape>::value;
    constexpr int64_t shapeSizeB = Std::tuple_size<typename T3::Shape>::value;
    constexpr int64_t shapeSizeC = Std::tuple_size<typename T0::Shape>::value;
    static_assert(
        shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[MatmulMX ERROR]: Shape dim size shoulde be 2");

    constexpr int64_t staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename T1::TileShape>::type::value;
    constexpr int64_t staticL0AW = Std::tuple_element<shapeSizeA - 1, typename T1::TileShape>::type::value;
    constexpr int64_t staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename T3::TileShape>::type::value;
    constexpr int64_t staticL0BW = Std::tuple_element<shapeSizeB - 1, typename T3::TileShape>::type::value;
    constexpr int64_t staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T0::TileShape>::type::value;
    constexpr int64_t staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T0::TileShape>::type::value;

    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validN = GetShape<1>(b);
    if (validM == 0 || validK == 0 || validN == 0) {
        return;
    }
    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;

    using tileL0ATensor = pto::TileLeft<typename T1::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0AScaleTensor =
        pto::TileLeftScaleCompact<typename T2::Type, staticL0AH, staticL0AW * SHAPE_DIM2, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename T3::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0BScaleTensor =
        pto::TileRightScaleCompact<typename T4::Type, staticL0BH * SHAPE_DIM2, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename T0::Type, staticL0CH, staticL0CW, -1, -1>;

    tileL0ATensor l0a(validM, validK);
    tileL0AScaleTensor l0aScale(validM, validK * SHAPE_DIM2);
    tileL0BTensor l0b(validK, validN);
    tileL0BScaleTensor l0bScale(validK * SHAPE_DIM2, validN);
    tileL0CTensor l0c(validM, validN);

    if constexpr (hasBias) {
        using tileBiasTensor =
            pto::Tile<pto::TileType::Bias, typename T5::Type, 1, staticL0CW, pto::BLayout::RowMajor, -1, -1>;
        tileBiasTensor biasT(1, validN);
        ExecuteMatmulMXPipeline<isZeroC, hasBias>(
            l0a, l0aScale, l0b, l0bScale, l0c, c, a, aScale, b, bScale, bias, biasT);
    } else {
        ExecuteMatmulMXPipeline<isZeroC, hasBias>(
            l0a, l0aScale, l0b, l0bScale, l0c, c, a, aScale, b, bScale, nullptr, nullptr);
    }
}

#endif // TILEOP_TILE_OPERATOR_ARCH35_MATMUL_MX__H