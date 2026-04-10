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
 * \file mmad_impl.h
 * \brief 通用矩阵乘内部实现
 */

#ifndef TILEOP_TILE_OPERATOR_MATMUL__H
#define TILEOP_TILE_OPERATOR_MATMUL__H

#include "cube_tools.h"

template <
    bool isZeroC, TransMode transMode, bool hasBias, typename tileL0ATensor, typename tileL0BTensor,
    typename tileL0CTensor, typename T0, typename T1, typename T2, typename T3 = void*, typename tileBiasTensor = void*>
INLINE void ExecuteMatmulPipeline(
    tileL0ATensor& l0a, tileL0BTensor& l0b, tileL0CTensor& l0c, T0& c, T1& a, T2& b, T3 bias = nullptr,
    tileBiasTensor biasT = nullptr)
{
    if constexpr (std::is_same_v<typename tileL0ATensor::DType, float>) {
        l0a.ResetMadMode();
        if constexpr (!hasBias) {
            l0a.SetKAligned(true);
        }
    }
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.SetMadTF32Mode(static_cast<pto::RoundMode>(transMode));
    }
    pto::TASSIGN(l0a, (uint64_t)a.GetAddr());
    pto::TASSIGN(l0b, (uint64_t)b.GetAddr());
    pto::TASSIGN(l0c, (uint64_t)c.GetAddr());
    if constexpr (hasBias) {
        static_assert(!std::is_same_v<T3, std::nullptr_t>);
        pto::TASSIGN(biasT, (uint64_t)bias.GetAddr());
    }
    if constexpr (hasBias) {
        pto::TMATMUL_BIAS(l0c, l0a, l0b, biasT);
    } else {
        if constexpr (!isZeroC) {
            pto::TMATMUL(l0c, l0a, l0b);
        } else {
            pto::TMATMUL_ACC(l0c, l0c, l0a, l0b);
        }
    }
    if constexpr (transMode != TransMode::CAST_NONE) {
        l0a.ResetMadMode();
    }
}

template <bool isZeroC, TransMode transMode, bool hasBias, typename T, typename U, typename V, typename T3 = void*>
INLINE void TMatmulStatic(T& c, U& a, V& b, T3 bias = nullptr)
{
    constexpr int64_t shapeSizeA = Std::tuple_size<typename U::Shape>::value;
    constexpr int64_t shapeSizeB = Std::tuple_size<typename V::Shape>::value;
    constexpr int64_t shapeSizeC = Std::tuple_size<typename T::Shape>::value;
    static_assert(
        shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[Matmul ERROR]: Shape dim size shoulde be 2");

    constexpr int64_t staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL0AW = Std::tuple_element<shapeSizeA - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename V::TileShape>::type::value;
    constexpr int64_t staticL0BW = Std::tuple_element<shapeSizeB - 1, typename V::TileShape>::type::value;
    constexpr int64_t staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T::TileShape>::type::value;

    constexpr int64_t staticValidM = TileOp::GetTensorShapeDim<U, shapeSizeA - SHAPE_DIM2>();
    constexpr int64_t staticValidK = TileOp::GetTensorShapeDim<U, shapeSizeA - 1>();
    constexpr int64_t staticValidN = TileOp::GetTensorShapeDim<V, shapeSizeB - 1>();
    if constexpr (staticValidM == 0 || staticValidK == 0 || staticValidN == 0) {
        return;
    }

    using tileL0ATensor = pto::TileLeft<typename U::Type, staticL0AH, staticL0AW, staticValidM, staticValidK>;
    using tileL0BTensor = pto::TileRight<typename V::Type, staticL0BH, staticL0BW, staticValidK, staticValidN>;
    using tileL0CTensor = pto::TileAcc<typename T::Type, staticL0CH, staticL0CW, staticValidM, staticValidN>;

    tileL0ATensor l0a;
    tileL0BTensor l0b;
    tileL0CTensor l0c;

    if constexpr (hasBias) {
        using tileBiasTensor =
            pto::Tile<pto::TileType::Bias, typename T3::Type, 1, staticL0BW, pto::BLayout::RowMajor, 1, staticValidN>;
        tileBiasTensor biasT;
        ExecuteMatmulPipeline<isZeroC, transMode, hasBias>(l0a, l0b, l0c, c, a, b, bias, biasT);
    } else {
        ExecuteMatmulPipeline<isZeroC, transMode, hasBias>(l0a, l0b, l0c, c, a, b);
    }
}

template <bool isZeroC, TransMode transMode, bool hasBias, typename T, typename U, typename V, typename T3 = void*>
INLINE void TMatmulDynamic(T& c, U& a, V& b, T3 bias = nullptr)
{
    constexpr int64_t shapeSizeA = Std::tuple_size<typename U::Shape>::value;
    constexpr int64_t shapeSizeB = Std::tuple_size<typename V::Shape>::value;
    constexpr int64_t shapeSizeC = Std::tuple_size<typename T::Shape>::value;
    static_assert(
        shapeSizeA == SHAPE_DIM2 && shapeSizeB == SHAPE_DIM2 && shapeSizeC == SHAPE_DIM2,
        "[Matmul ERROR]: Shape dim size shoulde be 2");

    constexpr int64_t staticL0AH = Std::tuple_element<shapeSizeA - SHAPE_DIM2, typename U::TileShape>::type::value;
    constexpr int64_t staticL0AW = Std::tuple_element<shapeSizeA - 1, typename U::TileShape>::type::value;
    constexpr int64_t staticL0BH = Std::tuple_element<shapeSizeB - SHAPE_DIM2, typename V::TileShape>::type::value;
    constexpr int64_t staticL0BW = Std::tuple_element<shapeSizeB - 1, typename V::TileShape>::type::value;
    constexpr int64_t staticL0CH = Std::tuple_element<shapeSizeC - SHAPE_DIM2, typename T::TileShape>::type::value;
    constexpr int64_t staticL0CW = Std::tuple_element<shapeSizeC - 1, typename T::TileShape>::type::value;

    int64_t validM = GetShape<0>(a);
    int64_t validK = GetShape<1>(a);
    int64_t validN = GetShape<1>(b);
    if (validM == 0 || validK == 0 || validN == 0) {
        return;
    }
    validM = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;

    using tileL0ATensor = pto::TileLeft<typename U::Type, staticL0AH, staticL0AW, -1, -1>;
    using tileL0BTensor = pto::TileRight<typename V::Type, staticL0BH, staticL0BW, -1, -1>;
    using tileL0CTensor = pto::TileAcc<typename T::Type, staticL0CH, staticL0CW, -1, -1>;

    tileL0ATensor l0a(validM, validK);
    tileL0BTensor l0b(validK, validN);
    tileL0CTensor l0c(validM, validN);

    if constexpr (hasBias) {
        using tileBiasTensor =
            pto::Tile<pto::TileType::Bias, typename T3::Type, 1, staticL0BW, pto::BLayout::RowMajor, -1, -1>;
        tileBiasTensor biasT(1, validN);
        ExecuteMatmulPipeline<isZeroC, transMode, hasBias>(l0a, l0b, l0c, c, a, b, bias, biasT);
    } else {
        ExecuteMatmulPipeline<isZeroC, transMode, hasBias>(l0a, l0b, l0c, c, a, b);
    }
}

#endif // TILEOP_TILE_OPERATOR_MATMUL__H