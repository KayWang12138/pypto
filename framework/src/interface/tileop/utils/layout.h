/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layout.h
 * \brief
 */

#ifndef TILEOP_UTILS_LAYOUT_H
#define TILEOP_UTILS_LAYOUT_H

#include "tuple.h"
#include "../tileop_common.h"

constexpr size_t DIM_1ST = 0;
constexpr size_t DIM_2ND = 1;
constexpr size_t DIM_3RD = 2;
constexpr size_t DIM_4TH = 3;
constexpr size_t DIM_5TH = 4;

namespace TileOp {
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType &coord, const ShapeType &shape, const StrideType &stride);

template <typename... Shapes>
using Shape = Std::tuple<Shapes...>;

template <typename... Strides>
using Stride = Std::tuple<Strides...>;

template <typename... TileShapes>
using TileShape = Std::tuple<TileShapes...>;

template <typename... Coords>
using Coord = Std::tuple<Coords...>;

template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts &...t) {
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts &...t) {
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr TileShape<Ts...> MakeTileShape(const Ts &...t) {
    return {t...};
}

template <typename ShapeType, typename StrideType, typename TileShapeType>
struct Layout : private Std::tuple<ShapeType, StrideType, TileShapeType> {
    using Shape = ShapeType;
    using Stride = StrideType;
    using TileShape = TileShapeType;
    __aicore__ inline constexpr Layout(
        const ShapeType &shape = {}, const StrideType &stride = {}, const TileShapeType &tileShape = {})
        : Std::tuple<ShapeType, StrideType, TileShapeType>(shape, stride, tileShape) {
        static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType> && Std::is_tuple_v<TileShapeType>,
            "Shape, Stride or TileShape is not tuple!");
    }

    __aicore__ inline constexpr decltype(auto) layout() { return *this; }

    __aicore__ inline constexpr decltype(auto) layout() const { return *this; }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape() {
        return GetValue<0, I...>(static_cast<Std::tuple<ShapeType, StrideType, TileShapeType> &>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape() const {
        return GetValue<0, I...>(static_cast<const Std::tuple<ShapeType, StrideType, TileShapeType> &>(*this));
    }

    template <size_t index, size_t expect_size = Std::tuple_size<ShapeType>::value>
    __aicore__ inline constexpr decltype(auto) GetShapeDim() const {
        static_assert(index < expect_size, "The index of Shape is out of range.");
        constexpr auto size = Std::tuple_size<ShapeType>::value;
        if constexpr (Std::IsIntegralConstantV<ShapeType> == true) {
            return GetStaticDim<ShapeType, index, expect_size>();
        } else {
            return GetDynDim<ShapeType, index, expect_size>(GetShape());
        }
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride() {
        return GetValue<1, I...>(static_cast<Std::tuple<ShapeType, StrideType, TileShapeType> &>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride() const {
        return GetValue<1, I...>(static_cast<const Std::tuple<ShapeType, StrideType, TileShapeType> &>(*this));
    }

    template <size_t index, size_t expect_size = Std::tuple_size<StrideType>::value>
    __aicore__ inline constexpr decltype(auto) GetStrideDim() const {
        static_assert(index < expect_size, "The index of Stride is out of range.");
        constexpr auto size = Std::tuple_size<StrideType>::value;
        if constexpr (Std::IsIntegralConstantV<StrideType> == true) {
            return GetStaticDim<StrideType, index, expect_size>();
        } else {
            return GetDynDim<StrideType, index, expect_size, Kind::STRIDE>(GetStride());
        }
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetTileShape() {
        return GetValue<2, I...>(static_cast<Std::tuple<ShapeType, StrideType, TileShapeType> &>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetTileShape() const {
        return GetValue<2, I...>(static_cast<const Std::tuple<ShapeType, StrideType, TileShapeType> &>(*this));
    }

    template <size_t index, size_t expect_size = Std::tuple_size<TileShapeType>::value>
    __aicore__ inline constexpr decltype(auto) GetTileShapeDim() const {
        static_assert(index < expect_size, "The index of TileShape is out of range.");
        constexpr auto size = Std::tuple_size<TileShapeType>::value;
        if constexpr (Std::IsIntegralConstantV<TileShapeType> == true) {
            return GetStaticDim<TileShapeType, index, expect_size>();
        } else {
            return GetDynDim<TileShapeType, index, expect_size, Kind::TILE>(GetTileShape());
        }
    }

    __aicore__ inline constexpr auto IsStaticLayout() const { return Std::IsIntegralConstantV<ShapeType> == true; }

    template <typename CoordType>
    __aicore__ inline constexpr auto operator()(const CoordType &coord) const {
        return Crd2Idx(coord, GetShape(), GetStride());
    }

    template <typename Tuple, size_t expect_size = Std::tuple_size<Tuple>::value>
    __aicore__ inline constexpr decltype(auto) GetGmOffset(const Tuple &coordinate) const {
        auto s0 = GetStrideDim<DIM_1ST, expect_size>();
        auto s1 = GetStrideDim<DIM_2ND, expect_size>();
        auto s2 = GetStrideDim<DIM_3RD, expect_size>();
        auto s3 = GetStrideDim<DIM_4TH, expect_size>();
        auto s4 = GetStrideDim<DIM_5TH, expect_size>();
        auto c0 = GetDynDim<Tuple, DIM_1ST, expect_size, Kind::OTHER>(coordinate);
        auto c1 = GetDynDim<Tuple, DIM_2ND, expect_size, Kind::OTHER>(coordinate);
        auto c2 = GetDynDim<Tuple, DIM_3RD, expect_size, Kind::OTHER>(coordinate);
        auto c3 = GetDynDim<Tuple, DIM_4TH, expect_size, Kind::OTHER>(coordinate);
        auto c4 = GetDynDim<Tuple, DIM_5TH, expect_size, Kind::OTHER>(coordinate);
        return s4 * c4 + s3 * c3 + s2 * c2 + s1 * c1 + s0 * c0;
    }

private:
    enum class Kind : uint8_t { SHAPE = 0, STRIDE, TILE, OTHER };

    template <size_t index, size_t I, size_t... Is, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple &t) {
        auto tupleEle = Std::get<index>(t);
        return Std::make_tuple(Std::get<I>(tupleEle), Std::get<Is>(tupleEle)...);
    }

    template <size_t index, size_t I, size_t... Is, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple &t) const {
        auto tupleEle = Std::get<index>(t);
        return Std::make_tuple(Std::get<I>(tupleEle), Std::get<Is>(tupleEle)...);
    }

    template <size_t index, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple &t) {
        return Std::get<index>(t);
    }

    template <size_t index, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple &t) const {
        return Std::get<index>(t);
    }

    template <typename Tuple, size_t index, size_t expect_size = Std::tuple_size<Tuple>::value, Kind k = Kind::SHAPE>
    __aicore__ inline constexpr decltype(auto) GetStaticDim() const {
        static_assert(index < expect_size, "Out of range.");
        constexpr auto size = Std::tuple_size<Tuple>::value;
        if constexpr (size >= expect_size || index >= (expect_size - size)) {
            return Std::tuple_element<index + size - expect_size, Tuple>::type::value;
        } else {
            if constexpr (k == Kind::SHAPE) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    template <typename Tuple, size_t index, size_t expect_size = Std::tuple_size<Tuple>::value, Kind k = Kind::SHAPE>
    __aicore__ inline constexpr decltype(auto) GetDynDim(const Tuple &t) const {
        static_assert(index < expect_size, "Out of range.");
        constexpr auto size = Std::tuple_size<Tuple>::value;
        if constexpr (size >= expect_size || index >= (expect_size - size)) {
            return Std::get<index + size - expect_size>(t);
        } else {
            if constexpr (k == Kind::SHAPE) {
                return 1;
            } else {
                return 0;
            }
        }
    }
};

template <typename ShapeType, typename StrideType, typename TileShapeType>
__aicore__ inline constexpr auto MakeLayout(
    const ShapeType &shape, const StrideType &stride, const TileShapeType &tileShape) {
    return Layout<ShapeType, StrideType, TileShapeType>(shape, stride, tileShape);
}

template <typename T>
struct is_layout : Std::false_type {};

template <typename ShapeType, typename StrideType, typename TileShapeType>
struct is_layout<Layout<ShapeType, StrideType, TileShapeType>> : Std::true_type {};

template <typename T>
constexpr bool is_layout_v = is_layout<T>::value;

template <typename StrideType>
__aicore__ inline constexpr auto GetOuterStride() {
    return Std::tuple_element<0, StrideType>::type::value;
}

template <int leftAxis, int rightAxis, typename Shape>
__aicore__ inline constexpr size_t GetAnyAxisMergeResult() {
    constexpr auto n0 = []() constexpr {
        if constexpr (leftAxis <= 1 && 1 <= rightAxis) {
            return Std::tuple_element<DIM_1ST, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    constexpr auto n1 = []() constexpr {
        if constexpr (leftAxis <= 2 && 2 <= rightAxis) {
            return Std::tuple_element<DIM_2ND, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    constexpr auto n2 = []() constexpr {
        if constexpr (leftAxis <= 3 && 3 <= rightAxis) {
            return Std::tuple_element<DIM_3RD, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    constexpr auto n3 = []() constexpr {
        if constexpr (leftAxis <= 4 && 4 <= rightAxis) {
            return Std::tuple_element<DIM_4TH, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    constexpr auto n4 = []() constexpr {
        if constexpr (leftAxis <= 5 && 5 <= rightAxis) {
            return Std::tuple_element<DIM_5TH, Shape>::type::value;
        } else {
            return 1;
        }
    }();
    return n0 * n1 * n2 * n3 * n4;
}

template <size_t shapeSize, typename Shape>
__aicore__ inline constexpr size_t GetNonFirstAxisMergeResult() {
    if constexpr (shapeSize == 5) {
        constexpr auto n1 = Std::tuple_element<DIM_2ND, Shape>::type::value;
        constexpr auto n2 = Std::tuple_element<DIM_3RD, Shape>::type::value;
        constexpr auto n3 = Std::tuple_element<DIM_4TH, Shape>::type::value;
        constexpr auto n4 = Std::tuple_element<DIM_5TH, Shape>::type::value;
        return n1 * n2 * n3 * n4;
    }
    if constexpr (shapeSize == 4) {
        constexpr auto n1 = Std::tuple_element<DIM_2ND, Shape>::type::value;
        constexpr auto n2 = Std::tuple_element<DIM_3RD, Shape>::type::value;
        constexpr auto n3 = Std::tuple_element<DIM_4TH, Shape>::type::value;
        return n1 * n2 * n3;
    }
    if constexpr (shapeSize == 3) {
        constexpr auto n1 = Std::tuple_element<DIM_2ND, Shape>::type::value;
        constexpr auto n2 = Std::tuple_element<DIM_3RD, Shape>::type::value;
        return n1 * n2;
    }
    return 1;
}

template <size_t shapeSize, typename Shape>
__aicore__ inline constexpr size_t GetOutterAxisMergeResult() {
    if constexpr (shapeSize == 5) {
        constexpr auto n0 = Std::tuple_element<DIM_1ST, Shape>::type::value;
        constexpr auto n1 = Std::tuple_element<DIM_2ND, Shape>::type::value;
        constexpr auto n2 = Std::tuple_element<DIM_3RD, Shape>::type::value;
        constexpr auto n3 = Std::tuple_element<DIM_4TH, Shape>::type::value;
        return n0 * n1 * n2 * n3;
    }
    if constexpr (shapeSize == 4) {
        constexpr auto n0 = Std::tuple_element<DIM_1ST, Shape>::type::value;
        constexpr auto n1 = Std::tuple_element<DIM_2ND, Shape>::type::value;
        constexpr auto n2 = Std::tuple_element<DIM_3RD, Shape>::type::value;
        return n0 * n1 * n2;
    }
    if constexpr (shapeSize == 3) {
        constexpr auto n0 = Std::tuple_element<DIM_1ST, Shape>::type::value;
        constexpr auto n1 = Std::tuple_element<DIM_2ND, Shape>::type::value;
        return n0 * n1;
    }
    if constexpr (shapeSize == 2) {
        return Std::tuple_element<DIM_1ST, Shape>::type::value;
    }
    return 1;
}

template <typename T0>
__aicore__ inline constexpr bool JudgeValidShapeEqualTileShape() {
    using ShapeValueType = typename Std::tuple_element<0, typename T0::Shape>::type;
    if constexpr (Std::IsIntegralConstantV<ShapeValueType> == true) {
        constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;
        if constexpr (shapeSize == 1 || shapeSize == 2) {
            return true;
        }
        constexpr auto outterStride = GetOuterStride<typename T0::Stride>();
        constexpr auto nonFirstAxis = GetNonFirstAxisMergeResult<shapeSize, typename T0::Shape>();
        if constexpr (outterStride == nonFirstAxis) {
            return true;
        }
        return false;
    }
    return false;
}

template <typename T0>
__aicore__ constexpr bool IsConstContinous() {
    return JudgeValidShapeEqualTileShape<T0>();
}

template <typename T0, typename T1, typename... Args>
__aicore__ constexpr bool IsConstContinous() {
    if constexpr (!JudgeValidShapeEqualTileShape<T0>()) {
        return false;
    }
    return IsConstContinous<T1, Args...>();
}
} // namespace TileOp

// common shape
using Shape1Dim = TileOp::Shape<size_t>;
using Shape2Dim = TileOp::Shape<size_t, size_t>;
using Shape3Dim = TileOp::Shape<size_t, size_t, size_t>;
using Shape4Dim = TileOp::Shape<size_t, size_t, size_t, size_t>;
using Shape5Dim = TileOp::Shape<size_t, size_t, size_t, size_t, size_t>;

// common stride
using Stride1Dim = TileOp::Stride<size_t>;
using Stride2Dim = TileOp::Stride<size_t, size_t>;
using Stride3Dim = TileOp::Stride<size_t, size_t, size_t>;
using Stride4Dim = TileOp::Stride<size_t, size_t, size_t, size_t>;
using Stride5Dim = TileOp::Stride<size_t, size_t, size_t, size_t, size_t>;

// common coord
using Coord1Dim = TileOp::Coord<size_t>;
using Coord2Dim = TileOp::Coord<size_t, size_t>;
using Coord3Dim = TileOp::Coord<size_t, size_t, size_t>;
using Coord4Dim = TileOp::Coord<size_t, size_t, size_t, size_t>;
using Coord5Dim = TileOp::Coord<size_t, size_t, size_t, size_t, size_t>;

// common dynamic layouts
using DynLayout1Dim = TileOp::Layout<Shape1Dim, Stride1Dim, TileOp::TileShape<size_t>>;
using DynLayout2Dim = TileOp::Layout<Shape2Dim, Stride2Dim, TileOp::TileShape<size_t, size_t>>;
using DynLayout3Dim = TileOp::Layout<Shape3Dim, Stride3Dim, TileOp::TileShape<size_t, size_t, size_t>>;
using DynLayout4Dim = TileOp::Layout<Shape4Dim, Stride4Dim, TileOp::TileShape<size_t, size_t, size_t, size_t>>;
using DynLayout5Dim = TileOp::Layout<Shape5Dim, Stride5Dim, TileOp::TileShape<size_t, size_t, size_t, size_t, size_t>>;

// common Local layouts
template <size_t TileH, size_t TileW>
using LocalLayout2Dim = TileOp::Layout<Shape2Dim, TileOp::Stride<Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t TileD, size_t TileH, size_t TileW>
using LocalLayout3Dim = TileOp::Layout<Shape3Dim, TileOp::Stride<Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t TileN, size_t TileD, size_t TileH, size_t TileW>
using LocalLayout4Dim = TileOp::Layout<Shape4Dim,
    TileOp::Stride<Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t TileS, size_t TileN, size_t TileD, size_t TileH, size_t TileW>
using LocalLayout5Dim = TileOp::Layout<Shape5Dim,
    TileOp::Stride<Std::Int<TileN * TileD * TileH * TileW>, Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>,
        Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileS>, Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

// common static layouts
template <size_t H, size_t W, size_t TileH, size_t TileW>
using StaticLayout2Dim = TileOp::Layout<TileOp::Shape<Std::Int<H>, Std::Int<W>>,
    TileOp::Stride<Std::Int<TileW>, Std::Int<1>>, TileOp::TileShape<Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t D, size_t H, size_t W, size_t TileD, size_t TileH, size_t TileW>
using StaticLayout3Dim = TileOp::Layout<TileOp::Shape<Std::Int<D>, Std::Int<H>, Std::Int<W>>,
    TileOp::Stride<Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t N, size_t D, size_t H, size_t W, size_t TileN, size_t TileD, size_t TileH, size_t TileW>
using StaticLayout4Dim = TileOp::Layout<TileOp::Shape<Std::Int<N>, Std::Int<D>, Std::Int<H>, Std::Int<W>>,
    TileOp::Stride<Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>, Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;

template <size_t S, size_t N, size_t D, size_t H, size_t W, size_t TileS, size_t TileN, size_t TileD, size_t TileH,
    size_t TileW>
using StaticLayout5Dim = TileOp::Layout<TileOp::Shape<Std::Int<S>, Std::Int<N>, Std::Int<D>, Std::Int<H>, Std::Int<W>>,
    TileOp::Stride<Std::Int<TileN * TileD * TileH * TileW>, Std::Int<TileD * TileH * TileW>, Std::Int<TileH * TileW>,
        Std::Int<TileW>, Std::Int<1>>,
    TileOp::TileShape<Std::Int<TileS>, Std::Int<TileN>, Std::Int<TileD>, Std::Int<TileH>, Std::Int<TileW>>>;
#endif // TILEOP_UTILS_LAYOUT_H
