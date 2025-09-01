/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType &coord, const ShapeType &shape, const StrideType &stride);

template <typename... Shapes>
using Shape = Std::tuple<Shapes...>;

template <typename... Strides>
using Stride = Std::tuple<Strides...>;

template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts &...t) {
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts &...t) {
    return {t...};
}

template <typename ShapeType, typename StrideType>
struct Layout : private Std::tuple<ShapeType, StrideType> {
    __aicore__ inline constexpr Layout(const ShapeType &shape = {}, const StrideType &stride = {})
        : Std::tuple<ShapeType, StrideType>(shape, stride) {
        static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "Shape or Stride is not tuple!");
    }

    __aicore__ inline constexpr decltype(auto) layout() { return *this; }

    __aicore__ inline constexpr decltype(auto) layout() const { return *this; }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape() {
        return GetValue<0, I...>(static_cast<Std::tuple<ShapeType, StrideType> &>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape() const {
        return GetValue<0, I...>(static_cast<const Std::tuple<ShapeType, StrideType> &>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride() {
        return GetValue<1, I...>(static_cast<Std::tuple<ShapeType, StrideType> &>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride() const {
        return GetValue<1, I...>(static_cast<const Std::tuple<ShapeType, StrideType> &>(*this));
    }

    template <typename CoordType>
    __aicore__ inline constexpr auto operator()(const CoordType &coord) const {
        return Crd2Idx(coord, GetShape(), GetStride());
    }

private:
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
};

template <typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto MakeLayout(const ShapeType &shape, const StrideType &stride) {
    return Layout<ShapeType, StrideType>(shape, stride);
}

template <typename T>
struct is_layout : Std::false_type {};

template <typename ShapeType, typename StrideType>
struct is_layout<Layout<ShapeType, StrideType>> : Std::true_type {};

template <typename T>
constexpr bool is_layout_v = is_layout<T>::value;

// common layouts
using Layout1Dim = Layout<Shape<int>, Stride<int>>;
using Layout2Dim = Layout<Shape<int, int>, Stride<int, int>>;
using Layout3Dim = Layout<Shape<int, int, int>, Stride<int, int, int>>;
using Layout4Dim = Layout<Shape<int, int, int, int>, Stride<int, int, int, int>>;
using Layout5Dim = Layout<Shape<int, int, int, int, int, int>, Stride<int, int, int, int, int>>;

#endif // TILEOP_UTILS_LAYOUT_H
