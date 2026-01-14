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
* \file iterator_concepts.hpp
* \brief
*/

#ifndef OSP_ITERATOR_CONCEPTS_HPP
#define OSP_ITERATOR_CONCEPTS_HPP

#include <iterator>
#include <type_traits>

namespace npu::tile_fwk {
namespace osp {
/**
 * @file iterator_concepts.hpp
 * @brief C++17 compatible concept checks (type traits) for iterators and ranges.
 *
 * This file provides type traits that emulate C++20 concepts for iterators and ranges.
 * These are used to ensure type safety and correct usage of templates within the library
 * while maintaining compatibility with C++17.
 */

/**
 * @brief Checks if a type is a forward iterator.
 *
 * This type trait checks if `T` satisfies the requirements of a forward iterator.
 * It verifies the existence of standard iterator typedefs and checks if the iterator category
 * is derived from `std::forward_iterator_tag`.
 *
 * @note Equivalent to C++20 `std::forward_iterator`.
 *
 * @tparam T The type to check.
 */
template <typename T, typename = void>
struct IsForwardIterator : std::false_type {};

template <typename T>
struct IsForwardIterator<T,
                         std::void_t<typename std::iterator_traits<T>::difference_type,
                                     typename std::iterator_traits<T>::value_type,
                                     typename std::iterator_traits<T>::pointer,
                                     typename std::iterator_traits<T>::reference,
                                     typename std::iterator_traits<T>::iterator_category>>
    : std::conjunction<std::is_base_of<std::forward_iterator_tag, typename std::iterator_traits<T>::iterator_category>> {};

template <typename T>
inline constexpr bool isForwardIteratorV = IsForwardIterator<T>::value;

/**
 * @brief Checks if a type is a range of forward iterators with a specific value type.
 *
 * This type trait checks if `T` is a range (provides `begin()` and `end()`) whose iterator
 * satisfies `is_forward_iterator` and whose value type matches `ValueType`.
 *
 * @note Equivalent to C++20 `std::ranges::forward_range` combined with a value type check.
 *
 * @tparam T The range type to check.
 * @tparam ValueType The expected value type of the range.
 */
template <typename T, typename ValueType, typename = void>
struct IsForwardRangeOf : std::false_type {};

template <typename T, typename ValueType>
struct IsForwardRangeOf<T, ValueType, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::conjunction<IsForwardIterator<decltype(std::begin(std::declval<T>()))>,
                       std::is_same<ValueType, typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>> {
};

template <typename T, typename ValueType>
inline constexpr bool isForwardRangeOfV = IsForwardRangeOf<T, ValueType>::value;

/**
 * @brief Checks if a type is an input iterator.
 *
 * This type trait checks if `T` satisfies the requirements of an input iterator.
 * It verifies the existence of standard iterator typedefs and checks if the iterator category
 * is derived from `std::input_iterator_tag`.
 *
 * @note Equivalent to C++20 `std::input_iterator`.
 *
 * @tparam T The type to check.
 */
template <typename T, typename = void>
struct IsInputIterator : std::false_type {};

template <typename T>
struct IsInputIterator<T,
                       std::void_t<typename std::iterator_traits<T>::difference_type,
                                   typename std::iterator_traits<T>::value_type,
                                   typename std::iterator_traits<T>::pointer,
                                   typename std::iterator_traits<T>::reference,
                                   typename std::iterator_traits<T>::iterator_category>>
    : std::conjunction<std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<T>::iterator_category>> {};

template <typename T>
inline constexpr bool isInputIteratorV = IsInputIterator<T>::value;

/**
 * @brief Checks if a type is a range of input iterators with a specific value type.
 *
 * This type trait checks if `T` is a range (provides `begin()` and `end()`) whose iterator
 * satisfies `is_input_iterator` and whose value type matches `ValueType`.
 *
 * @note Equivalent to C++20 `std::ranges::input_range` combined with a value type check.
 *
 * @tparam T The range type to check.
 * @tparam ValueType The expected value type of the range.
 */
template <typename T, typename ValueType, typename = void>
struct IsInputRangeOf : std::false_type {};

template <typename T, typename ValueType>
struct IsInputRangeOf<T, ValueType, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>>
    : std::conjunction<IsInputIterator<decltype(std::begin(std::declval<T>()))>,
                       std::is_same<ValueType, typename std::iterator_traits<decltype(std::begin(std::declval<T>()))>::value_type>> {
};

template <typename T, typename ValueType>
inline constexpr bool isInputRangeOfV = IsInputRangeOf<T, ValueType>::value;

}    // namespace osp
}    // namespace npu::tile_fwk
#endif // OSP_ITERATOR_CONCEPTS_HPP