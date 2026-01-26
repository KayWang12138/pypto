/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file divisors.hpp
 * \brief
 */

#ifndef PASS_OSP_MATH_H
#define PASS_OSP_MATH_H

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace npu::tile_fwk {
namespace osp {

template <typename IntegralType>
IntegralType IntSqrtFloor(IntegralType num) {
    static_assert(std::is_integral_v<IntegralType>);
    if (num <= 0) {
        return 0;
    }

    constexpr IntegralType numberTwo = 2;
    constexpr IntegralType numberFour = numberTwo * numberTwo;
    IntegralType sqrt = 1;
    IntegralType numCopy = num;
    while (numCopy >= numberFour) {
        sqrt *= numberTwo;
        numCopy /= numberFour;
    }
    IntegralType power2 = sqrt / numberTwo;
    while (power2 > 0) {
        IntegralType sum = sqrt + power2;
        if (sum * sum <= num) {
            sqrt = sum;
        }
        power2 /= numberTwo;
    }

    return sqrt;
}

template <typename IntegralType>
std::vector<IntegralType> DivisorsList(IntegralType num) {
    static_assert(std::is_integral_v<IntegralType>);
    if (num == 0) {
        return std::vector<IntegralType>({0});
    } else if (num < 0) {
        return std::vector<IntegralType>();
    }

    std::vector<IntegralType> divs;

    const IntegralType ub = IntSqrtFloor<IntegralType>(num);
    for (IntegralType div = 1; div <= ub; ++div) {
        if (num % div == 0) {
            divs.emplace_back(div);
        }
    }
    constexpr std::size_t numberTwo = 2U;
    const std::size_t beginIndx = divs.back() * divs.back() == num ? divs.size() - numberTwo : divs.size() - 1U;
    for (std::size_t indx = beginIndx; indx != std::numeric_limits<std::size_t>::max(); --indx) {
        divs.emplace_back(num / divs[indx]);
    }

    return divs;
}

}    // end namespace osp
}    // namespace npu::tile_fwk
#endif   // PASS_OSP_MATH_H