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
 * \file tensor_offset.h
 * \brief
 */

#pragma once

#include <vector>
#include <set>
#include <string>
#include <memory>
#include <unordered_set>
#include <functional>

#include "tilefwk/error.h"
#include "symbolic_scalar.h"

namespace npu::tile_fwk {

    /* replace std::vector<int> using TensorOffset for further unify concrete offset and symbolic offset */
    class TensorOffset {
    public:
        TensorOffset(const std::vector<int> &offset, const std::vector<SymbolicScalar> &dynOffset) : offset_(offset), dynOffset_(dynOffset) {}

        const std::vector<int> &GetOffset() const { return offset_; }
        const std::vector<SymbolicScalar> &GetDynOffset() const { return dynOffset_; }
        template<typename Tret, typename Tlhs, typename Trhs>
        static std::vector<Tret> AddRaw(const std::vector<Tlhs> &lhs, const std::vector<Trhs> &rhs) {
            ASSERT(lhs.size() == rhs.size()) << "lhs:" << lhs.size() <<"  rhs:"<< rhs.size();
            std::vector<Tret> ret(lhs.size());
            for (size_t k = 0; k < lhs.size(); k++) {
                ret[k] = lhs[k] + rhs[k];
            }
            return ret;
        }

        static std::vector<int> Add(const std::vector<int> &lhs, const std::vector<int> &rhs) {
            return AddRaw<int, int, int>(lhs, rhs);
        }
        static std::vector<SymbolicScalar> Add(const std::vector<SymbolicScalar> &lhs, const std::vector<int> &rhs) {
            return AddRaw<SymbolicScalar, SymbolicScalar, int>(lhs, rhs);
        }
        static std::vector<SymbolicScalar> Add(const std::vector<int> &lhs, const std::vector<SymbolicScalar> &rhs) {
            return AddRaw<SymbolicScalar, int, SymbolicScalar>(lhs, rhs);
        }
        static std::vector<SymbolicScalar> Add(const std::vector<SymbolicScalar> &lhs, const std::vector<SymbolicScalar> &rhs) {
            return AddRaw<SymbolicScalar, SymbolicScalar, SymbolicScalar>(lhs, rhs);
        }

        static std::vector<int> Zero(const std::vector<int> &off) {
            return std::vector<int>(off.size(), 0);
        }
        static bool IsZero(const std::vector<int> &off) {
            return Zero(off) == off;
        }

        static
        std::pair<std::vector<int>, std::vector<SymbolicScalar>> Add(
                const std::vector<int> &lhs, const std::vector<SymbolicScalar> &lhsDyn,
                const std::vector<int> &rhs, const std::vector<SymbolicScalar> &rhsDyn) {
            ASSERT(lhs.size() == rhs.size());
            std::vector<int> ret = Add(lhs, rhs);

            std::vector<SymbolicScalar> retDyn;

            if (lhsDyn.size() != 0 && rhsDyn.size() != 0) {
                retDyn = Add(lhsDyn, rhsDyn);
            } else if (lhsDyn.size() != 0) {
                retDyn = Add(lhsDyn, rhs);
            } else if (rhsDyn.size() != 0) {
                retDyn = Add(lhs, rhsDyn);
            }
            return std::make_pair(ret, retDyn);
        }

        static
        std::vector<int> Sub(const std::vector<int> &lhs, const std::vector<int> &rhs) {
            ASSERT(lhs.size() == rhs.size());
            std::vector<int> result(lhs.size());
            std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                [](int a, int b) { return a - b; });
            return result;
        }

        static
        std::vector<SymbolicScalar> Sub(const std::vector<SymbolicScalar> &lhs, const std::vector<int> &rhs) {
            if (lhs.size() == 0) {
                return {};
            }
            ASSERT(lhs.size() == rhs.size());
            std::vector<SymbolicScalar> result(lhs.size());
            std::transform(lhs.begin(), lhs.end(), rhs.begin(), result.begin(),
                [](const SymbolicScalar &a, int b) { return a - b; });
            return result;
        }
    public:
        const std::vector<int> &offset_;
        const std::vector<SymbolicScalar> &dynOffset_;
    };

}
