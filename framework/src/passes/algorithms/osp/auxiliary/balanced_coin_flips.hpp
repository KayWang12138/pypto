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
 * \file balanced_coin_flips.hpp
 * \brief
 */

#ifndef PASS_OSP_BALANCED_COIN_FLIPS_HPP
#define PASS_OSP_BALANCED_COIN_FLIPS_HPP

#include <random>
#include <vector>

namespace npu::tile_fwk {
namespace osp {

class BiasedRandom {
  public:
    bool GetFlip() {
        constexpr int genuineRandomSize = 3;
        constexpr int numberTwo = 2;
        int dieSize = numberTwo * genuineRandomSize + abs(trueBias_);
        std::uniform_int_distribution<int> distrib(0, dieSize - 1);
        int flip = distrib(gen_);
        if (trueBias_ >= 0) {
            if (flip >= genuineRandomSize) {
                trueBias_--;
                return true;
            } else {
                trueBias_++;
                return false;
            }
        } else {
            if (flip >= genuineRandomSize) {
                trueBias_++;
                return false;
            } else {
                trueBias_--;
                return true;
            }
        }
    }

    BiasedRandom(std::size_t seed = 1729U) : gen_(seed), trueBias_(0) {};

  private:
    /// @brief Random number generator
    std::mt19937 gen_;
    /// @brief Biases the coin towards true
    int trueBias_;
};

/// @brief Generates the Thue Morse Sequence
/// @param shift Starting point in the sequence
class ThueMorseSequence {
  public:
    ThueMorseSequence(long unsigned int shift = 0U) : next_(shift) { sequence_.emplace_back(false); }

    bool GetFlip() {
        for (long unsigned int i = sequence_.size(); i <= next_; i++) {
            constexpr long unsigned int numberTwo = 2U;
            if (i % numberTwo == 0) {
                sequence_.emplace_back(sequence_[i / numberTwo]);
            } else {
                sequence_.emplace_back(!sequence_[i / numberTwo]);
            }
        }
        return sequence_[next_++];
    }

  private:
    long unsigned int next_;
    std::vector<bool> sequence_;
};

}    // namespace osp
}    // namespace npu::tile_fwk
#endif    // PASS_OSP_BALANCED_COIN_FLIPS_HPP