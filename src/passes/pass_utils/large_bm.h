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
 * \file large_bm.h
 * \brief
 */

#pragma once

#include <vector>
#include <memory>

namespace npu::tile_fwk {
class LargeBitmap {
public:
    explicit LargeBitmap(const size_t &size);

    ~LargeBitmap() = default;

    bool operator==(const LargeBitmap &anotherBm) const;

    bool operator!=(const LargeBitmap &anotherBm) const;

    // set all vector to specific value
    void SetValues(const uint64_t &value);

    // Get the value on position index
    bool GetBit(const size_t &index) const;

    // Set the value on position index to 1
    void SetBit(const size_t &index);

    // Combine two bitmap with the following rule.
    // If one bit of either one of the two bitmaps is 1,
    // the result of final bitmap is 1.
    void Or(const LargeBitmap &anotherBm);

    // Combine two bitmap with the following rule.
    // If one bit of either one of the two bitmaps is 0,
    // the result of final bitmap is 0.
    void And(const LargeBitmap &anotherBm);

    void ClearBit(size_t bitIdx);

    void ResizeBits(size_t newSize);
private:
    // Number of element in vector bits
    size_t size_;

    std::vector<uint64_t> bits_;
};
}
