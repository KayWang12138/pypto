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
 * \file small_array.h
 * \brief
 */

#pragma once

#include <array>
#include <cstring>
#include <cassert>
#include "securec.h"

#include "runtime/utils/device_log.h"

template <typename T, int N>
class FixedArray {
public:
    inline size_t size() const { return size_; }

    inline void clear() { size_ = 0; }
    inline bool empty() { return size_ == 0; }

    inline T *back() { return &data_[size_]; }
    inline T *data() { return &data_.data(); }

    inline void push_back(T val) { data_[size_++] = val;}
    inline void CopyTo(T *ptr, int n, int offset = 0) const {
        memcpy_s(ptr, n * sizeof(T), data_.data() + offset, n * sizeof(T));
    }

    inline void resize(size_t size) {
        DEV_ASSERT(size <= N);
        size_ = size;
    }

    inline T &operator[](size_t index) { return data_[index]; }

    inline const T &operator[](size_t index) const { return data_[index]; }

private:
    size_t size_{0};
    std::array<T, N> data_;
};
