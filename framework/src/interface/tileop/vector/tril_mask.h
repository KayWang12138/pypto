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
 * \file tril_mask.h
 * \brief Lower triangular mask generation for causal attention
 */

#ifndef TILEOP_TILE_OPERATOR_TRIL_MASK__H
#define TILEOP_TILE_OPERATOR_TRIL_MASK__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

/**
 * @brief Generate lower triangular mask matrix
 *
 * For each position (i, j) in the output matrix:
 *   mask[i][j] = 1 if (k_idx + j <= q_idx + i), else 0
 *
 * This is equivalent to:
 *   mask[i][j] = 1 if (j <= i + offset), where offset = q_idx - k_idx
 *
 * @tparam DstTensor The destination tensor type
 * @param dst The output tensor
 * @param qIdx The starting query index
 * @param kIdx The starting key index
 */
template <typename DstTensor>
TILEOP void TTriLMask(DstTensor dst, int64_t qIdx, int64_t kIdx) {
    using DstDtype = typename DstTensor::Type;

    constexpr size_t expectSize = 2;
    const auto dstLayout = dst.GetLayout();

    auto rows = dstLayout.template GetShapeDim<0, expectSize>();
    auto cols = dstLayout.template GetShapeDim<1, expectSize>();
    auto stride = dstLayout.template GetStrideDim<0, expectSize>();

    // Early return for empty tensors
    if (rows == 0 || cols == 0) {
        return;
    }

    // Pre-compute offset: offset = q_idx - k_idx
    // mask[i][j] = 1 when j <= i + offset
    int64_t offset = qIdx - kIdx;

    __ubuf__ DstDtype* dstPtr = (__ubuf__ DstDtype*)dst.GetAddr();

    // Fill the mask matrix row by row
    for (size_t i = 0; i < rows; ++i) {
        // Calculate threshold for this row: j <= i + offset
        int64_t threshold = static_cast<int64_t>(i) + offset;

        for (size_t j = 0; j < cols; ++j) {
            DstDtype value = (static_cast<int64_t>(j) <= threshold)
                            ? static_cast<DstDtype>(1) : static_cast<DstDtype>(0);
            dstPtr[i * stride + j] = value;
        }
    }

    // Pipeline synchronization
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

/**
 * @brief Dynamic version of TTriLMask for runtime-determined shapes
 *
 * @tparam T The element type (typically uint8_t for bool mask)
 * @param dst Pointer to output buffer
 * @param qIdx The starting query index
 * @param kIdx The starting key index
 * @param rows Number of rows in the output matrix
 * @param cols Number of columns in the output matrix
 * @param stride Row stride in the output buffer
 */
template <typename T>
TILEOP void TTriLMaskDyn(__ubuf__ T* dst, int64_t qIdx, int64_t kIdx,
                         int64_t rows, int64_t cols, int64_t stride) {
    // Pre-compute offset
    int64_t offset = qIdx - kIdx;

    for (int64_t i = 0; i < rows; ++i) {
        int64_t threshold = i + offset;

        for (int64_t j = 0; j < cols; ++j) {
            T value = (j <= threshold) ? static_cast<T>(1) : static_cast<T>(0);
            dst[i * stride + j] = value;
        }
    }

    // Pipeline synchronization
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

/**
 * @brief Static template version with compile-time shape parameters
 *
 * @tparam T Element type
 * @tparam ROWS Number of rows (compile-time constant)
 * @tparam COLS Number of columns (compile-time constant)
 */
template <typename T, int64_t ROWS, int64_t COLS>
TILEOP void TTriLMaskStatic(__ubuf__ T* dst, int64_t qIdx, int64_t kIdx) {
    int64_t offset = qIdx - kIdx;

    for (int64_t i = 0; i < ROWS; ++i) {
        int64_t threshold = i + offset;

        for (int64_t j = 0; j < COLS; ++j) {
            T value = (j <= threshold) ? static_cast<T>(1) : static_cast<T>(0);
            dst[i * COLS + j] = value;
        }
    }

    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#endif // TILEOP_TILE_OPERATOR_TRIL_MASK__H
