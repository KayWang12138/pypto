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
 * \file derivation_tile_shape_new.h
 * \brief New implementation of tile shape derivation using linear index mapping algorithm
 *
 * This is a redesigned implementation that uses a simpler approach:
 * - Direct linear index mapping without intermediate aligned shape
 * - Factor decomposition for axis transformation tracking
 * - Boundary-based tile derivation
 */

#ifndef PASS_DERIVATION_TILE_SHAPE_NEW_H_
#define PASS_DERIVATION_TILE_SHAPE_NEW_H_

#include <vector>
#include <cstdint>

#include "passes/pass_interface/pass.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {

/**
 * @brief Linear index mapper for multi-dimensional arrays
 *
 * Provides utilities to convert between multi-dimensional indices and linear indices,
 * which is essential for tracking how tiles map across reshape operations.
 */
class LinearIndexMapper {
public:
    /**
     * @brief Convert multi-dimensional index to linear index
     * @param indices Multi-dimensional index
     * @param shape Shape of the tensor
     * @return Linear index in row-major order
     */
    static int64_t ToLinearIndex(const std::vector<int64_t>& indices, const std::vector<int64_t>& shape);

    /**
     * @brief Convert linear index to multi-dimensional index
     * @param linearIdx Linear index
     * @param shape Shape of the tensor
     * @return Multi-dimensional index
     */
    static std::vector<int64_t> ToMultiIndex(int64_t linearIdx, const std::vector<int64_t>& shape);

    /**
     * @brief Calculate strides from shape
     * @param shape Tensor shape
     * @return Stride vector
     */
    static std::vector<int64_t> CalculateStrides(const std::vector<int64_t>& shape);
};

/**
 * @brief Tile boundary tracker for reshape operations
 *
 * Tracks how tile boundaries transform when reshaping a tensor.
 * Uses linear index mapping to determine output tile boundaries.
 */
class TileBoundaryTracker {
public:
    /**
     * @brief Track tile boundaries through reshape
     * @param inShape Input tensor shape
     * @param outShape Output tensor shape
     * @param inTileShape Input tile shape
     * @return Output tile shape boundaries
     */
    static std::vector<int64_t> TrackBoundaries(
        const std::vector<int64_t>& inShape,
        const std::vector<int64_t>& outShape,
        const std::vector<int64_t>& inTileShape);

private:
    /**
     * @brief Find the smallest divisor of n that is >= threshold
     */
    static int64_t FindDivisor(int64_t n, int64_t threshold);

    /**
     * @brief Calculate GCD of two numbers
     */
    static int64_t GCD(int64_t a, int64_t b);
};

/**
 * @brief Axis factor decomposition for reshape analysis
 *
 * Decomposes tensor dimensions into prime factors to analyze
 * how axes split and merge during reshape operations.
 */
class AxisFactorAnalyzer {
public:
    /**
     * @brief Analyze how input axes map to output axes
     * @param inShape Input shape
     * @param outShape Output shape
     * @return Mapping from input axis index to output axis indices
     */
    static std::vector<std::vector<int64_t>> AnalyzeAxisMapping(
        const std::vector<int64_t>& inShape,
        const std::vector<int64_t>& outShape);

private:
    /**
     * @brief Factorize a number into prime factors
     */
    static std::vector<int64_t> Factorize(int64_t n);

    /**
     * @brief Calculate total size of shape
     */
    static int64_t GetShapeSize(const std::vector<int64_t>& shape);
};

/**
 * @brief New implementation of DerivationTileShape using linear index mapping
 *
 * Key differences from original implementation:
 * 1. No intermediate "aligned shape" - direct mapping from input to output
 * 2. Uses linear index to track tile boundaries
 * 3. Simpler algorithm with better readability
 * 4. More intuitive approach: reshape is just re-grouping of linear array
 *
 * Algorithm overview:
 * 1. Validate inputs (same as original)
 * 2. Use linear index mapping to track tile boundaries
 * 3. For each tile in input, find corresponding region in output
 * 4. Derive output tile shape from boundary analysis
 * 5. Verify memory layout consistency
 */
class DerivationTileShapeNew {
public:
    DerivationTileShapeNew() = default;
    ~DerivationTileShapeNew() = default;

    /**
     * @brief Derive output tile shape for reshape operation
     *
     * @param op Reshape operation (must be OP_RESHAPE)
     * @param inShape Input tensor shape
     * @param outShape Output tensor shape
     * @param inTileShape Input tensor tile shape
     * @param outTileShape [Output] Derived output tile shape
     * @return SUCCESS if derivation succeeds, WARNING otherwise
     *
     * Example:
     *   inShape = [8, 6], inTileShape = [2, 3]
     *   outShape = [2, 4, 6]
     *   => outTileShape = [1, 2, 3]
     *
     * The algorithm ensures that:
     * - Total elements remain the same
     * - Memory layout is preserved
     * - Tile boundaries are correctly mapped
     */
    Status DerivationReshapeTileShape(
        Operation *op,
        const Shape &inShape,
        const Shape &outShape,
        const std::vector<int64_t> &inTileShape,
        std::vector<int64_t> &outTileShape);

private:
    /**
     * @brief Validate input parameters
     */
    bool ValidateInputs(
        Operation *op,
        const Shape &inShape,
        const Shape &outShape,
        const std::vector<int64_t> &inTileShape);

    /**
     * @brief Core algorithm: derive output tile using linear index mapping
     */
    bool DeriveOutputTileByLinearMapping(
        const Shape &inShape,
        const Shape &outShape,
        const std::vector<int64_t> &inTileShape,
        std::vector<int64_t> &outTileShape);

    /**
     * @brief Verify that derived tile shape preserves memory layout
     */
    bool VerifyMemoryLayout(
        const Shape &inShape,
        const Shape &outShape,
        const std::vector<int64_t> &inTileShape,
        const std::vector<int64_t> &outTileShape);

    /**
     * @brief Check if a shape is valid (all dimensions > 0)
     */
    bool ValidShape(const std::vector<int64_t>& shape);
};

} // namespace tile_fwk
} // namespace npu

#endif // PASS_DERIVATION_TILE_SHAPE_NEW_H_
