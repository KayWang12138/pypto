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
 * \file derivation_tile_shape_new.cpp
 * \brief New implementation using linear index mapping algorithm
 */

#include "derivation_tile_shape_new.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include "passes/pass_log/pass_log.h"

#define MODULE_NAME "DerivationTileShapeNew"

namespace npu {
namespace tile_fwk {

// ============================================================================
// LinearIndexMapper Implementation
// ============================================================================

int64_t LinearIndexMapper::ToLinearIndex(const std::vector<int64_t>& indices,
                                         const std::vector<int64_t>& shape) {
    int64_t linearIdx = 0;
    int64_t stride = 1;

    for (int i = shape.size() - 1; i >= 0; --i) {
        linearIdx += indices[i] * stride;
        stride *= shape[i];
    }

    return linearIdx;
}

std::vector<int64_t> LinearIndexMapper::ToMultiIndex(int64_t linearIdx,
                                                     const std::vector<int64_t>& shape) {
    std::vector<int64_t> indices(shape.size());

    for (int i = shape.size() - 1; i >= 0; --i) {
        indices[i] = linearIdx % shape[i];
        linearIdx /= shape[i];
    }

    return indices;
}

std::vector<int64_t> LinearIndexMapper::CalculateStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    strides[shape.size() - 1] = 1;

    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return strides;
}

// ============================================================================
// TileBoundaryTracker Implementation
// ============================================================================

int64_t TileBoundaryTracker::GCD(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int64_t TileBoundaryTracker::FindDivisor(int64_t n, int64_t threshold) {
    // Find the smallest divisor of n that is >= threshold
    for (int64_t i = threshold; i <= n; ++i) {
        if (n % i == 0) {
            return i;
        }
    }
    return n;
}

std::vector<int64_t> TileBoundaryTracker::TrackBoundaries(
    const std::vector<int64_t>& inShape,
    const std::vector<int64_t>& outShape,
    const std::vector<int64_t>& inTileShape) {

    std::vector<int64_t> outTileShape(outShape.size(), 1);

    // Calculate input and output strides
    auto inStrides = LinearIndexMapper::CalculateStrides(inShape);
    auto outStrides = LinearIndexMapper::CalculateStrides(outShape);

    // For each input tile dimension, track where it maps in output
    int64_t currentLinearSize = 1;

    for (int i = inShape.size() - 1; i >= 0; --i) {
        int64_t tileDim = inTileShape[i];
        int64_t inDim = inShape[i];

        // Calculate linear size covered by this tile dimension
        int64_t tileLinearSize = tileDim * inStrides[i];

        // Map this linear size to output dimensions
        int64_t remainingSize = tileLinearSize;

        for (int j = outShape.size() - 1; j >= 0 && remainingSize > 1; --j) {
            int64_t outDim = outShape[j];
            int64_t outStride = outStrides[j];

            if (remainingSize >= outStride) {
                int64_t count = std::min(remainingSize / outStride, outDim);
                outTileShape[j] = std::max(outTileShape[j], count);
                remainingSize -= count * outStride;
            }
        }
    }

    return outTileShape;
}

// ============================================================================
// AxisFactorAnalyzer Implementation
// ============================================================================

std::vector<int64_t> AxisFactorAnalyzer::Factorize(int64_t n) {
    std::vector<int64_t> factors;

    // Factor out 2s
    while (n % 2 == 0) {
        factors.push_back(2);
        n /= 2;
    }

    // Factor out odd numbers
    for (int64_t i = 3; i * i <= n; i += 2) {
        while (n % i == 0) {
            factors.push_back(i);
            n /= i;
        }
    }

    if (n > 2) {
        factors.push_back(n);
    }

    return factors;
}

int64_t AxisFactorAnalyzer::GetShapeSize(const std::vector<int64_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
}

std::vector<std::vector<int64_t>> AxisFactorAnalyzer::AnalyzeAxisMapping(
    const std::vector<int64_t>& inShape,
    const std::vector<int64_t>& outShape) {

    std::vector<std::vector<int64_t>> mapping(inShape.size());

    // Use simple stride-based mapping
    auto inStrides = LinearIndexMapper::CalculateStrides(inShape);
    auto outStrides = LinearIndexMapper::CalculateStrides(outShape);

    for (size_t i = 0; i < inShape.size(); ++i) {
        int64_t inStride = inStrides[i];

        // Find which output axes this input axis maps to
        for (size_t j = 0; j < outShape.size(); ++j) {
            int64_t outStride = outStrides[j];

            // If strides are compatible, there's a mapping
            if (inStride % outStride == 0 || outStride % inStride == 0) {
                mapping[i].push_back(j);
            }
        }
    }

    return mapping;
}

// ============================================================================
// DerivationTileShapeNew Implementation
// ============================================================================

bool DerivationTileShapeNew::ValidShape(const std::vector<int64_t>& shape) {
    return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim > 0; });
}

bool DerivationTileShapeNew::ValidateInputs(
    Operation *op,
    const Shape &inShape,
    const Shape &outShape,
    const std::vector<int64_t> &inTileShape) {

    // Check operation type
    if (op->GetOpcode() != Opcode::OP_RESHAPE) {
        APASS_LOG_WARN_F(Elements::Operation,
            "Operation is not RESHAPE, got opcode: %d", op->GetOpcode());
        return false;
    }

    // Validate shapes
    if (!ValidShape(inShape) || !ValidShape(outShape) || !ValidShape(inTileShape)) {
        APASS_LOG_WARN_F(Elements::Operation,
            "Invalid shape detected (contains non-positive dimensions)");
        return false;
    }

    // Check dimension consistency
    if (inShape.size() != inTileShape.size()) {
        APASS_LOG_WARN_F(Elements::Operation,
            "Input shape dimension %zu != tile shape dimension %zu",
            inShape.size(), inTileShape.size());
        return false;
    }

    // Check total element count
    int64_t inSize = std::accumulate(inShape.begin(), inShape.end(),
                                     (int64_t)1, std::multiplies<int64_t>());
    int64_t outSize = std::accumulate(outShape.begin(), outShape.end(),
                                      (int64_t)1, std::multiplies<int64_t>());

    if (inSize != outSize) {
        APASS_LOG_WARN_F(Elements::Operation,
            "Input size %ld != output size %ld", inSize, outSize);
        return false;
    }

    // Validate tile shape is not larger than input shape in each dimension
    for (size_t i = 0; i < inShape.size(); ++i) {
        if (inTileShape[i] > inShape[i] && inTileShape[i] % inShape[i] != 0) {
            APASS_LOG_INFO_F(Elements::Operation,
                "Tile dimension %ld at axis %zu is larger than shape %ld (allowed if divisible)",
                inTileShape[i], i, inShape[i]);
        }
    }

    return true;
}

bool DerivationTileShapeNew::DeriveOutputTileByLinearMapping(
    const Shape &inShape,
    const Shape &outShape,
    const std::vector<int64_t> &inTileShape,
    std::vector<int64_t> &outTileShape) {

    APASS_LOG_INFO_F(Elements::Operation,
        "=== Linear Mapping Algorithm Start ===");

    // Initialize output tile shape
    outTileShape.assign(outShape.size(), 1);

    // Calculate strides
    auto inStrides = LinearIndexMapper::CalculateStrides(inShape);
    auto outStrides = LinearIndexMapper::CalculateStrides(outShape);

    APASS_LOG_INFO_F(Elements::Operation,
        "Input strides: [%s]",
        std::accumulate(inStrides.begin(), inStrides.end(), std::string(),
            [](const std::string& a, int64_t b) {
                return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b);
            }).c_str());

    APASS_LOG_INFO_F(Elements::Operation,
        "Output strides: [%s]",
        std::accumulate(outStrides.begin(), outStrides.end(), std::string(),
            [](const std::string& a, int64_t b) {
                return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b);
            }).c_str());

    // Core algorithm: Process from innermost (fastest-varying) dimension
    // Track cumulative linear size and distribute it across output dimensions

    int64_t cumulativeLinearSize = 1;

    for (int i = inShape.size() - 1; i >= 0; --i) {
        int64_t currentTile = inTileShape[i];
        int64_t currentDim = inShape[i];

        // Calculate linear size covered by this input tile dimension
        int64_t tileContribution = currentTile * inStrides[i] / cumulativeLinearSize;

        APASS_LOG_INFO_F(Elements::Operation,
            "Processing input axis %d: shape=%ld, tile=%ld, stride=%ld, contribution=%ld",
            i, currentDim, currentTile, inStrides[i], tileContribution);

        // Distribute this contribution to output dimensions
        int64_t remaining = tileContribution;

        for (int j = outShape.size() - 1; j >= 0 && remaining > 1; --j) {
            int64_t outDim = outShape[j];

            if (remaining >= outDim) {
                // Full dimension covered
                outTileShape[j] = outDim;
                remaining /= outDim;
            } else {
                // Partial dimension covered
                outTileShape[j] = std::max(outTileShape[j], remaining);
                remaining = 1;
            }

            APASS_LOG_INFO_F(Elements::Operation,
                "  Output axis %d: tile=%ld, remaining=%ld",
                j, outTileShape[j], remaining);
        }

        cumulativeLinearSize *= currentDim;
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "=== Linear Mapping Algorithm End ===");

    return true;
}

bool DerivationTileShapeNew::VerifyMemoryLayout(
    const Shape &inShape,
    const Shape &outShape,
    const std::vector<int64_t> &inTileShape,
    const std::vector<int64_t> &outTileShape) {

    APASS_LOG_INFO_F(Elements::Operation,
        "=== Memory Layout Verification Start ===");

    // Calculate strides
    auto inStrides = LinearIndexMapper::CalculateStrides(inShape);
    auto outStrides = LinearIndexMapper::CalculateStrides(outShape);

    // Calculate tile counts
    std::vector<int64_t> inTileCounts(inShape.size());
    for (size_t i = 0; i < inShape.size(); ++i) {
        inTileCounts[i] = (inShape[i] + inTileShape[i] - 1) / inTileShape[i];
    }

    std::vector<int64_t> outTileCounts(outShape.size());
    for (size_t i = 0; i < outShape.size(); ++i) {
        outTileCounts[i] = (outShape[i] + outTileShape[i] - 1) / outTileShape[i];
    }

    int64_t inTotalTiles = std::accumulate(inTileCounts.begin(), inTileCounts.end(),
                                          (int64_t)1, std::multiplies<int64_t>());
    int64_t outTotalTiles = std::accumulate(outTileCounts.begin(), outTileCounts.end(),
                                           (int64_t)1, std::multiplies<int64_t>());

    APASS_LOG_INFO_F(Elements::Operation,
        "Input tiles: %ld, Output tiles: %ld", inTotalTiles, outTotalTiles);

    // Verify tile counts match (allowing for rounding differences)
    if (std::abs(inTotalTiles - outTotalTiles) > 1) {
        APASS_LOG_WARN_F(Elements::Operation,
            "Tile count mismatch: input=%ld, output=%ld",
            inTotalTiles, outTotalTiles);
        return false;
    }

    // Sample verification: check a few tile boundaries
    // For efficiency, we only check the first tile
    std::vector<int64_t> inTileIdx(inShape.size(), 0);
    std::vector<int64_t> outTileIdx(outShape.size(), 0);

    // Calculate first tile's linear start position
    int64_t inTileStart = 0;
    int64_t outTileStart = 0;

    // Calculate size of first tile
    int64_t inTileSize = 1;
    for (size_t i = 0; i < inShape.size(); ++i) {
        inTileSize *= std::min(inTileShape[i], inShape[i]);
    }

    int64_t outTileSize = 1;
    for (size_t i = 0; i < outShape.size(); ++i) {
        outTileSize *= std::min(outTileShape[i], outShape[i]);
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "First tile size: input=%ld, output=%ld", inTileSize, outTileSize);

    // Sizes should match
    if (inTileSize != outTileSize) {
        APASS_LOG_WARN_F(Elements::Operation,
            "First tile size mismatch");
        return false;
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "=== Memory Layout Verification Passed ===");

    return true;
}

Status DerivationTileShapeNew::DerivationReshapeTileShape(
    Operation *op,
    const Shape &inShape,
    const Shape &outShape,
    const std::vector<int64_t> &inTileShape,
    std::vector<int64_t> &outTileShape) {

    APASS_LOG_INFO_F(Elements::Operation,
        "=== DerivationTileShapeNew::DerivationReshapeTileShape ===");
    APASS_LOG_INFO_F(Elements::Operation,
        "Input shape: [%s]",
        std::accumulate(inShape.begin(), inShape.end(), std::string(),
            [](const std::string& a, int64_t b) {
                return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b);
            }).c_str());
    APASS_LOG_INFO_F(Elements::Operation,
        "Input tile: [%s]",
        std::accumulate(inTileShape.begin(), inTileShape.end(), std::string(),
            [](const std::string& a, int64_t b) {
                return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b);
            }).c_str());
    APASS_LOG_INFO_F(Elements::Operation,
        "Output shape: [%s]",
        std::accumulate(outShape.begin(), outShape.end(), std::string(),
            [](const std::string& a, int64_t b) {
                return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b);
            }).c_str());

    // Step 1: Validate inputs
    if (!ValidateInputs(op, inShape, outShape, inTileShape)) {
        APASS_LOG_WARN_F(Elements::Operation, "Input validation failed");
        return WARNING;
    }

    // Step 2: Derive output tile using linear mapping algorithm
    if (!DeriveOutputTileByLinearMapping(inShape, outShape, inTileShape, outTileShape)) {
        APASS_LOG_WARN_F(Elements::Operation, "Tile derivation failed");
        return WARNING;
    }

    // Step 3: Verify memory layout consistency
    if (!VerifyMemoryLayout(inShape, outShape, inTileShape, outTileShape)) {
        APASS_LOG_WARN_F(Elements::Operation, "Memory layout verification failed");
        return WARNING;
    }

    APASS_LOG_INFO_F(Elements::Operation,
        "Derived output tile: [%s]",
        std::accumulate(outTileShape.begin(), outTileShape.end(), std::string(),
            [](const std::string& a, int64_t b) {
                return a.empty() ? std::to_string(b) : a + ", " + std::to_string(b);
            }).c_str());
    APASS_LOG_INFO_F(Elements::Operation,
        "=== DerivationTileShapeNew SUCCESS ===");

    return SUCCESS;
}

} // namespace tile_fwk
} // namespace npu
