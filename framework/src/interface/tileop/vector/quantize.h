/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize.h
 * \brief Quantize tile operator for INT8 symmetric and asymmetric quantization
 *
 * Supports:
 * - INT8 symmetric quantization (INT8_SYM): FP32 -> INT8
 * - INT8 asymmetric quantization (INT8_ASYM): FP32 -> UINT8
 * - 2-4 dimensional input tensors
 * - axis = -1 (last axis) and axis = -2 (second to last axis)
 */

#ifndef TILEOP_TILE_OPERATOR_QUANTIZE__H
#define TILEOP_TILE_OPERATOR_QUANTIZE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

// Forward declaration of QuantType from PTO-ISA
namespace pto {
    enum class QuantType {
        MXFP8,
        INT8_SYM,
        INT8_ASYM
    };
}

// =============================================================================
// Helper function to convert negative axis to positive axis index
// For 2-4 dim tensors mapped to 5D: axis -1 -> dim 4, axis -2 -> dim 3
// =============================================================================
template <int axis, int shapeSize>
constexpr int ConvertAxisToPositive() {
    constexpr int expectSize = 5;
    // Map shapeSize to 5D: e.g., 2D -> dims 3,4; 3D -> dims 2,3,4; 4D -> dims 1,2,3,4
    constexpr int offset = expectSize - shapeSize;
    if constexpr (axis < 0) {
        // axis = -1 -> last dim (dim 4 in 5D), axis = -2 -> second last (dim 3 in 5D)
        return expectSize + axis;  // -1 -> 4, -2 -> 3
    } else {
        return offset + axis;
    }
}

// =============================================================================
// INT8 Symmetric Quantization Implementation
// =============================================================================

/**
 * @brief INT8 Symmetric Quantization - compute implementation for per-row quantization
 *
 * Formula: int8 = round(fp32 * scale)
 * - Supports axis = -1 (per-row quantization along last axis)
 */
template <int axis, typename LastUse, typename T0, typename T1, typename T2>
TILEOP void TQuantInt8SymImpl(T0 dst, T1 src, T2 scale) {
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;

    // Convert axis to positive index in 5D representation
    constexpr int axisIn5D = ConvertAxisToPositive<axis, srcShapeSize>();

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();

    // Get shapes
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    // Get strides
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

    auto scaleStride0 = scaleLayout.template GetStrideDim<0, expectSize>();
    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, 5>();
    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();

    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr auto scaleTypeSize = sizeof(typename T2::Type);

    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;

    if constexpr (axisIn5D == 4) {
        // axis = -1: Per-row quantization (scale shape: [rows, 1])
        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, scaleTileH, scaleTileW, pto::BLayout::ColMajor, -1, -1>;

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                        DstTileDefine dstTile(dstShape3, dstShape4);
                        SrcTileDefine srcTile(srcShape3, srcShape4);
                        ScaleTileDefine scaleTile(srcShape3, 1);

                        auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 +
                                        n2Index * dstStride2 + n3Index * dstStride3;
                        auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 +
                                        n2Index * srcStride2 + n3Index * srcStride3;
                        auto scaleOffset = n0Index * scaleStride0 + n1Index * scaleStride1 +
                                          n2Index * scaleStride2 + n3Index * scaleStride3;

                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                        pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * scaleTypeSize));

                        PTO_WITH_LAST_USE(pto::TQUANT<pto::QuantType::INT8_SYM>(dstTile, srcTile, scaleTile), n1, n2, n3);
                    }
                }
            }
        }
    } else if constexpr (axisIn5D == 3) {
        // axis = -2: Per-column quantization (scale shape: [1, cols])
        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, scaleTileW, pto::BLayout::RowMajor, -1, -1>;

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    DstTileDefine dstTile(dstShape3, dstShape4);
                    SrcTileDefine srcTile(dstShape3, dstShape4);
                    ScaleTileDefine scaleTile(1, dstShape4);

                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    auto scaleOffset = n0Index * scaleStride0 + n1Index * scaleStride1 + n2Index * scaleStride2;

                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * scaleTypeSize));

                    PTO_WITH_LAST_USE(pto::TQUANT<pto::QuantType::INT8_SYM>(dstTile, srcTile, scaleTile), n1, n2, n3);
                }
            }
        }
    }
}

// =============================================================================
// INT8 Asymmetric Quantization Implementation
// =============================================================================

/**
 * @brief INT8 Asymmetric Quantization - compute implementation for per-row quantization
 *
 * Formula: uint8 = round(fp32 * scale + offset)
 * - Supports axis = -1 (per-row quantization along last axis)
 */
template <int axis, typename LastUse, typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuantInt8AsymImpl(T0 dst, T1 src, T2 scale, T3 offset) {
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;

    // Convert axis to positive index in 5D representation
    constexpr int axisIn5D = ConvertAxisToPositive<axis, srcShapeSize>();

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();
    const auto offsetLayout = offset.GetLayout();

    // Get shapes
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();

    // Get strides
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();

    auto scaleStride0 = scaleLayout.template GetStrideDim<0, expectSize>();
    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<3, expectSize>();

    auto offsetStride0 = offsetLayout.template GetStrideDim<0, expectSize>();
    auto offsetStride1 = offsetLayout.template GetStrideDim<1, expectSize>();
    auto offsetStride2 = offsetLayout.template GetStrideDim<2, expectSize>();
    auto offsetStride3 = offsetLayout.template GetStrideDim<3, expectSize>();

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, 5>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, 5>();
    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, 5>();
    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, 5>();
    constexpr auto offsetTileH = TileOp::GetTensorTileShapeDim<T3, 3, 5>();
    constexpr auto offsetTileW = TileOp::GetTensorTileShapeDim<T3, 4, 5>();

    constexpr auto dstTypeSize = sizeof(typename T0::Type);
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    constexpr auto scaleTypeSize = sizeof(typename T2::Type);
    constexpr auto offsetTypeSize = sizeof(typename T3::Type);

    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;

    if constexpr (axisIn5D == 4) {
        // axis = -1: Per-row quantization (scale/offset shape: [rows, 1])
        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, scaleTileH, scaleTileW, pto::BLayout::ColMajor, -1, -1>;
        using OffsetTileDefine = pto::Tile<pto::TileType::Vec, typename T3::Type, offsetTileH, offsetTileW, pto::BLayout::ColMajor, -1, -1>;

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    for (LoopVar n3Index = 0; n3Index < dstShape3; ++n3Index) {
                        DstTileDefine dstTile(dstShape3, dstShape4);
                        SrcTileDefine srcTile(srcShape3, srcShape4);
                        ScaleTileDefine scaleTile(srcShape3, 1);
                        OffsetTileDefine offsetTile(srcShape3, 1);

                        auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 +
                                        n2Index * dstStride2 + n3Index * dstStride3;
                        auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 +
                                        n2Index * srcStride2 + n3Index * srcStride3;
                        auto scaleOffset = n0Index * scaleStride0 + n1Index * scaleStride1 +
                                          n2Index * scaleStride2 + n3Index * scaleStride3;
                        auto offsetOffset = n0Index * offsetStride0 + n1Index * offsetStride1 +
                                           n2Index * offsetStride2 + n3Index * offsetStride3;

                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                        pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * scaleTypeSize));
                        pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * offsetTypeSize));

                        PTO_WITH_LAST_USE(pto::TQUANT<pto::QuantType::INT8_ASYM>(dstTile, srcTile, scaleTile, &offsetTile), n1, n2, n3);
                    }
                }
            }
        }
    } else if constexpr (axisIn5D == 3) {
        // axis = -2: Per-column quantization (scale/offset shape: [1, cols])
        using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, scaleTileW, pto::BLayout::RowMajor, -1, -1>;
        using OffsetTileDefine = pto::Tile<pto::TileType::Vec, typename T3::Type, 1, offsetTileW, pto::BLayout::RowMajor, -1, -1>;

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    DstTileDefine dstTile(dstShape3, dstShape4);
                    SrcTileDefine srcTile(dstShape3, dstShape4);
                    ScaleTileDefine scaleTile(1, dstShape4);
                    OffsetTileDefine offsetTile(1, dstShape4);

                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    auto scaleOffset = n0Index * scaleStride0 + n1Index * scaleStride1 + n2Index * scaleStride2;
                    auto offsetOffset = n0Index * offsetStride0 + n1Index * offsetStride1 + n2Index * offsetStride2;

                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * dstTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * scaleTypeSize));
                    pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * offsetTypeSize));

                    PTO_WITH_LAST_USE(pto::TQUANT<pto::QuantType::INT8_ASYM>(dstTile, srcTile, scaleTile, &offsetTile), n1, n2, n3);
                }
            }
        }
    }
}

// =============================================================================
// Public API - INT8 Symmetric Quantization
// =============================================================================

#define OP_TILE_OP_TQUANT_INT8_SYM TQuantInt8Sym

/**
 * @brief INT8 Symmetric Quantization operator
 *
 * @tparam axis Quantization axis: -1 for per-row (last axis), -2 for per-column (second to last axis)
 * @tparam LastUse Last use configuration for optimization
 * @tparam T0 Destination tensor type (int8_t)
 * @tparam T1 Source tensor type (float)
 * @tparam T2 Scale tensor type (float)
 *
 * @param dst Output tensor (INT8)
 * @param src Input tensor (FP32)
 * @param scale Scale factor tensor
 *
 * Formula: dst = round(src * scale), clamped to [-128, 127]
 *
 * @note Supports 2-4 dimensional input tensors
 *       - axis = -1: Per-row quantization, scale shape should be [rows, 1]
 *       - axis = -2: Per-column quantization, scale shape should be [1, cols]
 */
template <int axis = -1, typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TQuantInt8Sym(T0 dst, T1 src, T2 scale) {
    TQuantInt8SymImpl<axis, LastUse>(dst, src, scale);
}

// =============================================================================
// Public API - INT8 Asymmetric Quantization
// =============================================================================

#define OP_TILE_OP_TQUANT_INT8_ASYM TQuantInt8Asym

/**
 * @brief INT8 Asymmetric Quantization operator
 *
 * @tparam axis Quantization axis: -1 for per-row (last axis), -2 for per-column (second to last axis)
 * @tparam LastUse Last use configuration for optimization
 * @tparam T0 Destination tensor type (uint8_t)
 * @tparam T1 Source tensor type (float)
 * @tparam T2 Scale tensor type (float)
 * @tparam T3 Offset tensor type (float)
 *
 * @param dst Output tensor (UINT8)
 * @param src Input tensor (FP32)
 * @param scale Scale factor tensor
 * @param offset Offset tensor
 *
 * Formula: dst = round(src * scale + offset), clamped to [0, 255]
 *
 * @note Supports 2-4 dimensional input tensors
 *       - axis = -1: Per-row quantization, scale/offset shape should be [rows, 1]
 *       - axis = -2: Per-column quantization, scale/offset shape should be [1, cols]
 */
template <int axis = -1, typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuantInt8Asym(T0 dst, T1 src, T2 scale, T3 offset) {
    TQuantInt8AsymImpl<axis, LastUse>(dst, src, scale, offset);
}

// =============================================================================
// Unified API - Generic TQuant
// =============================================================================

#define OP_TILE_OP_TQUANT TQuant

/**
 * @brief Generic quantization operator (unified interface)
 *
 * @tparam quantType Quantization type: INT8_SYM or INT8_ASYM
 * @tparam axis Quantization axis: -1 for per-row, -2 for per-column
 * @tparam LastUse Last use configuration
 * @tparam T0 Destination tensor type
 * @tparam T1 Source tensor type (float)
 * @tparam T2 Scale tensor type (float)
 * @tparam T3 Offset tensor type (float, only used for INT8_ASYM)
 *
 * For INT8_SYM:
 *   TQuant<pto::QuantType::INT8_SYM>(dst, src, scale)
 *   - dst type: int8_t
 *
 * For INT8_ASYM:
 *   TQuant<pto::QuantType::INT8_ASYM>(dst, src, scale, offset)
 *   - dst type: uint8_t
 */
template <pto::QuantType quantType, int axis = -1, typename LastUse = LastUse3Dim<0, 0, 0>,
          typename T0, typename T1, typename T2, typename T3 = T2>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 offset = T3()) {
    if constexpr (quantType == pto::QuantType::INT8_SYM) {
        (void)offset;  // Unused for symmetric quantization
        TQuantInt8Sym<axis, LastUse>(dst, src, scale);
    } else if constexpr (quantType == pto::QuantType::INT8_ASYM) {
        TQuantInt8Asym<axis, LastUse>(dst, src, scale, offset);
    }
}

// Overload without offset parameter for symmetric quantization
template <pto::QuantType quantType, int axis = -1, typename LastUse = LastUse3Dim<0, 0, 0>,
          typename T0, typename T1, typename T2>
TILEOP void TQuant(T0 dst, T1 src, T2 scale) {
    static_assert(quantType == pto::QuantType::INT8_SYM,
                  "TQuant without offset only supports INT8_SYM quantization type");
    TQuantInt8Sym<axis, LastUse>(dst, src, scale);
}

#endif // TILEOP_TILE_OPERATOR_QUANTIZE__H
