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
 * - axis = -1 (last axis) and axis = -2 (second to last axis)
 */

#ifndef TILEOP_TILE_OPERATOR_QUANTIZE__H
#define TILEOP_TILE_OPERATOR_QUANTIZE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

// =============================================================================
// INT8 Symmetric Quantization
// =============================================================================

#define OP_TILE_OP_TQUANT_INT8_SYM TQuantInt8Sym

/**
 * @brief INT8 Symmetric Quantization
 *
 * Formula: int8 = round(fp32 * scale), clamped to [-128, 127]
 *
 * @tparam axis Quantization axis: -1 for per-row (last axis), -2 for per-column (second to last axis)
 * @tparam LastUse Last use configuration (default: LastUse3Dim<0, 0, 0>)
 * @param dst Output tensor (INT8)
 * @param src Input tensor (FP32)
 * @param scale Scale factor tensor
 */
template <int axis = -1, typename LastUse = LastUse3Dim<0, 0, 0>, typename T0, typename T1, typename T2>
TILEOP void TQuantInt8Sym(T0 dst, T1 src, T2 scale) {
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();

    // Get shapes
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

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

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>();
    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();

    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;

    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;

    // Determine axis in 5D representation
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr int axisIn5D = (axis < 0) ? (expectSize + axis) : (expectSize - srcShapeSize + axis);

    if constexpr (axisIn5D == 4) {
        // axis = -1: Per-row quantization (scale shape: [rows, 1])
        using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, scaleTileH, scaleTileW, pto::BLayout::ColMajor, -1, -1>;

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

                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                        pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));

                        PTO_WITH_LAST_USE(pto::TQUANT<pto::QuantType::INT8_SYM>(dstTile, srcTile, scaleTile), n1, n2, n3);
                    }
                }
            }
        }
    } else if constexpr (axisIn5D == 3) {
        // axis = -2: Per-column quantization (scale shape: [1, cols])
        using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, 1, scaleTileW, pto::BLayout::RowMajor, -1, -1>;

        for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    DstTileDefine dstTile(dstShape3, dstShape4);
                    SrcTileDefine srcTile(dstShape3, dstShape4);
                    ScaleTileDefine scaleTile(1, dstShape4);

                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    auto scaleOffset = n0Index * scaleStride0 + n1Index * scaleStride1 + n2Index * scaleStride2;

                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));

                    PTO_WITH_LAST_USE(pto::TQUANT<pto::QuantType::INT8_SYM>(dstTile, srcTile, scaleTile), n1, n2, n3);
                }
            }
        }
    }
}

// =============================================================================
// INT8 Asymmetric Quantization
// =============================================================================

#define OP_TILE_OP_TQUANT_INT8_ASYM TQuantInt8Asym

/**
 * @brief INT8 Asymmetric Quantization
 *
 * Formula: uint8 = round(fp32 * scale + offset), clamped to [0, 255]
 *
 * @tparam axis Quantization axis: -1 for per-row (last axis), -2 for per-column (second to last axis)
 * @tparam LastUse Last use configuration (default: LastUse3Dim<0, 0, 0>)
 * @param dst Output tensor (UINT8)
 * @param src Input tensor (FP32)
 * @param scale Scale factor tensor
 * @param offset Offset tensor
 */
template <int axis = -1, typename LastUse = LastUse3Dim<0, 0, 0>,
          typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuantInt8Asym(T0 dst, T1 src, T2 scale, T3 offset) {
    constexpr size_t expectSize = 5;

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

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

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

    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>();
    constexpr auto scaleTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();
    constexpr auto offsetTileH = TileOp::GetTensorTileShapeDim<T3, 3, expectSize>();
    constexpr auto offsetTileW = TileOp::GetTensorTileShapeDim<T3, 4, expectSize>();

    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;
    using OffsetDtype = typename T3::Type;

    constexpr auto n1 = Std::tuple_element<DIM_1ST, LastUse>::type::value;
    constexpr auto n2 = Std::tuple_element<DIM_2ND, LastUse>::type::value;
    constexpr auto n3 = Std::tuple_element<DIM_3RD, LastUse>::type::value;

    // Determine axis in 5D representation
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr int axisIn5D = (axis < 0) ? (expectSize + axis) : (expectSize - srcShapeSize + axis);

    if constexpr (axisIn5D == 4) {
        // axis = -1: Per-row quantization (scale/offset shape: [rows, 1])
        using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, scaleTileH, scaleTileW, pto::BLayout::ColMajor, -1, -1>;
        using OffsetTileDefine = pto::Tile<pto::TileType::Vec, OffsetDtype, offsetTileH, offsetTileW, pto::BLayout::ColMajor, -1, -1>;

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

                        pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                        pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                        pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                        pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));

                        PTO_WITH_LAST_USE(pto::TQUANT<pto::QuantType::INT8_ASYM>(dstTile, srcTile, scaleTile, &offsetTile), n1, n2, n3);
                    }
                }
            }
        }
    } else if constexpr (axisIn5D == 3) {
        // axis = -2: Per-column quantization (scale/offset shape: [1, cols])
        using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
        using ScaleTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, 1, scaleTileW, pto::BLayout::RowMajor, -1, -1>;
        using OffsetTileDefine = pto::Tile<pto::TileType::Vec, OffsetDtype, 1, offsetTileW, pto::BLayout::RowMajor, -1, -1>;

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

                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                    pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                    pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));

                    PTO_WITH_LAST_USE(pto::TQUANT<pto::QuantType::INT8_ASYM>(dstTile, srcTile, scaleTile, &offsetTile), n1, n2, n3);
                }
            }
        }
    }
}

// =============================================================================
// Unified TQuant API (Generic interface)
// =============================================================================

#define OP_TILE_OP_TQUANT TQuant

/**
 * @brief Generic quantization operator (unified interface)
 *
 * @tparam quantType Quantization type: INT8_SYM or INT8_ASYM
 * @tparam axis Quantization axis: -1 for per-row, -2 for per-column
 * @tparam LastUse Last use configuration
 *
 * Usage:
 *   TQuant<pto::QuantType::INT8_SYM>(dst, src, scale)
 *   TQuant<pto::QuantType::INT8_ASYM>(dst, src, scale, offset)
 */

// INT8_SYM: 3 parameters
template <pto::QuantType quantType, int axis = -1, typename LastUse = LastUse3Dim<0, 0, 0>,
          typename T0, typename T1, typename T2>
TILEOP void TQuant(T0 dst, T1 src, T2 scale) {
    static_assert(quantType == pto::QuantType::INT8_SYM,
                  "TQuant with 3 parameters only supports INT8_SYM. For INT8_ASYM, provide offset parameter.");
    TQuantInt8Sym<axis, LastUse>(dst, src, scale);
}

// INT8_ASYM: 4 parameters
template <pto::QuantType quantType, int axis = -1, typename LastUse = LastUse3Dim<0, 0, 0>,
          typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 offset) {
    static_assert(quantType == pto::QuantType::INT8_ASYM,
                  "TQuant with 4 parameters only supports INT8_ASYM. For INT8_SYM, use 3 parameters.");
    TQuantInt8Asym<axis, LastUse>(dst, src, scale, offset);
}

#endif // TILEOP_TILE_OPERATOR_QUANTIZE__H
