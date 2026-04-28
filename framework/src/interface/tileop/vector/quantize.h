/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quantize.h
 * \brief INT8 对称/非对称量化 Tile 算子
 *
 * - INT8_SYM:  FP32 -> INT8,  范围 [-128, 127]
 * - INT8_ASYM: FP32 -> UINT8, 范围 [0, 255]
 *
 * 只支持逐行量化 (axis=-1)，逐列量化在 Operation 层通过 Transpose 实现
 */

#ifndef TILEOP_TILE_OPERATOR_QUANTIZE__H
#define TILEOP_TILE_OPERATOR_QUANTIZE__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include <type_traits>

/// Tile 内存对齐字节数
constexpr size_t TILE_ALIGNMENT_BYTES = 32;

/// 向上取整到对齐边界
#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))

// =============================================================================
// INT8 对称量化
// =============================================================================
#define OP_TILE_OP_TQUANT_INT8_SYM TQuantInt8Sym

/**
 * @brief INT8 对称量化 (逐行)
 * @param dst   输出 INT8 张量, 形状 [..., H, W]
 * @param src   输入 FP32 张量, 形状 [..., H, W]
 * @param scale 缩放因子, 形状 [..., H]
 */
template <typename T0, typename T1, typename T2>
TILEOP void TQuantInt8Sym(T0 dst, T1 src, T2 scale) {
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();

    // 获取形状
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

    // 获取步长
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<3, expectSize>();

    // 获取 Tile 形状并计算对齐
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileH, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 数据类型
    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;

    // 定义 Tile 类型
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, paddedCol_dst,
                                    pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, paddedCol_src,
                                    pto::BLayout::RowMajor, -1, -1>;
    using ParaTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1,
                                    pto::BLayout::ColMajor, -1, -1>;

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ParaTileDefine scaleTile(srcShape3, 1);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                // scaleOffset should need to be shifted
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));

                pto::TQUANT<pto::QuantType::INT8_SYM>(dstTile, srcTile, scaleTile);
            }
        }
    }
}

// =============================================================================
// INT8 非对称量化
// =============================================================================
#define OP_TILE_OP_TQUANT_INT8_ASYM TQuantInt8Asym

/**
 * @brief INT8 非对称量化 (逐行)
 * @param dst    输出 UINT8 张量, 形状 [..., H, W]
 * @param src    输入 FP32 张量, 形状 [..., H, W]
 * @param scale  缩放因子, 形状 [..., H]
 * @param offset 零点偏移, 形状 [..., H]
 */
template <typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuantInt8Asym(T0 dst, T1 src, T2 scale, T3 offset) {
    constexpr size_t expectSize = 5;

    const auto dstLayout = dst.GetLayout();
    const auto srcLayout = src.GetLayout();
    const auto scaleLayout = scale.GetLayout();
    const auto offsetLayout = offset.GetLayout();

    // 获取形状
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

    // 获取步长
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    auto scaleStride1 = scaleLayout.template GetStrideDim<1, expectSize>();
    auto scaleStride2 = scaleLayout.template GetStrideDim<2, expectSize>();
    auto scaleStride3 = scaleLayout.template GetStrideDim<3, expectSize>();

    auto offsetStride1 = offsetLayout.template GetStrideDim<1, expectSize>();
    auto offsetStride2 = offsetLayout.template GetStrideDim<2, expectSize>();
    auto offsetStride3 = offsetLayout.template GetStrideDim<3, expectSize>();

    // 获取 Tile 形状
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr int paddedCol_dst = PTO_CEIL(dstTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(int8_t)));

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, 3, expectSize>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, 4, expectSize>();
    constexpr int paddedCol_src = PTO_CEIL(srcTileW, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    constexpr auto scaleTileH = TileOp::GetTensorTileShapeDim<T2, 3, expectSize>();
    constexpr int paddedRow_scale = PTO_CEIL(scaleTileH, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));
    constexpr auto offsetTileH = TileOp::GetTensorTileShapeDim<T3, 3, expectSize>();
    constexpr int paddedRow_offset = PTO_CEIL(offsetTileH, static_cast<int>(TILE_ALIGNMENT_BYTES / sizeof(float)));

    // 数据类型
    using DstDtype = typename T0::Type;
    using SrcDtype = typename T1::Type;
    using ScaleDtype = typename T2::Type;
    using OffsetDtype = typename T3::Type;

    // 定义 Tile 类型
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DstDtype, dstTileH, dstTileW,
                                    pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, srcTileH, srcTileW,
                                    pto::BLayout::RowMajor, -1, -1>;
    using ParaTileDefine = pto::Tile<pto::TileType::Vec, ScaleDtype, paddedRow_scale, 1,
                                    pto::BLayout::ColMajor, -1, -1>;

    // 遍历所有 Tile
    for (LoopVar n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < dstShape2; ++n2Index) {
                DstTileDefine dstTile(dstShape3, dstShape4);
                SrcTileDefine srcTile(srcShape3, srcShape4);
                ParaTileDefine scaleTile(srcShape3, 1);
                ParaTileDefine offsetTile(srcShape3, 1);

                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                // scaleOffset & offsetOffset should need to be shifted
                auto scaleOffset = n0Index * scaleStride1 + n1Index * scaleStride2 + n2Index * scaleStride3;
                auto offsetOffset = n0Index * offsetStride1 + n1Index * offsetStride2 + n2Index * offsetStride3;

                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstDtype)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcDtype)));
                pto::TASSIGN(scaleTile, (uint64_t)(scale.GetAddr() + scaleOffset * sizeof(ScaleDtype)));
                pto::TASSIGN(offsetTile, (uint64_t)(offset.GetAddr() + offsetOffset * sizeof(OffsetDtype)));

                pto::TQUANT<pto::QuantType::INT8_ASYM>(dstTile, srcTile, scaleTile, &offsetTile);
            }
        }
    }
}

// =============================================================================
// 统一量化接口
// =============================================================================
#define OP_TILE_OP_TQUANT TQuant

/**
 * @brief 统一量化接口
 * @tparam quantType INT8_SYM 或 INT8_ASYM
 */
template <pto::QuantType quantType, typename T0, typename T1, typename T2>
TILEOP void TQuant(T0 dst, T1 src, T2 scale) {
    static_assert(quantType == pto::QuantType::INT8_SYM,
                  "TQuant with 3 parameters only supports INT8_SYM. "
                  "TQuant only supports INT8_SYM(3 parameters) and INT8_ASYM(4 parameters).");
    TQuantInt8Sym(dst, src, scale);
}

template <pto::QuantType quantType, typename T0, typename T1, typename T2, typename T3>
TILEOP void TQuant(T0 dst, T1 src, T2 scale, T3 offset) {
    static_assert(quantType == pto::QuantType::INT8_ASYM,
                  "TQuant with 4 parameters only supports INT8_ASYM."
                  "TQuant only supports INT8_SYM(3 parameters) and INT8_ASYM(4 parameters).");
    TQuantInt8Asym(dst, src, scale, offset);
}

// =============================================================================
// MX 量化
// =============================================================================
#define OP_TILE_OP_QUANT_MX TQuantMX
constexpr int kDequantScaleRoundingModeRoundUp = 0;
constexpr int kDequantScaleRoundingModeRoundDown = 1;
constexpr int kQuantMXPerformanceModeOn = 1;

template <typename T, typename Layout>
__aicore__ inline size_t GetQuantMXPerformanceGroupedOffset(
    const Layout& layout, LoopVar n0Index, LoopVar n1Index, LoopVar n2Index)
{
    (void)n0Index;
    constexpr auto srcRank = Std::tuple_size<typename T::Shape>::value;
    static_assert(srcRank >= 1 && srcRank <= 4, "TQuantMX only supports 1D to 4D input.");
    if constexpr (srcRank <= 2) {
        return 0;
    } else if constexpr (srcRank == 3) {
        return n2Index * layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    } else {
        return n1Index * layout.template GetStrideDim<DIM_3RD, MAX_DIMS>() +
               n2Index * layout.template GetStrideDim<DIM_4TH, MAX_DIMS>();
    }
}

template <
    int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
    typename T2, typename T3, typename T4>
TILEOP void TQuantMXGeneral(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    constexpr int kMxQuantGroupSize = 32;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto expStride0 = expLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto expStride1 = expLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto expStride2 = expLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto expTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;

    auto dstTile = PtoTile<T0>(dst);
    auto maxTile = PtoTile<T2>(maxScratch);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    using SrcTileType = typename decltype(srcTile)::Type;
    using SrcPadTileType = pto::Tile<
        SrcTileType::Loc, typename SrcTileType::DType, SrcTileType::Rows, SrcTileType::Cols, SrcTileType::BFractal,
        SrcTileType::ValidRow, SrcTileType::ValidCol, SrcTileType::SFractal, SrcTileType::SFractalSize,
        pto::PadValue::Zero, SrcTileType::Compact>;
    ExpByteTile expByteTile(
        expLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(), expLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    (void)maxLayout;
    (void)scalingLayout;
    (void)srcLayout;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = n0Index * expStride0 + n1Index * expStride1 + n2Index * expStride2;
                auto srcTileAddr =
                    (uint64_t)(src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                maxTile.Assign(maxScratch, tileOffsets);
                scalingTile.Assign(scalingScratch, tileOffsets);
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                if (srcTile.Data().GetValidCol() % kMxQuantGroupSize != 0) {
                    if constexpr (T4::IsStaticLayout()) {
                        SrcPadTileType srcPadTile;
                        pto::TASSIGN(srcPadTile, srcTileAddr);
                        pto::TFILLPAD_INPLACE(srcPadTile, srcTile.Data());
                    } else {
                        SrcPadTileType srcPadTile(srcTile.Data().GetValidRow(), srcTile.Data().GetValidCol());
                        pto::TASSIGN(srcPadTile, srcTileAddr);
                        pto::TFILLPAD_INPLACE(srcPadTile, srcTile.Data());
                    }
                }
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown) {
                    pto::TQUANT<pto::QuantType::MXFP8>(
                        dstTile.Data(), srcTile.Data(), &expByteTile, &maxTile.Data(), &scalingTile.Data());
                } else {
                    static_assert(
                        DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown,
                        "TQuantMX only supports ROUND_DOWN (OCP standard) mode currently.");
                }
            }
        }
    }
}

template <
    int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, typename T0, typename T1,
    typename T2, typename T3, typename T4>
TILEOP void TQuantMXPerformance(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    (void)AXIS;
    const auto dstLayout = dst.GetLayout();
    const auto expLayout = exp.GetLayout();
    const auto maxLayout = maxScratch.GetLayout();
    const auto scalingLayout = scalingScratch.GetLayout();
    const auto srcLayout = src.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto scalingTile = PtoTile<T3>(scalingScratch);
    auto srcTile = PtoTile<T4>(src);
    constexpr auto expTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto expTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto maxTileH = TileOp::GetTensorTileShapeDim<T2, DIM_4TH, MAX_DIMS>();
    constexpr auto maxTileW = TileOp::GetTensorTileShapeDim<T2, DIM_5TH, MAX_DIMS>();
    using ExpByteTile = pto::Tile<pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;
    using MaxDtype = std::conditional_t<std::is_same_v<typename T2::Type, bool>, uint8_t, typename T2::Type>;
    using MaxTile = pto::Tile<pto::TileType::Vec, MaxDtype, maxTileH, maxTileW, pto::BLayout::RowMajor, -1, -1>;
    ExpByteTile expByteTile(
        expLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(), expLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());
    MaxTile maxTile(
        maxLayout.template GetShapeDim<DIM_4TH, MAX_DIMS>(), maxLayout.template GetShapeDim<DIM_5TH, MAX_DIMS>());

    (void)scalingLayout;
    (void)srcLayout;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                auto expTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(expLayout, n0Index, n1Index, n2Index);
                auto maxTileOffset = GetQuantMXPerformanceGroupedOffset<T4>(maxLayout, n0Index, n1Index, n2Index);
                auto srcTileAddr =
                    (uint64_t)(src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                scalingTile.Assign(scalingScratch, tileOffsets);
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                pto::TASSIGN(maxTile, (uint64_t)(maxScratch.GetAddr() + maxTileOffset * sizeof(typename T2::Type)));
                if constexpr (DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown) {
                    pto::TQUANT<pto::QuantType::MXFP8>(
                        dstTile.Data(), srcTile.Data(), &expByteTile, &maxTile, &scalingTile.Data());
                } else {
                    static_assert(
                        DEQUANT_SCALE_ROUNDING_MODE == kDequantScaleRoundingModeRoundDown,
                        "TQuantMX only supports ROUND_DOWN (OCP standard) mode currently.");
                }
            }
        }
    }
}

template <
    int DEQUANT_SCALE_ROUNDING_MODE = kDequantScaleRoundingModeRoundDown, int AXIS = -1, int PERFORMANCE_MODE = 0,
    typename T0, typename T1, typename T2, typename T3, typename T4>
TILEOP void TQuantMX(T0 dst, T1 exp, T2 maxScratch, T3 scalingScratch, T4 src)
{
    if constexpr (PERFORMANCE_MODE == kQuantMXPerformanceModeOn) {
        TQuantMXPerformance<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, exp, maxScratch, scalingScratch, src);
    } else {
        TQuantMXGeneral<DEQUANT_SCALE_ROUNDING_MODE, AXIS>(dst, exp, maxScratch, scalingScratch, src);
    }
}

#endif // TILEOP_TILE_OPERATOR_QUANTIZE__H

// /**
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// */

// #ifndef TQUANT_HPP
// #define TQUANT_HPP

// #include <pto/common/constants.hpp>
// #include <pto/common/utils.hpp>
// #include <pto/npu/a5/common.hpp>
// #include <pto/npu/a5/utils.hpp>
// #include "TReshape.hpp"
// #include <type_traits>

// namespace pto {

// enum class QuantType { MXFP8, INT8_SYM, INT8_ASYM };

// // Helper alias: creates a 1D flat tile from a 2D tile's total element count.
// template <typename TileData>
// using FlatTile1D = Tile<
//     TileType::Vec, typename TileData::DType, 1, TileData::Rows * TileData::Cols, BLayout::RowMajor, -1, -1,
//     SLayout::NoneBox, 512, PadValue::Zero>;

// PTO_INTERNAL void AbsReduceMax_Naive(
//     __ubuf__ float* srcPtr, __ubuf__ float* maxPtr, unsigned total_elements_count, unsigned vl_count,
//     unsigned elementsPerRepeat, MaskReg& preg_lower32, MaskReg& preg_upper32)
// {
//     RegTensor<float> vreg_b32;
//     vector_s32 vreg_zero;
//     vbr(vreg_zero, 0);
//     for (uint16_t i = 0; i < (uint16_t)vl_count; ++i) {
//         uint32_t offset = i * elementsPerRepeat;
//         uint32_t remaining = (total_elements_count > offset) ? (total_elements_count - offset) : 0;
//         if (remaining > elementsPerRepeat)
//             remaining = elementsPerRepeat;
//         MaskReg preg = CreatePredicate<float>(remaining);
//         RegTensor<float> vreg_max_0, vreg_max_1;
//         vlds(vreg_b32, srcPtr, offset, NORM);
//         vabs(vreg_b32, vreg_b32, preg);
//         vsel((vector_s32&)vreg_b32, (vector_s32&)vreg_b32, vreg_zero, preg);
//         vcmax(vreg_max_0, vreg_b32, preg_lower32);
//         vcmax(vreg_max_1, vreg_b32, preg_upper32);
//         vsts(vreg_max_0, maxPtr, 2 * i, ONEPT_B32, preg);
//         vsts(vreg_max_1, maxPtr + 1, 2 * i, ONEPT_B32, preg);
//     }
// }

// // Assumption: input total size is a multiple of 256 elements
// PTO_INTERNAL void AbsReduceMax_f32_opt(
//     __ubuf__ float* srcPtr, __ubuf__ float* maxPtr, unsigned vl_count, unsigned elementsPerRepeat,
//     unsigned total_elements_count)
// {
//     vector_f32 vreg_in_1, vreg_in_2, vreg_in_3, vreg_in_4, vreg_max_0, vreg_max_1, vreg_max;
//     vector_f32 vreg_dintlv_1, vreg_dintlv_2, vreg_dintlv_3, vreg_dintlv_4, vreg_gp_max;
//     vector_f32 vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_dintlv_out_3, vreg_dintlv_out_4;
//     vector_align ureg_max;
//     uint32_t total_count = total_elements_count;
//     MaskReg preg_lower8 = pset_b32(PAT_VL8);
//     static constexpr auto distValue =
//         std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
//     for (uint16_t i = 0; i < (uint16_t)vl_count / 4; ++i) {
//         MaskReg preg_vl0 = CreatePredicate<float>(total_count);
//         MaskReg preg_vl1 = CreatePredicate<float>(total_count);
//         MaskReg preg_vl2 = CreatePredicate<float>(total_count);
//         MaskReg preg_vl3 = CreatePredicate<float>(total_count);
//         vlds(vreg_in_1, vreg_in_2, srcPtr, i * 4 * elementsPerRepeat, DINTLV_B32);
//         vlds(vreg_in_3, vreg_in_4, srcPtr + 128, i * 4 * elementsPerRepeat, DINTLV_B32);
//         vabs(vreg_in_1, vreg_in_1, preg_vl0);
//         vabs(vreg_in_3, vreg_in_3, preg_vl2);
//         vdintlv(vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_in_1, vreg_in_3);
//         vabs(vreg_in_2, vreg_in_2, preg_vl1);
//         vabs(vreg_in_4, vreg_in_4, preg_vl3);
//         vdintlv(vreg_dintlv_out_3, vreg_dintlv_out_4, vreg_in_2, vreg_in_4);
//         vmax(vreg_max_0, vreg_dintlv_out_1, vreg_dintlv_out_2, preg_vl0);
//         vmax(vreg_max_1, vreg_dintlv_out_3, vreg_dintlv_out_4, preg_vl1);
//         vmax(vreg_max, vreg_max_0, vreg_max_1, preg_vl0);
//         vcgmax(vreg_gp_max, vreg_max, preg_vl0);
//         vsts(vreg_gp_max, maxPtr, i * 8, distValue, preg_lower8);
//     }
// }

// // Assumption: input total size is a multiple of 2K elements
// PTO_INTERNAL void AbsReduceMax_f32_opt_largesizes(
//     __ubuf__ float* srcPtr, __ubuf__ float* maxPtr, unsigned vl_count, unsigned elementsPerRepeat,
//     unsigned total_elements_count)
// {
//     vector_f32 vreg_in_1, vreg_in_2, vreg_in_3, vreg_in_4, vreg_max;
//     vector_f32 vreg_gp_max, vreg_dintlv_out_1, vreg_dintlv_out_2;
//     vector_align ureg_max;
//     uint32_t total_count = total_elements_count;
//     MaskReg preg_ALL_B32 = pset_b32(PAT_ALL);
//     static constexpr auto distValue =
//         std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
//     for (uint16_t i = 0; i < (uint16_t)vl_count / 32; ++i) {
//         for (uint16_t j = 0; j < 8; ++j) { // handling 4 VLs per loop, each VL is 256 B (64 fp32)
//             MaskReg preg_vl0 = CreatePredicate<float>(total_count);
//             MaskReg preg_vl1 = CreatePredicate<float>(total_count);
//             MaskReg preg_vl2 = CreatePredicate<float>(total_count);
//             MaskReg preg_vl3 = CreatePredicate<float>(total_count);
//             vlds(vreg_in_1, vreg_in_2, srcPtr, (i * 32 + j * 4) * elementsPerRepeat, DINTLV_B32);
//             vabs(vreg_in_1, vreg_in_1, preg_vl0);
//             vabs(vreg_in_2, vreg_in_2, preg_vl1);
//             vlds(
//                 vreg_in_3, vreg_in_4, srcPtr + 2 * elementsPerRepeat, (i * 32 + j * 4) * elementsPerRepeat, DINTLV_B32);
//             vabs(vreg_in_3, vreg_in_3, preg_vl2);
//             vabs(vreg_in_4, vreg_in_4, preg_vl3);
//             vmax(vreg_in_1, vreg_in_1, vreg_in_2, preg_vl0);
//             vmax(vreg_in_3, vreg_in_3, vreg_in_4, preg_vl2);
//             vdintlv(vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_in_1, vreg_in_3);
//             vmax(vreg_max, vreg_dintlv_out_1, vreg_dintlv_out_2, preg_vl0);
//             vcgmax(vreg_gp_max, vreg_max, preg_ALL_B32);
//             vstus(ureg_max, 8, vreg_gp_max, maxPtr + 64 * i + 8 * j);
//         }
//         vstas(ureg_max, maxPtr + 64 * i, 0);
//     }
// }

// // Reduce one 256-element DINTLV_B16 window to 8 per-block BF16 abs raw maxima.
// // This follows dynamic_mx_quant_tail_axis_fp8: FP16 is first converted to BF16,
// // then both FP16/BF16 paths reduce the BF16 abs bit pattern.
// template <typename T>
// PTO_INTERNAL void AbsReduceMax_b16_DintlvWindow(
//     __ubuf__ T* srcPtr, uint32_t offset, uint32_t remaining, RegTensor<T>& vb16_max)
// {
//     static_assert(
//         std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//         "AbsReduceMax_b16_DintlvWindow: T must be bfloat16_t or half");
//     constexpr uint16_t kBf16AbsMask = 0x7FFF;
//     constexpr uint16_t kFp16ExpMask = 0x7C00;
//     constexpr uint16_t kFp16MantissaMask = 0x03FF;
//     constexpr uint16_t kFp16InfBits = 0x7C00;
//     constexpr uint16_t kBf16InfBits = 0x7F80;
//     constexpr uint16_t kBf16NanBits = 0x7FC0;
//     RegTensor<T> vb16_in_1, vb16_in_2;
//     RegTensor<uint16_t> vu16_abs_1, vu16_abs_2, vu16_bf16_abs_mask;
//     uint32_t even_count = (remaining + 1) / 2;
//     uint32_t odd_count = remaining / 2;
//     MaskReg preg_vl0 = CreatePredicate<T>(even_count);
//     MaskReg preg_vl1 = CreatePredicate<T>(odd_count);
//     vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16);

//     vbr(vu16_bf16_abs_mask, kBf16AbsMask);
//     if constexpr (std::is_same<T, half>::value) {
//         vector_bf16 vb16_bf16_1, vb16_bf16_2;
//         RegTensor<uint16_t> vu16_fp16_abs_mask, vu16_fp16_exp_mask, vu16_fp16_mantissa_mask;
//         RegTensor<uint16_t> vu16_fp16_exp_1, vu16_fp16_exp_2;
//         RegTensor<uint16_t> vu16_fp16_mantissa_1, vu16_fp16_mantissa_2, vu16_bf16_inf, vu16_bf16_nan;
//         vector_bool preg_special_1, preg_special_2, preg_nan_1, preg_nan_2, preg_inf_1, preg_inf_2;

//         // Preserve fp16 Inf/NaN before abs/max, since NaN propagation requires a non-saturating
//         // f16->bf16 cast, while the following FP8 quantization path requires saturating mode.
//         vbr(vu16_fp16_abs_mask, kBf16AbsMask);
//         vbr(vu16_fp16_exp_mask, kFp16ExpMask);
//         vbr(vu16_fp16_mantissa_mask, kFp16MantissaMask);
//         vbr(vu16_bf16_inf, kBf16InfBits);
//         vbr(vu16_bf16_nan, kBf16NanBits);
//         vand(vu16_abs_1, (vector_u16&)vb16_in_1, vu16_fp16_abs_mask, preg_vl0, MODE_ZEROING);
//         vand(vu16_abs_2, (vector_u16&)vb16_in_2, vu16_fp16_abs_mask, preg_vl1, MODE_ZEROING);
//         vand(vu16_fp16_exp_1, vu16_abs_1, vu16_fp16_exp_mask, preg_vl0, MODE_ZEROING);
//         vand(vu16_fp16_exp_2, vu16_abs_2, vu16_fp16_exp_mask, preg_vl1, MODE_ZEROING);
//         vand(vu16_fp16_mantissa_1, vu16_abs_1, vu16_fp16_mantissa_mask, preg_vl0, MODE_ZEROING);
//         vand(vu16_fp16_mantissa_2, vu16_abs_2, vu16_fp16_mantissa_mask, preg_vl1, MODE_ZEROING);
//         vcmps_eq(preg_special_1, vu16_fp16_exp_1, kFp16ExpMask, preg_vl0);
//         vcmps_eq(preg_special_2, vu16_fp16_exp_2, kFp16ExpMask, preg_vl1);
//         vcmps_ne(preg_nan_1, vu16_fp16_mantissa_1, 0, preg_special_1);
//         vcmps_ne(preg_nan_2, vu16_fp16_mantissa_2, 0, preg_special_2);
//         vcmps_eq(preg_inf_1, vu16_abs_1, kFp16InfBits, preg_vl0);
//         vcmps_eq(preg_inf_2, vu16_abs_2, kFp16InfBits, preg_vl1);
//         vcvt(vb16_bf16_1, vb16_in_1, preg_vl0, ROUND_Z);
//         vcvt(vb16_bf16_2, vb16_in_2, preg_vl1, ROUND_Z);
//         vsel((vector_u16&)vb16_bf16_1, vu16_bf16_inf, (vector_u16&)vb16_bf16_1, preg_inf_1);
//         vsel((vector_u16&)vb16_bf16_2, vu16_bf16_inf, (vector_u16&)vb16_bf16_2, preg_inf_2);
//         vsel((vector_u16&)vb16_bf16_1, vu16_bf16_nan, (vector_u16&)vb16_bf16_1, preg_nan_1);
//         vsel((vector_u16&)vb16_bf16_2, vu16_bf16_nan, (vector_u16&)vb16_bf16_2, preg_nan_2);
//         vand(vu16_abs_1, (vector_u16&)vb16_bf16_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
//         vand(vu16_abs_2, (vector_u16&)vb16_bf16_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
//     } else {
//         vand(vu16_abs_1, (vector_u16&)vb16_in_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
//         vand(vu16_abs_2, (vector_u16&)vb16_in_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
//     }

//     vmax(vu16_abs_1, vu16_abs_1, vu16_abs_2, preg_vl0, MODE_ZEROING);
//     vcgmax((vector_u16&)vb16_max, vu16_abs_1, preg_vl0, MODE_ZEROING);
// }

// // See npu_skills/pto-isa/instructions/tquant-mxfp8.md for the full rationale
// // on why we branch on loop_num and how the vstus/vstas continuation works.
// template <typename T>
// PTO_INTERNAL void AbsReduceMax_b16_ND(
//     __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned vl_count, unsigned total_elem_count)
// {
//     constexpr uint32_t elements_per_dintlv = 2 * REPEAT_BYTE / sizeof(T); // 256 b16 per DINTLV
//     constexpr uint32_t grps_per_dintlv = elements_per_dintlv / 32;        // 8 BF16 abs maxima per iter
//     constexpr uint32_t blks_per_vl = REPEAT_BYTE / BLOCK_SIZE;
//     static constexpr auto distValue =
//         std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
//     uint16_t loop_num = CeilDivision(vl_count, 2);
//     RegTensor<T> vb16_max;

//     // loop_num==1: single window writes only 16 B of BF16 abs maxima. Using
//     // vstus+vstas would leave 16 B pending and trip VSTAI. Use predicated
//     // vsts directly at maxPtr (always 32-B aligned).
//     if (loop_num == 1) {
//         uint32_t remaining = (total_elem_count < elements_per_dintlv) ? total_elem_count : elements_per_dintlv;
//         uint32_t out_count = CeilDivision(remaining, 32u);
//         MaskReg preg_out = CreatePredicate<T>(out_count);
//         AbsReduceMax_b16_DintlvWindow(srcPtr, 0u, remaining, vb16_max);
//         vsts(vb16_max, maxPtr, 0, distValue, preg_out);
//         return;
//     }
//     // loop_num>=2: stream via vstus, then vstas flushes st_align remainder
//     // at the continuation addr (maxPtr + loop_num*grps_per_dintlv).
//     // Board: stores within a loop may not be ordered w.r.t. one another;
//     // VST_VST forces each iter's vstus to commit before the next iter's.
//     vector_align ureg_max;
//     for (uint16_t i = 0; i < loop_num; ++i) {
//         uint32_t offset = i * elements_per_dintlv;
//         uint32_t remaining = (total_elem_count > offset) ? (total_elem_count - offset) : 0;
//         if (remaining > elements_per_dintlv)
//             remaining = elements_per_dintlv;
//         AbsReduceMax_b16_DintlvWindow(srcPtr, offset, remaining, vb16_max);
//         vstus(ureg_max, blks_per_vl, vb16_max, maxPtr + i * grps_per_dintlv);
//     }
//     vstas(ureg_max, maxPtr + loop_num * grps_per_dintlv, 0);
// }

// // Assumption: input total size is a multiple of 32 VLs.
// // Uses 2 VLs per inner iteration (1 DINTLV + 1 vcgmax + 1 vstus) to avoid
// // WAW hazard on the vstus auto-increment scalar register when using 2 vstus per iteration.
// template <typename T>
// PTO_INTERNAL void AbsReduceMax_b16_ND_largesizes(
//     __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned vl_count, unsigned total_elements_count)
// {
//     static_assert(
//         std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//         "AbsReduceMax_b16_ND_largesizes: T must be bfloat16_t or half");
//     constexpr uint16_t kBf16AbsMask = 0x7FFF;
//     constexpr uint16_t kFp16ExpMask = 0x7C00;
//     constexpr uint16_t kFp16MantissaMask = 0x03FF;
//     constexpr uint16_t kFp16InfBits = 0x7C00;
//     constexpr uint16_t kBf16InfBits = 0x7F80;
//     constexpr uint16_t kBf16NanBits = 0x7FC0;
//     RegTensor<T> vb16_in_1, vb16_in_2, vb16_max_1;
//     RegTensor<uint16_t> vu16_abs_1, vu16_abs_2, vu16_bf16_abs_mask, vu16_fp16_abs_mask, vu16_bf16_inf;
//     RegTensor<uint16_t> vu16_fp16_exp_mask, vu16_fp16_mantissa_mask;
//     RegTensor<uint16_t> vu16_fp16_exp_1, vu16_fp16_exp_2, vu16_fp16_mantissa_1, vu16_fp16_mantissa_2;
//     RegTensor<uint16_t> vu16_bf16_nan;
//     vector_bf16 vb16_bf16_1, vb16_bf16_2;
//     vector_align ureg_max;
//     uint32_t total_count = total_elements_count;
//     constexpr uint32_t grp_size = 32;
//     constexpr uint32_t elements_per_vl = REPEAT_BYTE / sizeof(T); // 256 B / 2 B = 128 elements per VL
//     constexpr uint32_t grps_per_vl = elements_per_vl / grp_size;  // 128 / 32 = 4 groups per VL
//     constexpr uint32_t num_vl_per_inner_loop = 2;                 // 2 VLs per inner loop (1 DINTLV load)
//     constexpr uint32_t num_vl_per_outer_loop = 32;
//     constexpr uint32_t grps_per_inner_loop = num_vl_per_inner_loop * grps_per_vl; // 2 * 4 = 8 grps per inner loop
//     constexpr uint32_t grps_per_outer_loop = num_vl_per_outer_loop * grps_per_vl; // 32 * 4 = 128
//     constexpr uint32_t blks_per_vl = REPEAT_BYTE / BLOCK_SIZE;                    // 8 blocks per VL
//     static constexpr auto distValue =
//         std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
//     vbr(vu16_bf16_abs_mask, kBf16AbsMask);
//     if constexpr (std::is_same<T, half>::value) {
//         vbr(vu16_fp16_abs_mask, kBf16AbsMask);
//         vbr(vu16_fp16_exp_mask, kFp16ExpMask);
//         vbr(vu16_fp16_mantissa_mask, kFp16MantissaMask);
//         vbr(vu16_bf16_inf, kBf16InfBits);
//         vbr(vu16_bf16_nan, kBf16NanBits);
//     }
//     for (uint16_t i = 0; i < (uint16_t)vl_count / num_vl_per_outer_loop; ++i) {        // 32 VLs per outer loop
//         for (uint16_t j = 0; j < num_vl_per_outer_loop / num_vl_per_inner_loop; ++j) { // 2 VLs per inner loop
//             MaskReg preg_vl0 = CreatePredicate<T>(total_count);
//             MaskReg preg_vl1 = CreatePredicate<T>(total_count);
//             uint32_t offset = (i * num_vl_per_outer_loop + j * num_vl_per_inner_loop) * elements_per_vl;
//             uint32_t grp_offset = grps_per_outer_loop * i + grps_per_inner_loop * j;
//             vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16); // loads 2 VLs (256 bf16 elements)

//             if constexpr (std::is_same<T, half>::value) {
//                 vector_bool preg_special_1, preg_special_2, preg_nan_1, preg_nan_2, preg_inf_1, preg_inf_2;
//                 vand(vu16_abs_1, (vector_u16&)vb16_in_1, vu16_fp16_abs_mask, preg_vl0, MODE_ZEROING);
//                 vand(vu16_abs_2, (vector_u16&)vb16_in_2, vu16_fp16_abs_mask, preg_vl1, MODE_ZEROING);
//                 vand(vu16_fp16_exp_1, vu16_abs_1, vu16_fp16_exp_mask, preg_vl0, MODE_ZEROING);
//                 vand(vu16_fp16_exp_2, vu16_abs_2, vu16_fp16_exp_mask, preg_vl1, MODE_ZEROING);
//                 vand(vu16_fp16_mantissa_1, vu16_abs_1, vu16_fp16_mantissa_mask, preg_vl0, MODE_ZEROING);
//                 vand(vu16_fp16_mantissa_2, vu16_abs_2, vu16_fp16_mantissa_mask, preg_vl1, MODE_ZEROING);
//                 vcmps_eq(preg_special_1, vu16_fp16_exp_1, kFp16ExpMask, preg_vl0);
//                 vcmps_eq(preg_special_2, vu16_fp16_exp_2, kFp16ExpMask, preg_vl1);
//                 vcmps_ne(preg_nan_1, vu16_fp16_mantissa_1, 0, preg_special_1);
//                 vcmps_ne(preg_nan_2, vu16_fp16_mantissa_2, 0, preg_special_2);
//                 vcmps_eq(preg_inf_1, vu16_abs_1, kFp16InfBits, preg_vl0);
//                 vcmps_eq(preg_inf_2, vu16_abs_2, kFp16InfBits, preg_vl1);
//                 vcvt(vb16_bf16_1, vb16_in_1, preg_vl0, ROUND_Z);
//                 vcvt(vb16_bf16_2, vb16_in_2, preg_vl1, ROUND_Z);
//                 vsel((vector_u16&)vb16_bf16_1, vu16_bf16_inf, (vector_u16&)vb16_bf16_1, preg_inf_1);
//                 vsel((vector_u16&)vb16_bf16_2, vu16_bf16_inf, (vector_u16&)vb16_bf16_2, preg_inf_2);
//                 vsel((vector_u16&)vb16_bf16_1, vu16_bf16_nan, (vector_u16&)vb16_bf16_1, preg_nan_1);
//                 vsel((vector_u16&)vb16_bf16_2, vu16_bf16_nan, (vector_u16&)vb16_bf16_2, preg_nan_2);
//                 vand(vu16_abs_1, (vector_u16&)vb16_bf16_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
//                 vand(vu16_abs_2, (vector_u16&)vb16_bf16_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
//             } else {
//                 vand(vu16_abs_1, (vector_u16&)vb16_in_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
//                 vand(vu16_abs_2, (vector_u16&)vb16_in_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
//             }

//             vmax(vu16_abs_1, vu16_abs_1, vu16_abs_2, preg_vl0, MODE_ZEROING);
//             vcgmax((vector_u16&)vb16_max_1, vu16_abs_1, preg_vl0, MODE_ZEROING);
//             vstus(ureg_max, blks_per_vl, vb16_max_1, maxPtr + grp_offset);
//         }
//         vstas(ureg_max, maxPtr + grps_per_outer_loop * i, 0);
//     }
// }

// // 2D version of AbsReduceMax_b16: iterates row-by-row, respecting a physical row
// // stride (srcCols) distinct from the valid element count per row (validCols).
// // Use when the dynamic valid width differs from the static (padded) tile width so
// // rows are NOT contiguous in UB. Assumes pad columns [validCols, srcCols) of the
// // source tile have been zero-filled (e.g. by ZeroPadSourceTile) so that pad groups
// // produce a zero group-max naturally through vmax/vcgmax.
// //
// // Max buffer layout: per-row stride = srcCols / 32 (groups per row), matching the
// // flattened layout used by the downstream ExtractB8ExponentAndScaling pass.
// template <typename T>
// PTO_INTERNAL void AbsReduceMax_b16_ND_2D(
//     __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned validRows, unsigned validCols, unsigned srcCols)
// {
//     RegTensor<T> vb16_max_1;
//     vector_align ureg_max;
//     constexpr uint32_t grp_size = 32;
//     constexpr uint32_t elements_per_vl = REPEAT_BYTE / sizeof(T);        // 128
//     constexpr uint32_t elements_per_dintlv = 2 * elements_per_vl;        // 256
//     constexpr uint32_t grps_per_dintlv = elements_per_dintlv / grp_size; // 8 group maxes per DINTLV
//     uint32_t groupsPerRow = srcCols / grp_size;                          // srcCols is always 32-aligned
//     uint16_t loop_num_per_row = CeilDivision(srcCols, elements_per_dintlv);
//     // Max buffer is packed contiguously across rows (row N's maxes sit right after row N-1's).
//     // Stream the stores through a single alignment register with POST_UPDATE so the hardware
//     // tracks its own position; a single vstas at the end drains the residual.
//     __ubuf__ T* writePtr = maxPtr;
//     for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
//         uint32_t src_row_off = row * srcCols;
//         for (uint16_t i = 0; i < loop_num_per_row; ++i) {
//             // Predicates reflect per-DINTLV-register valid element count, computed
//             // against the padded srcCols (source pad lanes are zero → safe for max).
//             uint32_t col_offset = i * elements_per_dintlv;
//             uint32_t remaining = (srcCols > col_offset) ? (srcCols - col_offset) : 0;
//             if (remaining > elements_per_dintlv)
//                 remaining = elements_per_dintlv;
//             AbsReduceMax_b16_DintlvWindow(srcPtr, src_row_off + col_offset, remaining, vb16_max_1);
//             // Clamp store width to the groups actually present in this row; writing a
//             // full grps_per_dintlv (=8) would overshoot into the next row's max slots
//             // when groupsPerRow < 8 (e.g. srcCols=32 → 1 group/row).
//             uint32_t grps_written_in_row = (uint32_t)i * grps_per_dintlv;
//             uint32_t grps_remaining = (groupsPerRow > grps_written_in_row) ? (groupsPerRow - grps_written_in_row) : 0;
//             uint32_t grps_this_iter = (grps_remaining > grps_per_dintlv) ? grps_per_dintlv : grps_remaining;
//             vstus(ureg_max, grps_this_iter, vb16_max_1, writePtr, POST_UPDATE);
//         }
//     }
//     vstas(ureg_max, writePtr, 0, POST_UPDATE);
//     (void)validCols; // padded source makes validCols implicit; retained for API symmetry
// }

// // Computing scalar focus and exponent for F32 -> b8 e4m3 quantization
// template <bool unroll = false>
// PTO_INTERNAL void ExtractB8ExponentAndScaling(
//     __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, unsigned exp_max_loop_count,
//     unsigned total_elements_count, unsigned elementsPerRepeat)
// {
//     static constexpr auto distValue =
//         std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
//     vector_f32 vb32_max;
//     vector_s32 vb32_exponent, vb32_mantissa, vb32_shared_exp, vb32_scaling;
//     vector_s32 vb32_b8_nan, vb32_f32_nan, vb32_b8_emax, vb32_exp_mask, vb32_mantissa_mask, vb32_exp_max;
//     vector_s32 vb32_recip_min_scale, vb32_zero;
//     constexpr int shr = 23;
//     vbr(vb32_exp_mask, 0x7F800000);
//     vbr(vb32_mantissa_mask, 0x007FFFFF);
//     vbr(vb32_b8_nan, 0xFF);
//     vbr(vb32_f32_nan, 0x7FC00000);
//     vbr(vb32_exp_max, 0xFE);
//     vbr(vb32_b8_emax, 8); // Max exponent for e4m3 is 8
//     vbr(vb32_recip_min_scale, 0x7F000000);
//     vbr(vb32_zero, 0);
//     vector_bool preg_special, preg_nan, preg_min_scale;
//     uint32_t scaling_elem_count = total_elements_count * 2;
//     for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
//         uint32_t offset = i * elementsPerRepeat;
//         uint32_t remaining = (total_elements_count > offset) ? (total_elements_count - offset) : 0;
//         if (remaining > elementsPerRepeat)
//             remaining = elementsPerRepeat;
//         vector_bool preg_b32 = CreatePredicate<float>(remaining);
//         vlds((vector_s32&)vb32_max, (__ubuf__ int32_t*)maxPtr, offset, NORM);
//         vand((vector_s32&)vb32_exponent, (vector_s32&)vb32_max, vb32_exp_mask, preg_b32, MODE_ZEROING);
//         vand((vector_s32&)vb32_mantissa, (vector_s32&)vb32_max, vb32_mantissa_mask, preg_b32, MODE_ZEROING);
//         vshrs((vector_s32&)vb32_exponent, (vector_s32&)vb32_exponent, shr, preg_b32, MODE_ZEROING);
//         vsub((vector_u32&)vb32_shared_exp, (vector_u32&)vb32_exponent, (vector_u32&)vb32_b8_emax, preg_b32);
//         vsub((vector_s32&)vb32_scaling, (vector_s32&)vb32_exp_max, (vector_s32&)vb32_shared_exp, preg_b32);
//         vshls((vector_u32&)vb32_scaling, (vector_u32&)vb32_scaling, shr, preg_b32, MODE_ZEROING);

//         vcmps_le(preg_min_scale, (vector_s32&)vb32_exponent, 8, preg_b32);
//         vsel(vb32_scaling, vb32_recip_min_scale, vb32_scaling, preg_min_scale);
//         vsel(vb32_shared_exp, vb32_zero, vb32_shared_exp, preg_min_scale);

//         vcmps_eq(preg_special, (vector_s32&)vb32_exponent, 0xFF, preg_b32);
//         vcmps_ne(preg_nan, (vector_s32&)vb32_mantissa, 0, preg_special);
//         vsel(vb32_scaling, vb32_f32_nan, vb32_scaling, preg_nan);
//         vsel(vb32_shared_exp, vb32_b8_nan, vb32_shared_exp, preg_nan);
//         vsts((vector_s32&)vb32_shared_exp, ((__ubuf__ int32_t*)expPtr), i * elementsPerRepeat / 4, PK4_B32, preg_b32);
//         if constexpr (unroll) {
//             vector_s32 vb32_scaling_0, vb32_scaling_1;
//             vintlv(vb32_scaling_0, vb32_scaling_1, vb32_scaling, vb32_scaling);
//             uint32_t scalingOffset = 2 * offset;
//             uint32_t scalingRemaining0 =
//                 (scaling_elem_count > scalingOffset) ? (scaling_elem_count - scalingOffset) : 0;
//             if (scalingRemaining0 > elementsPerRepeat)
//                 scalingRemaining0 = elementsPerRepeat;
//             uint32_t scalingOffset1 = scalingOffset + elementsPerRepeat;
//             uint32_t scalingRemaining1 =
//                 (scaling_elem_count > scalingOffset1) ? (scaling_elem_count - scalingOffset1) : 0;
//             if (scalingRemaining1 > elementsPerRepeat)
//                 scalingRemaining1 = elementsPerRepeat;
//             MaskReg preg_scaling_0 = CreatePredicate<float>(scalingRemaining0);
//             MaskReg preg_scaling_1 = CreatePredicate<float>(scalingRemaining1);
//             vsts((vector_s32&)vb32_scaling_0, ((__ubuf__ int32_t*)scalingPtr), scalingOffset, NORM_B32, preg_scaling_0);
//             vsts(
//                 (vector_s32&)vb32_scaling_1, ((__ubuf__ int32_t*)scalingPtr), scalingOffset1, NORM_B32, preg_scaling_1);

//         } else
//             vsts((vector_s32&)vb32_scaling, ((__ubuf__ int32_t*)scalingPtr), offset, distValue, preg_b32);
//     }
// }

// // B16 (BF16/FP16) -> FP8 shared-exponent + BF16 reciprocal scaling for MXFP8.
// // AbsReduceMax_b16_ND stores BF16 abs raw bits in maxPtr for both BF16 and FP16.
// // E8M0 encoded 0 is the minimum scale 2^-127, so maxExp==0 keeps the reciprocal
// // BF16 scale at 2^127 instead of becoming numeric zero.
// template <typename T>
// PTO_INTERNAL void ExtractB8ExponentAndScalingVL(
//     __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, uint32_t off, uint32_t rem)
// {
//     static_assert(
//         std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//         "ExtractB8ExponentAndScalingVL B16: T must be bfloat16_t or half");
//     constexpr uint16_t kBf16ExpMask = 0x7F80;
//     constexpr uint16_t kBf16MantissaMask = 0x007F;
//     constexpr uint16_t kFp8E4M3MaxExp = 0x0400;
//     constexpr uint16_t kBf16ExpBias = 0x7F00;
//     constexpr uint16_t kFp8Nan = 0x00FF;
//     constexpr uint16_t kNanCustomization = 0x7F81;

//     __ubuf__ uint16_t* maxPtr_u16 = (__ubuf__ uint16_t*)maxPtr;
//     __ubuf__ uint16_t* scalingPtr_u16 = (__ubuf__ uint16_t*)scalingPtr;
//     RegTensor<uint16_t> vu16_max_abs, vu16_max_exp, vu16_mantissa;
//     RegTensor<uint16_t> vu16_shared_exp, vu16_scale_value, vu16_recip_scale;
//     RegTensor<uint16_t> vu16_max_exp_value, vu16_scale_bias, vu16_fp8_nan;
//     RegTensor<uint16_t> vu16_nan, vu16_exp_mask, vu16_mantissa_mask;
//     vector_bool preg_clamp, preg_special, preg_nan;
//     vector_bool preg_b16 = CreatePredicate<T>(rem);

//     vbr(vu16_max_exp_value, kFp8E4M3MaxExp);
//     vbr(vu16_scale_bias, kBf16ExpBias);
//     vbr(vu16_fp8_nan, kFp8Nan);
//     vbr(vu16_nan, kNanCustomization);
//     vbr(vu16_exp_mask, kBf16ExpMask);
//     vbr(vu16_mantissa_mask, kBf16MantissaMask);

//     vlds(vu16_max_abs, maxPtr_u16, off, NORM);
//     vand(vu16_max_exp, vu16_max_abs, vu16_exp_mask, preg_b16, MODE_ZEROING);
//     vand(vu16_mantissa, vu16_max_abs, vu16_mantissa_mask, preg_b16, MODE_ZEROING);
//     vcmps_eq(preg_special, vu16_max_exp, kBf16ExpMask, preg_b16);
//     vcmps_ne(preg_nan, vu16_mantissa, 0, preg_special);
//     vcmps_le(preg_clamp, vu16_max_exp, kFp8E4M3MaxExp, preg_b16);
//     vsel(vu16_max_exp, vu16_max_exp_value, vu16_max_exp, preg_clamp);

//     vsub(vu16_shared_exp, vu16_max_exp, vu16_max_exp_value, preg_b16, MODE_ZEROING);
//     vshrs(vu16_scale_value, vu16_shared_exp, 7, preg_b16, MODE_ZEROING);
//     vsel(vu16_scale_value, vu16_fp8_nan, vu16_scale_value, preg_nan);
//     vsts(vu16_scale_value, (__ubuf__ uint16_t*)expPtr, off / sizeof(T), PK_B16, preg_b16);

//     // reciprocal_scale = 2^(127 - e8m0_biased_exp), stored as BF16 bits.
//     vsub(vu16_recip_scale, vu16_scale_bias, vu16_shared_exp, preg_b16, MODE_ZEROING);
//     vsel(vu16_recip_scale, vu16_nan, vu16_recip_scale, preg_nan);
//     vsts(vu16_recip_scale, scalingPtr_u16, off, NORM_B16, preg_b16);
// }

// template <typename T>
// PTO_INTERNAL void ExtractB8ExponentAndScaling(
//     __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned exp_max_loop_count,
//     unsigned total_elements_count)
// {
//     static_assert(
//         std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//         "ExtractB8ExponentAndScaling B16: T must be bfloat16_t or half");
//     constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);

//     for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
//         ExtractB8ExponentAndScalingVL<T>(maxPtr, expPtr, scalingPtr, i * elementsPerVL, total_elements_count);
//     }
// }

// // 2D variant of ExtractB8ExponentAndScaling for the padded (validCols != srcCols) path.
// // Iterates per row, processing only the groups backing valid columns. Max, exp and
// // scaling buffers share a packed per-row layout (row r's first group at row * groupsPerRow).
// // Only safe when groupsPerRow * sizeof(T) is a multiple of 32 B (i.e. srcCols >= 512 for
// // B16) so that each per-row NORM load/store address is 32-byte aligned. Callers must
// // gate on that condition.
// template <typename T>
// PTO_INTERNAL void ExtractB8ExponentAndScaling_2D(
//     __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned validRows, unsigned validCols,
//     unsigned srcCols)
// {
//     static_assert(
//         std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//         "ExtractB8ExponentAndScaling_2D: T must be bfloat16_t or half");
//     constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T); // 128 group-maxes per VL

//     uint32_t groupsPerRow = srcCols / 32; // srcCols is 32-aligned
//     uint32_t validGroupsPerRow = CeilDivision((uint32_t)validCols, 32u);
//     uint16_t loopsPerRow = CeilDivision(validGroupsPerRow, elementsPerVL);
//     for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
//         uint32_t rowOff = row * groupsPerRow; // group-indexed offset into packed buffers
//         for (uint16_t i = 0; i < loopsPerRow; ++i) {
//             uint32_t off = i * elementsPerVL;
//             uint32_t rem = (validGroupsPerRow > off) ? (validGroupsPerRow - off) : 0;
//             if (rem > elementsPerVL)
//                 rem = elementsPerVL;
//             ExtractB8ExponentAndScalingVL<T>(maxPtr + rowOff, expPtr + rowOff, scalingPtr + rowOff, off, rem);
//         }
//     }
// }

// // FP32 -> FP8
// PTO_INTERNAL void CalcQuantizedFP8Values(
//     __ubuf__ float* srcPtr, __ubuf__ float* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned vl_count,
//     unsigned elementsPerRepeat, unsigned total_elements_count, MaskReg& preg_lower32, MaskReg& preg_upper32)
// {
//     vector_f32 vb32_scaling_0, vb32_scaling_1, vb32_in, vb32_out_1, vb32_out_2, vb32_out;
//     vector_f8e4m3 vb8_out;
//     uint32_t elem_count = total_elements_count;
//     MaskReg preg_ALL = pset_b32(PAT_ALL);
//     for (uint16_t i = 0; i < (uint16_t)vl_count; ++i) {
//         MaskReg preg = CreatePredicate<float>(elem_count);
//         vlds(vb32_scaling_0, scalingPtr, 2 * i, BRC_B32);
//         vlds(vb32_scaling_1, scalingPtr + 1, 2 * i, BRC_B32);
//         vlds(vb32_in, srcPtr, i * elementsPerRepeat, NORM);
//         vmul(vb32_out_1, vb32_in, vb32_scaling_0, preg_lower32, MODE_ZEROING);
//         vmul(vb32_out_2, vb32_in, vb32_scaling_1, preg_upper32, MODE_ZEROING);
//         vor(vb32_out, vb32_out_1, vb32_out_2, preg_ALL);
//         vcvt((vector_f8e4m3&)vb8_out, (vector_f32&)vb32_out, preg, ROUND_R, RS_ENABLE, PART_P0);
//         vsts((vector_u8&)vb8_out, (__ubuf__ uint8_t*)dstPtr, i * elementsPerRepeat, PK4_B32, preg);
//     }
// }

// PTO_INTERNAL void CalcQuantizedFP8Values_Unroll2(
//     __ubuf__ float* srcPtr, __ubuf__ float* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned vl_count,
//     unsigned elementsPerRepeat, unsigned total_elements_count)
// {
//     vector_f32 vb32_scaling, vb32_in_even, vb32_in_odd, vb32_out_1, vb32_out_2, vb32_out;
//     vector_f8e4m3 vb8_out_P0, vb8_out_P1, vb8_out;
//     uint32_t elem_count = total_elements_count;
//     MaskReg preg_ALL = pset_b32(PAT_ALL);
//     MaskReg preg_ALL_b8 = pset_b8(PAT_ALL);
//     for (uint16_t i = 0; i < (uint16_t)vl_count / 2; ++i) {
//         vlds(vb32_scaling, scalingPtr, 8 * i, E2B_B32);
//         vlds(vb32_in_even, vb32_in_odd, srcPtr, 2 * i * elementsPerRepeat, DINTLV_B32);
//         vmul(vb32_out_1, vb32_in_even, vb32_scaling, preg_ALL, MODE_ZEROING);
//         vmul(vb32_out_2, vb32_in_odd, vb32_scaling, preg_ALL, MODE_ZEROING);
//         vcvt((vector_f8e4m3&)vb8_out_P0, (vector_f32&)vb32_out_1, preg_ALL, ROUND_R, RS_ENABLE, PART_P0);
//         vcvt((vector_f8e4m3&)vb8_out_P1, (vector_f32&)vb32_out_2, preg_ALL, ROUND_R, RS_ENABLE, PART_P1);
//         vor(vb8_out, vb8_out_P0, vb8_out_P1, preg_ALL_b8);
//         vsts((vector_u16&)vb8_out, (__ubuf__ uint16_t*)dstPtr, i * elementsPerRepeat, PK_B32, preg_ALL);
//     }
// }

// // B16 (BF16/FP16) -> FP8. FP16 uses BF16 reciprocal scale, matching dynamic_mx_quant:
// // convert input and BF16 scale to fp32, multiply in fp32, then downcast to fp8.
// // Quantize one 256-element DINTLV_B16 window to FP8: scale via broadcast of
// // 8 per-group scaling values, upcast b16->fp32 (EVEN/ODD), downcast fp32->fp8
// // (PART_P0-P3 pack mod-4 bytes), OR-combine, and store.
// template <typename T>
// PTO_INTERNAL void CalcQuantizedFP8Values_B16_Window(
//     __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, uint16_t i, uint32_t offset_b16,
//     uint32_t remaining)
// {
//     constexpr uint32_t elementsPerVL_b8 = REPEAT_BYTE / sizeof(uint8_t);
//     RegTensor<T> vb16_scaling, vb16_in_1, vb16_in_2, vb16_out_1, vb16_out_2;
//     vector_f32 vb32_cvt_1, vb32_cvt_2, vb32_cvt_3, vb32_cvt_4;
//     vector_f8e4m3 vb8_or1, vb8_or2, vb8_out, vb8_p0, vb8_p1, vb8_p2, vb8_p3;
//     uint32_t even_count = (remaining + 1) / 2;
//     uint32_t odd_count = remaining / 2;
//     uint32_t b8_count = remaining;
//     MaskReg preg_b16_1 = CreatePredicate<T>(even_count);
//     MaskReg preg_b16_2 = CreatePredicate<T>(odd_count);
//     MaskReg preg_b8 = CreatePredicate<uint8_t>(b8_count);
//     vlds(vb16_in_1, vb16_in_2, srcPtr, offset_b16, DINTLV_B16);
//     if constexpr (std::is_same<T, half>::value) {
//         vector_bf16 vb16_scaling_bf16;
//         vector_f32 vb32_scaling;
//         MaskReg preg_all_b16 = pset_b16(PAT_ALL);
//         vlds((vector_u16&)vb16_scaling_bf16, (__ubuf__ uint16_t*)scalingPtr, 8 * i, E2B_B16);
//         vcvt(vb32_scaling, vb16_scaling_bf16, preg_all_b16, PART_EVEN);
//         // b16->fp32 EVEN/ODD splits each 128-lane reg into 2x64 fp32 (mod-4: 0,2,1,3).
//         vcvt(vb32_cvt_1, vb16_in_1, preg_b16_1, PART_EVEN);
//         vcvt(vb32_cvt_2, vb16_in_1, preg_b16_1, PART_ODD);
//         vcvt(vb32_cvt_3, vb16_in_2, preg_b16_2, PART_EVEN);
//         vcvt(vb32_cvt_4, vb16_in_2, preg_b16_2, PART_ODD);
//         vmul(vb32_cvt_1, vb32_cvt_1, vb32_scaling, preg_b16_1, MODE_ZEROING);
//         vmul(vb32_cvt_2, vb32_cvt_2, vb32_scaling, preg_b16_1, MODE_ZEROING);
//         vmul(vb32_cvt_3, vb32_cvt_3, vb32_scaling, preg_b16_2, MODE_ZEROING);
//         vmul(vb32_cvt_4, vb32_cvt_4, vb32_scaling, preg_b16_2, MODE_ZEROING);
//     } else {
//         vlds((vector_u16&)vb16_scaling, (__ubuf__ uint16_t*)scalingPtr, 8 * i, E2B_B16);
//         vmul(vb16_out_1, vb16_in_1, vb16_scaling, preg_b16_1, MODE_ZEROING);
//         vmul(vb16_out_2, vb16_in_2, vb16_scaling, preg_b16_2, MODE_ZEROING);
//         // b16->fp32 EVEN/ODD splits each 128-lane reg into 2x64 fp32 (mod-4: 0,2,1,3).
//         vcvt(vb32_cvt_1, vb16_out_1, preg_b16_1, PART_EVEN);
//         vcvt(vb32_cvt_2, vb16_out_1, preg_b16_1, PART_ODD);
//         vcvt(vb32_cvt_3, vb16_out_2, preg_b16_2, PART_EVEN);
//         vcvt(vb32_cvt_4, vb16_out_2, preg_b16_2, PART_ODD);
//     }
//     // fp32->fp8 P0..P3 writes to bytes 0..3 of each 32-bit slot; pair with mod-4 index.
//     vcvt(vb8_p0, vb32_cvt_1, preg_b16_1, ROUND_R, RS_ENABLE, PART_P0);
//     vcvt(vb8_p1, vb32_cvt_3, preg_b16_2, ROUND_R, RS_ENABLE, PART_P1);
//     vcvt(vb8_p2, vb32_cvt_2, preg_b16_1, ROUND_R, RS_ENABLE, PART_P2);
//     vcvt(vb8_p3, vb32_cvt_4, preg_b16_2, ROUND_R, RS_ENABLE, PART_P3);
//     vor(vb8_or1, vb8_p0, vb8_p1, preg_b8);
//     vor(vb8_or2, vb8_p2, vb8_p3, preg_b8);
//     vor(vb8_out, vb8_or1, vb8_or2, preg_b8);
//     vsts((vector_u8&)vb8_out, (__ubuf__ uint8_t*)dstPtr, i * elementsPerVL_b8, NORM_B8, preg_b8);
// }

// // B16 (BF16/FP16) -> FP8. 2 VLs per iter (one DINTLV_B16 load). Ceil-div on
// // iter count so a partial final window is still covered; per-window predicates
// // clamp to the exact remaining element count.
// template <typename T>
// PTO_INTERNAL void CalcQuantizedFP8Values(
//     __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned total_elements_count)
// {
//     static_assert(
//         std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//         "CalcQuantizedFP8Values B16: T must be bfloat16_t or half");
//     constexpr uint32_t elementsPerVL_b16 = REPEAT_BYTE / sizeof(T);
//     constexpr uint32_t elementsPerDintlv = 2 * elementsPerVL_b16;
//     uint32_t vl_count = CeilDivision(total_elements_count, elementsPerVL_b16);
//     uint16_t quant_iters = (uint16_t)CeilDivision(vl_count, static_cast<uint32_t>(2u));
//     for (uint16_t i = 0; i < quant_iters; ++i) {
//         uint32_t offset_b16 = i * elementsPerDintlv;
//         uint32_t remaining = (total_elements_count > offset_b16) ? (total_elements_count - offset_b16) : 0;
//         if (remaining > elementsPerDintlv)
//             remaining = elementsPerDintlv;
//         CalcQuantizedFP8Values_B16_Window<T>(srcPtr, scalingPtr, dstPtr, i, offset_b16, remaining);
//     }
// }

// // 2D variant of CalcQuantizedFP8Values for the padded (validCols != srcCols) path.
// // Iterates per row using srcCols as src/dst stride (elements) and groupsPerRow as the
// // packed scaling-buffer stride. Processes only validCols elements per row; pad-col dst
// // bytes are not written (TSTORE trims them via GM shape). Alignment requirements:
// //   - scalingPtr + row * groupsPerRow must be 16 B-aligned for E2B_B16 load
// //     (groupsPerRow * sizeof(T) % 16 == 0, i.e. srcCols % 256 == 0)
// //   - srcPtr  + row * srcCols must be 32 B-aligned for DINTLV_B16 (srcCols % 16 == 0)
// //   - dstPtr  + row * srcCols must be 32 B-aligned for NORM_B8 (srcCols % 32 == 0, always true)
// // Callers must gate on the srcCols %% 256 == 0 condition.
// template <typename T>
// PTO_INTERNAL void CalcQuantizedFP8Values_2D(
//     __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned validRows, unsigned validCols,
//     unsigned srcCols)
// {
//     static_assert(
//         std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//         "CalcQuantizedFP8Values_2D B16: T must be bfloat16_t or half");
//     constexpr uint32_t elementsPerVL_b16 = REPEAT_BYTE / sizeof(T); // 128
//     constexpr uint32_t elementsPerDintlv = 2 * elementsPerVL_b16;   // 256
//     uint32_t groupsPerRow = srcCols / 32;
//     uint16_t loopsPerRow = CeilDivision((uint32_t)validCols, elementsPerDintlv);
//     for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
//         uint32_t srcRowOff = row * srcCols;        // T-indexed
//         uint32_t dstRowOff = row * srcCols;        // uint8_t-indexed (1 byte per elem)
//         uint32_t scaleRowOff = row * groupsPerRow; // T-indexed, packed scaling layout
//         for (uint16_t i = 0; i < loopsPerRow; ++i) {
//             uint32_t colOff = i * elementsPerDintlv;
//             uint32_t remaining = (validCols > colOff) ? (validCols - colOff) : 0;
//             if (remaining > elementsPerDintlv)
//                 remaining = elementsPerDintlv;
//             CalcQuantizedFP8Values_B16_Window<T>(
//                 srcPtr + srcRowOff, scalingPtr + scaleRowOff, dstPtr + dstRowOff, i, colOff, remaining);
//         }
//     }
// }

// // FP32 -> MXFP8 quantization: AbsReduceMax + ExponentScaling + FP8 conversion.
// template <unsigned StaticRows, unsigned StaticCols>
// PTO_INTERNAL void TQuant_MXFP8_F32(
//     __ubuf__ float* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ float* maxPtr,
//     __ubuf__ float* scalingPtr, uint16_t vl_count, unsigned exp_loop_count, uint32_t numGroups,
//     unsigned elementsPerRepeat, uint32_t total_elements_count, unsigned validRows, unsigned validCols)
// {
//     MaskReg preg_lower32 = pset_b32(PAT_VL32), preg_upper32, preg_ALL = pset_b32(PAT_ALL);
//     pxor(preg_upper32, preg_ALL, preg_lower32, preg_ALL);
//     __ubuf__ float* maxPtr_backup = maxPtr;
//     if (validRows * validCols <= 1024)
//         AbsReduceMax_Naive(
//             srcPtr, maxPtr, total_elements_count, vl_count, elementsPerRepeat, preg_lower32, preg_upper32);
//     else {
//         constexpr uint32_t kLargeReduceElems = 2048;
//         uint32_t large_total = (total_elements_count / kLargeReduceElems) * kLargeReduceElems;
//         uint32_t tail_total = total_elements_count - large_total;
//         if (large_total > 0) {
//             uint16_t large_vl_count = large_total / elementsPerRepeat;
//             AbsReduceMax_f32_opt_largesizes(srcPtr, maxPtr, large_vl_count, elementsPerRepeat, large_total);
//         }
//         if (tail_total > 0) {
//             uint32_t large_groups = large_total / 32;
//             uint16_t tail_vl_count = CeilDivision(tail_total, elementsPerRepeat);
//             AbsReduceMax_Naive(
//                 srcPtr + large_total, maxPtr + large_groups, tail_total, tail_vl_count, elementsPerRepeat, preg_lower32,
//                 preg_upper32);
//         }
//     }
//     mem_bar(VST_VLD);
//     maxPtr = maxPtr_backup;
//     constexpr bool unroll = (StaticRows * StaticCols > 1024) && (StaticRows * StaticCols % 256 == 0);
//     ExtractB8ExponentAndScaling<unroll>(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups, elementsPerRepeat);
//     mem_bar(VST_VLD);
//     if constexpr (unroll)
//         CalcQuantizedFP8Values_Unroll2(srcPtr, scalingPtr, dstPtr, vl_count, elementsPerRepeat, total_elements_count);
//     else
//         CalcQuantizedFP8Values(
//             srcPtr, scalingPtr, dstPtr, vl_count, elementsPerRepeat, total_elements_count, preg_lower32, preg_upper32);
// }

// // B16 (BF16/FP16) -> MXFP8 quantization: AbsReduceMax + ExponentScaling + FP8 conversion.
// // When validCols == srcCols (static == dynamic width), the source tile is contiguous in UB
// // so the flat 1D reducer applies. Otherwise rows are padded to srcCols (ZeroPadSourceTile)
// // and we dispatch the 2D per-row reducer that honors the row stride. The 2D Extract/Calc
// // passes are only used when srcCols % 512 == 0 (NORM 32 B / E2B_B16 16 B alignment), else
// // we fall back to the flat Extract/Calc over the zero-padded buffer (pad lanes are zero
// // so the result is exact; TSTORE trims pad cols via the GM shape).
// template <typename T>
// PTO_INTERNAL void TQuant_MXFP8_B16(
//     __ubuf__ T* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxPtr, __ubuf__ T* scalingPtr,
//     uint16_t vl_count, unsigned exp_loop_count, uint32_t numGroups, uint32_t total_elements_count, unsigned validRows,
//     unsigned validCols, unsigned srcCols)
// {
//     __ubuf__ T* maxPtr_backup = maxPtr;
//     if (validCols == srcCols) {
//         // 1D fast path: source is contiguous; pick the best flat reducer by size.znme
//         constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
//         constexpr uint32_t elementsPerLargeLoop = 32 * elementsPerVL;
//         if (total_elements_count % elementsPerLargeLoop == 0)
//             AbsReduceMax_b16_ND_largesizes(srcPtr, maxPtr, vl_count, total_elements_count);
//         else
//             AbsReduceMax_b16_ND(srcPtr, maxPtr, vl_count, total_elements_count);
//         // Board: add VST_VST alongside VST_VLD/VV_ALL. Sim orders stores implicitly,
//         // board does not — missing VST_VST lets Phase-3 E2B_B16 read stale scaling.
//         mem_bar(VST_VLD);
//         maxPtr = maxPtr_backup;
//         ExtractB8ExponentAndScaling(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
//         mem_bar(VST_VLD);
//         CalcQuantizedFP8Values(srcPtr, scalingPtr, dstPtr, total_elements_count);
//     } else {
//         // 2D path: iterate per row with srcCols stride. ZeroPadSourceTile has zeroed
//         // pad lanes so per-row max is correct.
//         AbsReduceMax_b16_ND_2D(srcPtr, maxPtr, validRows, validCols, srcCols);
//         mem_bar(VST_VLD);
//         maxPtr = maxPtr_backup;
//         // Downstream 2D Extract/Calc need per-row addresses to meet NORM/E2B_B16
//         // alignment. NORM B16 requires 32 B → groupsPerRow*sizeof(T) % 32 == 0,
//         // i.e. srcCols % 512 == 0. When that holds we skip the pad-col work;
//         // otherwise fall back to flat 1D over the padded buffer (pad lanes are zero
//         // so the result is exact — TSTORE trims pad cols via the GM shape).
//         if (srcCols % 512 == 0) {
//             ExtractB8ExponentAndScaling_2D<T>(maxPtr, expPtr, scalingPtr, validRows, validCols, srcCols);
//             mem_bar(VST_VLD);
//             CalcQuantizedFP8Values_2D<T>(srcPtr, scalingPtr, dstPtr, validRows, validCols, srcCols);
//         } else {
//             ExtractB8ExponentAndScaling(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
//             mem_bar(VST_VLD);
//             CalcQuantizedFP8Values(srcPtr, scalingPtr, dstPtr, total_elements_count);
//         }
//     }
// }

// // Zero-pad columns [validCols, StaticCols) of a 16-bit source tile at VL-aligned
// // offsets (full-VL vlds -> vsel -> vsts). Sub-VL stores at non-VL-aligned offsets
// // are unreliable on some hardware revisions. Requires StaticCols | elemPerVL.
// // Must be called from inside a __VEC_SCOPE__.
// template <typename T, unsigned StaticCols>
// PTO_INTERNAL void ZeroPadColumns_VLAligned(__ubuf__ T* srcPtr, unsigned validRows, unsigned validCols)
// {
//     constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
//     static_assert(elemPerVL % StaticCols == 0, "StaticCols must evenly divide elements-per-VL for VL-aligned padding");
//     constexpr unsigned rowsPerVL = elemPerVL / StaticCols;

//     MaskReg pg_all = PSetTyped<T>(PAT_ALL);

//     // Build a periodic predicate: bit p is set iff (p % StaticCols) < validCols.
//     // Row 0 contributes positions [0, validCols).
//     uint32_t vc = (uint32_t)validCols;
//     MaskReg preg_valid = CreatePredicate<T>(vc);
//     for (uint16_t r = 1; r < (uint16_t)rowsPerVL; ++r) {
//         uint32_t rangeStart = (uint32_t)(r * StaticCols);
//         uint32_t rangeEnd = rangeStart + (uint32_t)validCols;
//         MaskReg p_end = CreatePredicate<T>(rangeEnd);
//         MaskReg p_start = CreatePredicate<T>(rangeStart);
//         MaskReg p_row;
//         pnot(p_row, p_start, p_end);
//         por(preg_valid, preg_valid, p_row, pg_all);
//     }

//     RegTensor<T> vreg_zero;
//     vdup(vreg_zero, (T)0, pg_all, MODE_ZEROING);

//     // Write-only: store zeros at padding positions without reading the source.
//     // Avoids RMW on MTE2-written UB data which can race on hardware.
//     MaskReg preg_pad;
//     pxor(preg_pad, pg_all, preg_valid, pg_all);

//     uint32_t totalElems = (uint32_t)(validRows * StaticCols);
//     uint16_t vlCount = CeilDivision(totalElems, (unsigned)elemPerVL);

//     for (uint16_t vi = 0; vi < vlCount; ++vi) {
//         vsts(vreg_zero, srcPtr, vi * elemPerVL, NORM_B16, preg_pad);
//     }
// }

// // Fallback zero-padding using vstus/vstas for cases where StaticCols doesn't divide VL.
// // Must be called from inside a __VEC_SCOPE__.
// template <typename T, unsigned StaticCols>
// PTO_INTERNAL void ZeroPadColumns_Unaligned(__ubuf__ T* srcPtr, unsigned validRows, unsigned validCols)
// {
//     constexpr unsigned padElemPerRepeat = REPEAT_BYTE / sizeof(T);
//     unsigned padCols = StaticCols - validCols;
//     uint16_t padRepeatTimes = CeilDivision(padCols, padElemPerRepeat);
//     RegTensor<T> vreg_zero;
//     UnalignReg ureg_pad;
//     MaskReg pg_all = PSetTyped<T>(PAT_ALL);
//     vdup(vreg_zero, (T)0, pg_all, MODE_ZEROING);
//     for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
//         uint32_t cols = (uint32_t)(padCols);
//         __ubuf__ T* pdst = srcPtr + i * StaticCols + validCols;
//         for (uint16_t j = 0; j < padRepeatTimes; ++j) {
//             uint32_t sreg = cols > padElemPerRepeat ? padElemPerRepeat : cols;
//             vstus(ureg_pad, sreg, vreg_zero, pdst, POST_UPDATE);
//             cols -= padElemPerRepeat;
//         }
//         vstas(ureg_pad, pdst, 0, POST_UPDATE);
//     }
// }

// // Zero-pad source tile columns for non-float types. Dispatches between VL-aligned
// // (full-VL vlds/vsel/vsts) and unaligned (vstus/vstas) paths based on tile geometry.
// // Must be called from inside a __VEC_SCOPE__.
// template <typename T, unsigned StaticCols>
// PTO_INTERNAL void ZeroPadSourceTile(__ubuf__ T* srcPtr, unsigned validRows, unsigned validCols)
// {
//     if constexpr (!std::is_same<T, float>::value) {
//         if (validCols < StaticCols) {
//             constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
//             if constexpr (elemPerVL % StaticCols == 0)
//                 ZeroPadColumns_VLAligned<T, StaticCols>(srcPtr, validRows, validCols);
//             else
//                 ZeroPadColumns_Unaligned<T, StaticCols>(srcPtr, validRows, validCols);
//         }
//     }
// }

// // TQuant: FP32/BF16/FP16 -> MXFP8 (e4m3) quantization, ND mode only.
// template <
//     typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax, typename TileDataScaling>
// __tf__ PTO_INTERNAL void TQuant_MXFP8_Impl(
//     typename TileDataOut::TileDType __out__ dst, typename TileDataExp::TileDType __out__ exp,
//     typename TileDataMax::TileDType __out__ max, typename TileDataScaling::TileDType __out__ scaling,
//     typename TileDataSrc::TileDType __in__ src, unsigned validRows, unsigned validCols)
// {
//     using T = typename TileDataSrc::DType;
//     using ExpT = typename TileDataExp::DType;
//     using OutT = typename TileDataOut::DType;
//     __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
//     __ubuf__ ExpT* expPtr = (__ubuf__ ExpT*)__cce_get_tile_ptr(exp);
//     __ubuf__ OutT* dstPtr = (__ubuf__ OutT*)__cce_get_tile_ptr(dst);
//     __ubuf__ T* maxPtr = (__ubuf__ T*)__cce_get_tile_ptr(max);
//     __ubuf__ T* scalingPtr = (__ubuf__ T*)__cce_get_tile_ptr(scaling);

//     set_ctrl(static_cast<uint64_t>(1) << 50);
//     __VEC_SCOPE__
//     {
//         ZeroPadSourceTile<T, TileDataSrc::Cols>(srcPtr, validRows, validCols);
//         mem_bar(VST_VLD);

//         constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
//         uint32_t totalElems = validRows * (unsigned)TileDataSrc::Cols;
//         uint16_t vlCount = CeilDivision(totalElems, elemPerVL);
//         uint32_t numGroups = totalElems / 32;
//         unsigned expLoopCount = CeilDivision(numGroups, elemPerVL);
//         if constexpr (std::is_same<T, float>::value)
//             TQuant_MXFP8_F32<TileDataSrc::Rows, TileDataSrc::Cols>(
//                 srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, vlCount, expLoopCount,
//                 numGroups, elemPerVL, totalElems, validRows, validCols);
//         else
//             TQuant_MXFP8_B16(
//                 srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, vlCount, expLoopCount,
//                 numGroups, totalElems, validRows, validCols, (unsigned)TileDataSrc::Cols);
//     }
// }

// template <typename TileDataOut, typename TileDataSrc, typename TileDataPara>
// __tf__ PTO_INTERNAL void TQuant_Int8Sym(
//     typename TileDataOut::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
//     typename TileDataPara::TileDType __in__ scale, unsigned validRows, unsigned validCols)
// {
//     using T = typename TileDataSrc::DType;  // fp32
//     using S = typename TileDataPara::DType; // fp32
//     using U = typename TileDataOut::DType;  // int8
//     __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
//     __ubuf__ U* dstPtr = (__ubuf__ U*)__cce_get_tile_ptr(dst);
//     __ubuf__ S* scalePtr = (__ubuf__ S*)__cce_get_tile_ptr(scale);
//     uint16_t repeatTimes = CeilDivision(validCols, ELE_CNT_B32);
//     __VEC_SCOPE__
//     {
//         RegTensor<float> v_input, v_scale;
//         RegTensor<int32_t> v_s32;
//         RegTensor<half> vb16;
//         RegTensor<int8_t> v_output_s8;
//         for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
//             uint32_t sreg = validCols;
//             for (uint16_t idx = 0; idx < repeatTimes; ++idx) {
//                 MaskReg preg_b32 = CreatePredicate<float>(sreg);
//                 vlds(v_scale, scalePtr, row, BRC_B32); // broadcast row scaling
//                 vlds(v_input, srcPtr, ELE_CNT_B32 * idx + row * TileDataSrc::Cols, NORM);
//                 vmul(v_input, v_input, v_scale, preg_b32, MODE_ZEROING);
//                 // Round once at fp32 (s32 round-trip) then exact fp32->fp16->s8.
//                 vcvt(v_s32, v_input, preg_b32, ROUND_R, RS_ENABLE);
//                 vcvt(v_input, v_s32, preg_b32, ROUND_R);
//                 vcvt(vb16, v_input, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
//                 vcvt(v_output_s8, vb16, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
//                 vsts(v_output_s8, dstPtr, ELE_CNT_B32 * idx + row * TileDataOut::Cols, PK4_B32, preg_b32);
//             }
//         }
//     }
// }

// // TQuant: fp32 -> u8 conversion, Int8Asym
// template <typename TileDataOut, typename TileDataSrc, typename TileDataPara>
// __tf__ PTO_INTERNAL void TQuant_Int8Asym(
//     typename TileDataOut::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
//     typename TileDataPara::TileDType __in__ scale, typename TileDataPara::TileDType __in__ offset, unsigned validRows,
//     unsigned validCols)
// {
//     using T = typename TileDataSrc::DType;  // fp32
//     using U = typename TileDataOut::DType;  // uint8
//     using S = typename TileDataPara::DType; // fp32
//     __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
//     __ubuf__ U* dstPtr = (__ubuf__ U*)__cce_get_tile_ptr(dst);
//     __ubuf__ S* scalePtr = (__ubuf__ S*)__cce_get_tile_ptr(scale);
//     __ubuf__ S* offsetPtr = (__ubuf__ S*)__cce_get_tile_ptr(offset);
//     uint16_t repeatTimes = CeilDivision(validCols, ELE_CNT_B32);
//     __VEC_SCOPE__
//     {
//         RegTensor<float> vb32_scale, vb32_input, vb32_offset;
//         RegTensor<int32_t> vb32_int;
//         RegTensor<half> vb16_output;
//         RegTensor<uint8_t> vb8_output;
//         for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
//             uint32_t sreg = validCols;
//             for (uint16_t idx = 0; idx < repeatTimes; ++idx) {
//                 MaskReg preg_b32 = CreatePredicate<float>(sreg);
//                 vlds(vb32_scale, scalePtr, row, BRC_B32);   // broadcast row scaling
//                 vlds(vb32_offset, offsetPtr, row, BRC_B32); // broadcast row offset
//                 vlds(vb32_input, srcPtr, ELE_CNT_B32 * idx + row * TileDataSrc::Cols, NORM);
//                 vmul(vb32_input, vb32_input, vb32_scale, preg_b32, MODE_ZEROING);
//                 vadd(vb32_input, vb32_input, vb32_offset, preg_b32, MODE_ZEROING);
//                 // Round once at fp32 (s32 round-trip) then exact fp32->fp16->u8.
//                 vcvt(vb32_int, vb32_input, preg_b32, ROUND_R, RS_ENABLE);
//                 vcvt(vb32_input, vb32_int, preg_b32, ROUND_R);
//                 vcvt(vb16_output, vb32_input, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
//                 vcvt(vb8_output, vb16_output, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
//                 vsts(vb8_output, dstPtr, ELE_CNT_B32 * idx + row * TileDataOut::Cols, PK4_B32, preg_b32);
//             }
//         }
//     }
// }

// // TQuant Interface for FP32/FP16/BF16->INT4/8/16
// template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
// PTO_INTERNAL void TQUANT_IMPL(TileDataOut& dst, TileDataSrc& src, TileDataPara& scale, TileDataPara* offset = nullptr)
// {
//     using T = typename TileDataSrc::DType;
//     static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");

//     if constexpr (quant_type == QuantType::INT8_SYM) {
//         using U = typename TileDataOut::DType;
//         static_assert(std::is_same<U, int8_t>::value, "Fix: Quant INT8 sym: Out data type has to be int8");
//         TQuant_Int8Sym<TileDataOut, TileDataSrc, TileDataPara>(
//             dst.data(), src.data(), scale.data(), src.GetValidRow(), src.GetValidCol());
//     } else if constexpr (quant_type == QuantType::INT8_ASYM) {
//         using U = typename TileDataOut::DType;
//         static_assert(std::is_same<U, uint8_t>::value, "Fix: Quant INT8 asym: Out data type has to be uint8");
//         TQuant_Int8Asym<TileDataOut, TileDataSrc, TileDataPara>(
//             dst.data(), src.data(), scale.data(), offset->data(), src.GetValidRow(), src.GetValidCol());
//     }
// }

// // TQuant Interface for FP32/BF16/FP16->MXFP8 (ND mode)
// // E8M0, max, and scaling tiles may be passed as 2D; TQuant reshapes them to 1D internally.
// template <
//     QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
//     typename TileDataScaling>
// PTO_INTERNAL void TQUANT_IMPL(
//     TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
// {
//     using T = typename TileDataSrc::DType;
//     static_assert(
//         std::is_same<T, float32_t>::value || std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//         "Fix: Input has to be float32, bfloat16, or float16 (half)");
//     // Create 1D flat views — TQuant operates on flattened buffers internally.
//     constexpr int expN = TileDataExp::Rows * TileDataExp::Cols;
//     FlatTile1D<TileDataExp> flatExp(1, expN);
//     TRESHAPE_IMPL(flatExp, *exp);
//     constexpr int maxN = TileDataMax::Rows * TileDataMax::Cols;
//     FlatTile1D<TileDataMax> flatMax(1, maxN);
//     TRESHAPE_IMPL(flatMax, *max);
//     constexpr int scalN = TileDataScaling::Rows * TileDataScaling::Cols;
//     FlatTile1D<TileDataScaling> flatScaling(1, scalN);
//     TRESHAPE_IMPL(flatScaling, *scaling);
//     TQuant_MXFP8_Impl<
//         TileDataOut, TileDataSrc, FlatTile1D<TileDataExp>, FlatTile1D<TileDataMax>, FlatTile1D<TileDataScaling>>(
//         dst.data(), flatExp.data(), flatMax.data(), flatScaling.data(), src.data(), src.GetValidRow(),
//         src.GetValidCol());
//     // Reshape exp back to user's original tile shape. Max and scaling are scratch buffers.
//     TRESHAPE_IMPL(*exp, flatExp);
// }
// } // namespace pto
// #endif // TQUANT_HPP
