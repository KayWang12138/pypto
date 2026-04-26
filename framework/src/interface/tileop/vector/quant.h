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
 * \file quant.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_QUANT__H
#define TILEOP_TILE_OPERATOR_QUANT__H

#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

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
    using ExpByteTile = pto::Tile<
        pto::TileType::Vec, uint8_t, expTileH, expTileW, pto::BLayout::RowMajor, -1, -1>;
    using MaxDtype = std::conditional_t<std::is_same_v<typename T2::Type, bool>, uint8_t, typename T2::Type>;
    using MaxTile = pto::Tile<
        pto::TileType::Vec, MaxDtype, maxTileH, maxTileW, pto::BLayout::RowMajor, -1, -1>;
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
                auto expTileOffset =
                    GetQuantMXPerformanceGroupedOffset<T4>(expLayout, n0Index, n1Index, n2Index);
                auto maxTileOffset =
                    GetQuantMXPerformanceGroupedOffset<T4>(maxLayout, n0Index, n1Index, n2Index);
                auto srcTileAddr =
                    (uint64_t)(src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(typename T4::Type));
                dstTile.Assign(dst, tileOffsets);
                scalingTile.Assign(scalingScratch, tileOffsets);
                srcTile.Assign(srcTileAddr);
                pto::TASSIGN(expByteTile, (uint64_t)(exp.GetAddr() + expTileOffset * sizeof(typename T1::Type)));
                pto::TASSIGN(
                    maxTile, (uint64_t)(maxScratch.GetAddr() + maxTileOffset * sizeof(typename T2::Type)));
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

#endif



// // Computing scalar focus and exponent for B16 (BF16/FP16) -> b8 e4m3 quantization.
// // Compile-time constants are selected based on the source data type T (bfloat16_t or half).
// //   BF16: shr=7,  exp_mask=0x7F80, nan_check=0xFF,  exp_max=0xFE, subnorm=0x7F80, clamp=-127
// //   FP16: shr=10, exp_mask=0x7C00, nan_check=0x1F,  exp_max=0x1E, subnorm=0x7C00, clamp=-15
// // E8M0 uses bias 127. For BF16 (bias 127) emax_e8m0=8. For FP16 (bias 15) we subtract
// // the bias difference: emax_e8m0 = 8 - (127 - 15) = -104, so E8M0 = biased_fp16 + 104.
// // For BF16 values whose ideal shared scale would underflow below E8M0's minimum normal value,
// // shared_exp is clamped to 0 (shared scale = 2^-127) and the reciprocal scaling used by the
// // data path is clamped to 2^127. For FP16, shared_exp uses the same E8M0 semantics, but the
// // reciprocal scaling scratch is stored as BF16-encoded 16-bit values so large powers of two do
// // not overflow the FP16 range before the later FP32 multiply.
// template <typename T>
// PTO_INTERNAL void ExtractB8ExponentAndScaling(__ubuf__ T *maxPtr, __ubuf__ uint8_t *expPtr, __ubuf__ T *scalingPtr,
//                                               unsigned exp_max_loop_count, unsigned total_elements_count)
// {
//     static_assert(std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//                   "ExtractB8ExponentAndScaling B16: T must be bfloat16_t or half");
//     static constexpr auto distValue =
//         std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
//     // Compile-time format-specific constants
//     constexpr bool is_bf16 = std::is_same<T, bfloat16_t>::value;
//     constexpr int shr = is_bf16 ? 7 : 10;                       // mantissa bits
//     constexpr int16_t exp_mask_val = is_bf16 ? 0x7F80 : 0x7C00; // exponent field mask
//     constexpr int16_t nan_check = is_bf16 ? 0xFF : 0x1F;        // all-ones exponent (NaN/Inf)
//     constexpr int16_t exp_max_val = is_bf16 ? 0xFE : 0x1E;      // max non-Inf biased exponent
//     constexpr int16_t subnorm_val = is_bf16 ? 0x7F80 : 0x7C00;  // +Inf sentinel for clamping
//     constexpr int16_t clamp_val = is_bf16 ? -127 : -15;         // negative bias (clamping threshold)
//     constexpr int16_t min_e8m0_exp_threshold = 8;
//     constexpr int16_t min_e8m0_shared_exp = 0;
//     constexpr int16_t recip_min_scale_val = 0x7F00; // bf16 2^127
//     constexpr int16_t recip_scale_base = 0xFE;      // bf16 biased exponent for reciprocal scale construction
//     constexpr int16_t bf16_nan_val = 0x7F81;        // bf16 qNaN payload used for internal scale scratch

//     constexpr int16_t emax_e8m0 = is_bf16 ? 8 : (8 - 112);
//     RegTensor<T> vb16_max;
//     vector_s16 vb16_exponent, vb16_shared_exp, vb16_scaling, vb16_subnorm;
//     vector_s16 vb16_b8_nan, vb16_bf16_nan, vb16_zero, vb16_b8_emax, vb16_exp_mask, vb16_recip_scale_base,
//         vb16_recip_min_scale, vb16_min_e8m0_shared_exp;
//     vbr(vb16_exp_mask, exp_mask_val);
//     vbr(vb16_b8_nan, 0xFF);
//     vbr(vb16_bf16_nan, bf16_nan_val);
//     vbr(vb16_zero, 0);
//     vbr(vb16_subnorm, subnorm_val);
//     vbr(vb16_recip_scale_base, recip_scale_base);
//     vbr(vb16_b8_emax, emax_e8m0);
//     vbr(vb16_recip_min_scale, recip_min_scale_val);
//     vbr(vb16_min_e8m0_shared_exp, min_e8m0_shared_exp);
//     vector_bool preg_special, preg_zero, preg_no_scale;
//     constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
//     uint32_t total_count = total_elements_count;
//     for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
//         vector_bool preg_b16 = CreatePredicate<T>(total_count);
//         vlds(vb16_max, maxPtr, i * elementsPerVL, NORM);
//         // Getting biased exponent
//         vand((vector_s16 &)vb16_exponent, (vector_s16 &)vb16_max, vb16_exp_mask, preg_b16, MODE_ZEROING);
//         vshrs((vector_s16 &)vb16_exponent, (vector_s16 &)vb16_exponent, shr, preg_b16, MODE_ZEROING);
//         // E8M0: shared_exp = exponent - emax_e8m0 (bias-corrected for FP16)
//         vsub((vector_s16 &)vb16_shared_exp, (vector_s16 &)vb16_exponent, (vector_s16 &)vb16_b8_emax, preg_b16); //
//         if constexpr (is_bf16) {
//             vcmps_le(preg_no_scale, (vector_s16 &)vb16_exponent, min_e8m0_exp_threshold, preg_b16);
//             vsel(vb16_shared_exp, vb16_min_e8m0_shared_exp, vb16_shared_exp, preg_no_scale);
//         }
//         // Reciprocal scaling is materialized in BF16 encoding for both BF16 and FP16 paths.
//         vsub((vector_s16 &)vb16_scaling, (vector_s16 &)vb16_recip_scale_base, (vector_s16 &)vb16_shared_exp, preg_b16);
//         vshls((vector_s16 &)vb16_scaling, (vector_s16 &)vb16_scaling, 7, preg_b16, MODE_ZEROING);
//         if constexpr (is_bf16) {
//             vsel(vb16_scaling, vb16_recip_min_scale, vb16_scaling, preg_no_scale);
//         }
//         // Match dynamic_mx_quant_tail_axis_fp8.h semantics:
//         // zero block   -> exp = 0,    reciprocal scale = 0
//         // inf/nan block-> exp = 0xFF, reciprocal scale = bf16 NaN
//         vcmps_eq(preg_zero, (vector_s16 &)vb16_max, 0, preg_b16);
//         vsel(vb16_scaling, vb16_zero, vb16_scaling, preg_zero);
//         vsel(vb16_shared_exp, vb16_zero, vb16_shared_exp, preg_zero);

//         vcmps_eq(preg_special, (vector_s16 &)vb16_exponent, nan_check, preg_b16);
//         vsel(vb16_scaling, vb16_bf16_nan, vb16_scaling, preg_special);
//         vsel(vb16_shared_exp, vb16_b8_nan, vb16_shared_exp, preg_special);

//         vsts((vector_s16 &)vb16_shared_exp, ((__ubuf__ int16_t *)expPtr), i * elementsPerVL / sizeof(T), PK_B16,
//              preg_b16);
//         vsts((vector_s16 &)vb16_scaling, ((__ubuf__ int16_t *)scalingPtr), i * elementsPerVL, distValue, preg_b16);
//     }
// }

// // B16 (BF16/FP16) -> FP8 (No direct b16->e4m3 support; convert up to fp32 then down to fp8)
// // Uses RegTensor<T> to dispatch the correct vector type (vector_bf16 or vector_f16).
// template <typename T>
// PTO_INTERNAL void CalcQuantizedFP8Values(__ubuf__ T *srcPtr, __ubuf__ T *scalingPtr, __ubuf__ uint8_t *dstPtr,
//                                          unsigned total_elements_count)
// {
//     static_assert(std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
//                   "CalcQuantizedFP8Values B16: T must be bfloat16_t or half");
//     RegTensor<T> vb16_scaling, vb16_in_1, vb16_in_2, vb16_out_1, vb16_out_2;
//     RegTensor<bfloat16_t> vb16_scaling_bf16;
//     vector_f32 vb32_cvt_1, vb32_cvt_2, vb32_cvt_3, vb32_cvt_4, vb32_scale_even, vb32_scale_odd;
//     vector_f8e4m3 vb8_or1, vb8_or2, vb8_out, vb8_p0, vb8_p1, vb8_p2, vb8_p3;
//     constexpr uint32_t elementsPerVL_b16 = REPEAT_BYTE / sizeof(T);
//     constexpr uint32_t elementsPerVL_b8 = REPEAT_BYTE / sizeof(uint8_t);
//     constexpr uint32_t elementsPerDintlv = 2 * elementsPerVL_b16; // 256 bf16 per DINTLV load
//     uint32_t vl_count = CeilDivision(total_elements_count, elementsPerVL_b16);
//     for (uint16_t i = 0; i < (uint16_t)vl_count / 2; ++i) {
//         // DINTLV_B16 deinterleaves 256 elements into even/odd registers.
//         // Predicates must reflect per-register valid count (half of remaining),
//         // NOT sequential 128-subtraction which assumes linear VL consumption.
//         uint32_t offset_b16 = i * elementsPerDintlv;
//         uint32_t remaining = (total_elements_count > offset_b16) ? (total_elements_count - offset_b16) : 0;
//         if (remaining > elementsPerDintlv)
//             remaining = elementsPerDintlv;
//         uint32_t even_count = (remaining + 1) / 2;
//         uint32_t odd_count = remaining / 2;
//         MaskReg preg_b16_1 = CreatePredicate<T>(even_count);
//         MaskReg preg_b16_2 = CreatePredicate<T>(odd_count);
//         uint32_t remaining_b8 = remaining; // 1 bf16 → 1 FP8 byte
//         MaskReg preg_b8 = CreatePredicate<uint8_t>(remaining_b8);
//         vlds(vb16_in_1, vb16_in_2, srcPtr, offset_b16, DINTLV_B16);
//         if constexpr (std::is_same<T, half>::value) {
//             // FP16 uses BF16-encoded reciprocal scales to avoid overflow on large powers of two.
//             vlds((vector_u16 &)vb16_scaling_bf16, (__ubuf__ uint16_t *)scalingPtr, 8 * i, E2B_B16);
//             vcvt(vb32_cvt_1, vb16_in_1, preg_b16_1, PART_EVEN); // indices mod 4 = 0: [b0, b4, b8, ...]
//             vcvt(vb32_cvt_2, vb16_in_1, preg_b16_1, PART_ODD);  // indices mod 4 = 2: [b2, b6, b10, ...]
//             vcvt(vb32_cvt_3, vb16_in_2, preg_b16_2, PART_EVEN); // indices mod 4 = 1: [b1, b5, b9, ...]
//             vcvt(vb32_cvt_4, vb16_in_2, preg_b16_2, PART_ODD);  // indices mod 4 = 3: [b3, b7, b11, ...]
//             vcvt(vb32_scale_even, vb16_scaling_bf16, preg_b16_1, PART_EVEN);
//             vcvt(vb32_scale_odd, vb16_scaling_bf16, preg_b16_1, PART_ODD);
//             vmul(vb32_cvt_1, vb32_cvt_1, vb32_scale_even, preg_b16_1, MODE_ZEROING);
//             vmul(vb32_cvt_2, vb32_cvt_2, vb32_scale_odd, preg_b16_1, MODE_ZEROING);
//             vmul(vb32_cvt_3, vb32_cvt_3, vb32_scale_even, preg_b16_2, MODE_ZEROING);
//             vmul(vb32_cvt_4, vb32_cvt_4, vb32_scale_odd, preg_b16_2, MODE_ZEROING);
//         } else {
//             vlds((vector_u16 &)vb16_scaling, (__ubuf__ uint16_t *)scalingPtr, 8 * i, E2B_B16);
//             vmul(vb16_out_1, vb16_in_1, vb16_scaling, preg_b16_1, MODE_ZEROING);
//             vmul(vb16_out_2, vb16_in_2, vb16_scaling, preg_b16_2, MODE_ZEROING);
//             // b16->fp32: EVEN/ODD split each 128-element b16 register into 2x64 fp32
//             vcvt(vb32_cvt_1, vb16_out_1, preg_b16_1, PART_EVEN); // indices mod 4 = 0: [b0, b4, b8, ...]
//             vcvt(vb32_cvt_2, vb16_out_1, preg_b16_1, PART_ODD);  // indices mod 4 = 2: [b2, b6, b10, ...]
//             vcvt(vb32_cvt_3, vb16_out_2, preg_b16_2, PART_EVEN); // indices mod 4 = 1: [b1, b5, b9, ...]
//             vcvt(vb32_cvt_4, vb16_out_2, preg_b16_2, PART_ODD);  // indices mod 4 = 3: [b3, b7, b11, ...]
//         }
//         // fp32 -> fp8: P0-P3 place fp8 bytes at byte 0-3 of each 32-bit slot
//         // Must map: P0=mod0, P1=mod1, P2=mod2, P3=mod3 for correct sequential output
//         vcvt(vb8_p0, vb32_cvt_1, preg_b16_1, ROUND_R, RS_ENABLE, PART_P0); // mod 0 → byte 0
//         vcvt(vb8_p1, vb32_cvt_3, preg_b16_2, ROUND_R, RS_ENABLE, PART_P1); // mod 1 → byte 1
//         vcvt(vb8_p2, vb32_cvt_2, preg_b16_1, ROUND_R, RS_ENABLE, PART_P2); // mod 2 → byte 2
//         vcvt(vb8_p3, vb32_cvt_4, preg_b16_2, ROUND_R, RS_ENABLE, PART_P3); // mod 3 → byte 3

//         vor(vb8_or1, vb8_p0, vb8_p1, preg_b8);
//         vor(vb8_or2, vb8_p2, vb8_p3, preg_b8);
//         vor(vb8_out, vb8_or1, vb8_or2, preg_b8);
//         vsts((vector_u8 &)vb8_out, (__ubuf__ uint8_t *)dstPtr, i * elementsPerVL_b8, NORM_B8, preg_b8);
//     }
// }
