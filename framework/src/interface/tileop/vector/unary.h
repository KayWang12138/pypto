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
 * \file vec_unary.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_UNARY__H
#define TILEOP_TILE_OPERATOR_VEC_UNARY__H
#include "pto_tile.h"
#include "utils/layout.h"
#include "utils/tile_tensor.h"

#include <cmath>

TILEOP void SyncV() {
#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
}

template <UnaryOp op, typename T0, typename T1>
TILEOP void UnaryComputeImpl(T0 dst, T1 src) {
    if constexpr (op == UnaryOp::EXP) {
        pto::TEXP(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::RSQRT) {
        pto::TRSQRT(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::SQRT) {
        pto::TSQRT(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::BRCB) {
        pto::TROWEXPAND(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::ABS) {
        pto::TABS(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::RECIPROCAL) {
        pto::TRECIP(dst, src);
        return;
    }
    if constexpr (op == UnaryOp::BITWISENOT) {
        pto::TNOT(dst, src);
        return;
    }
}

template<typename T, typename HalfTileDefineSrc, typename TileDefineDst, typename B16TileDefineSrc>
TILEOP void IsFiniteComputeImpl(TileDefineDst dst, B16TileDefineSrc src, HalfTileDefineSrc buffer) {
    HalfTileDefineSrc bufferFP16(src.GetValidRow(), src.GetValidCol());
    pto::TASSIGN(bufferFP16, reinterpret_cast<std::uintptr_t>(buffer.data()));

    B16TileDefineSrc bufferB16(src.GetValidRow(), src.GetValidCol());
    pto::TASSIGN(bufferB16, reinterpret_cast<std::uintptr_t>(buffer.data()));

    int16_t mask = 0;
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        mask = 0x7F80;
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float>) {
        mask = 0x7C00;
    }
    pto::TANDS(bufferB16, src, mask);
    SyncV();
    pto::TSUBS(bufferB16, bufferB16, mask);
    SyncV();
    pto::TMAXS(bufferB16, bufferB16, (int16_t) -1);
    SyncV();
    pto::TMULS(bufferB16, bufferB16, (int16_t) -1);
    SyncV();
    pto::TCVT(dst, bufferFP16, pto::RoundMode::CAST_CEIL);
    SyncV();
}

template <UnaryOp op, typename T0, typename T1>
TILEOP void UnaryCompute(T0 dst, T1 src) {
    if constexpr (TileOp::IsConstContinous<T0, T1>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data();
        auto srcTile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data();
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        UnaryComputeImpl<op>(dstTile, srcTile);
        return;
    }
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcTile = PtoTile<T1>(src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                UnaryComputeImpl<op>(dstTile.Data(), srcTile.Data());
            }
        }
    }
}

#define OP_TILE_OP_EXP TExp
template <typename T0, typename T1>
TILEOP void BrcbCompute(T0 dst, T1 src) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();
    const auto srcLayout = src.GetLayout();
    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, MAX_DIMS>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, MAX_DIMS>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, MAX_DIMS>();

    constexpr auto srcTileH = TileOp::GetTensorTileShapeDim<T1, DIM_4TH, MAX_DIMS>();
    constexpr auto srcTileW = TileOp::GetTensorTileShapeDim<T1, DIM_5TH, MAX_DIMS>();
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, DIM_4TH, MAX_DIMS>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, DIM_5TH, MAX_DIMS>();

    using DstTileDefine =pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor>;
    using SrcTileDefine = typename std::conditional<(srcTileW == 1), 
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::ColMajor>,
        pto::Tile<pto::TileType::Vec, typename T1::Type, srcTileW, srcTileH, pto::BLayout::ColMajor>>::type;

    SrcTileDefine srcTile;
    DstTileDefine dstTile;
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto dstTileOffsets = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                auto srcTileOffsets = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstTileOffsets * sizeof(typename T0::Type)));
                pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcTileOffsets * sizeof(typename T1::Type)));
                UnaryComputeImpl<UnaryOp::BRCB>(dstTile, srcTile);
            }
        }
    }
}

template <typename T0, typename T1>
TILEOP void TExp(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::EXP>(dst, src);
}

#define OP_TILE_OP_RSQRT TRsqrt
template <typename T0, typename T1>
TILEOP void TRsqrt(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::RSQRT>(dst, src);
}

#define OP_TILE_OP_SQRT TSqrt
template <typename T0, typename T1>
TILEOP void TSqrt(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::SQRT>(dst, src);
}

#define OP_TILE_OP_ISFINITE TIsFinite
template <typename DstTileTensor, typename SrcTileTensor, typename BufferTileTensor>
TILEOP void TIsFinite(DstTileTensor dst, SrcTileTensor src, BufferTileTensor buffer) {
    using DstType = std::conditional_t<std::is_same_v<typename DstTileTensor::Type, bool>, uint8_t, typename DstTileTensor::Type>;
    using SrcType = typename SrcTileTensor::Type;

    if constexpr (TileOp::IsConstContinous<DstTileTensor, SrcTileTensor>() == true) {
        constexpr size_t tileH = GetMergedAxisIfNeed<SrcTileTensor, true>();
        constexpr size_t tileW = TileOp::GetTensorTileShapeDim<SrcTileTensor, DIM_5TH, MAX_DIMS>();
        constexpr int validH = GetValidHeight<SrcTileTensor, true>();
        constexpr int validW = GetValidWidth<SrcTileTensor>();
        using TileDefineDst = pto::Tile<pto::TileType::Vec, DstType, tileH, tileW * sizeof(SrcType) / sizeof(DstType), pto::BLayout::RowMajor, validH, validW>;
        using HalfTileDefineSrc = pto::Tile<pto::TileType::Vec, half, tileH, tileW* sizeof(SrcType) / sizeof(half), pto::BLayout::RowMajor, validH, validW>;
        using B16TileDefineSrc = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW * sizeof(SrcType) / sizeof(int16_t), pto::BLayout::RowMajor, validH, validW>;

        HalfTileDefineSrc bufferTile;
        TileDefineDst dstTile;
        B16TileDefineSrc srcTile;
        pto::TASSIGN(bufferTile, buffer.GetAddr());
        pto::TASSIGN(dstTile, dst.GetAddr());
        pto::TASSIGN(srcTile, src.GetAddr());

        if constexpr (std::is_same_v<typename SrcTileTensor::Type, float>) {
            using FP32TileDefineSrc = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, validH, validW>;
            FP32TileDefineSrc srcFP32;
            HalfTileDefineSrc srcFP16;
            pto::TASSIGN(srcFP32, src.GetAddr());
            pto::TASSIGN(srcFP16, src.GetAddr());
            pto::TCVT(srcFP16, srcFP32, pto::RoundMode::CAST_NONE);
            SyncV();
        }

        IsFiniteComputeImpl<typename SrcTileTensor::Type, HalfTileDefineSrc>(dstTile, srcTile, bufferTile);
        return;
    }

    constexpr size_t tileH = GetMergedAxisIfNeed<SrcTileTensor, false>();
    constexpr size_t tileW = TileOp::GetTensorTileShapeDim<SrcTileTensor, DIM_5TH, MAX_DIMS>();
    int validH = src.GetLayout().template GetShapeDim<DIM_4TH, MAX_DIMS>();
    int validW = src.GetLayout().template GetShapeDim<DIM_5TH, MAX_DIMS>();
    using TileDefineDst = pto::Tile<pto::TileType::Vec, DstType, tileH, tileW * sizeof(SrcType) / sizeof(DstType), pto::BLayout::RowMajor, -1, -1>;
    using HalfTileDefineSrc = pto::Tile<pto::TileType::Vec, half, tileH, tileW * sizeof(SrcType) / sizeof(half), pto::BLayout::RowMajor, -1, -1>;
    using B16TileDefineSrc = pto::Tile<pto::TileType::Vec, int16_t, tileH, tileW * sizeof(SrcType) / sizeof(int16_t), pto::BLayout::RowMajor, -1, -1>;

    HalfTileDefineSrc bufferTile(validH, validW);
    pto::TASSIGN(bufferTile, buffer.GetAddr());

    TileDefineDst dstTile(validH, validW);
    B16TileDefineSrc srcTile(validH, validW);

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                pto::TASSIGN(dstTile, dst.GetAddr() + GenTileOffset(dst, tileOffsets) * sizeof(typename DstTileTensor::Type));
                pto::TASSIGN(srcTile, src.GetAddr() + GenTileOffset(src, tileOffsets) * sizeof(int16_t));
                if constexpr (std::is_same_v<typename SrcTileTensor::Type, float>) {
                    using FP32TileDefineSrc = pto::Tile<pto::TileType::Vec, float, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
                    FP32TileDefineSrc srcFP32(validH, validW);
                    HalfTileDefineSrc srcFP16(validH, validW);
                    pto::TASSIGN(srcFP32, src.GetAddr() + GenTileOffset(dst, tileOffsets) * sizeof(float));
                    pto::TASSIGN(srcFP16, src.GetAddr()+ GenTileOffset(dst, tileOffsets) * sizeof(half));
                    pto::TCVT(srcFP16, srcFP32, pto::RoundMode::CAST_NONE);
                    SyncV();
                }
                IsFiniteComputeImpl<typename SrcTileTensor::Type, HalfTileDefineSrc>(dstTile, srcTile, bufferTile);
            }
        }
    }
}

#define OP_TILE_OP_BRCB Tbrcb
template <typename T0, typename T1>
TILEOP void Tbrcb(T0 dst, T1 src) {
    BrcbCompute(dst, src);
}

#define OP_TILE_OP_ABS TAbs
template <typename T0, typename T1>
TILEOP void TAbs(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::ABS>(dst, src);
}

#define OP_TILE_OP_BITWISENOT TBitwiseNot
template <typename T0, typename T1>
TILEOP void TBitwiseNot(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::BITWISENOT>(dst, src);
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void CeilComputeImpl(T0 dst, T1 src) {
    pto::TCVT(dst, src, pto::RoundMode::CAST_CEIL);
}
#define OP_TILE_OP_CEIL TCEIL
template <typename T0, typename T1>
TILEOP void TCeil(T0 dst, T1 src) {
    if constexpr (TileOp::IsConstContinous<T0, T1>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data;
        auto srcTile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data;
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        CeilComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcTile = PtoTile<T1>(src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                CeilComputeImpl<float>(dstTile.Data(), srcTile.Data());
            }
        }
    }
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void FloorComputeImpl(T0 dst, T1 src) {
    pto::TCVT(dst, src, pto::RoundMode::CAST_FLOOR);
}
#define OP_TILE_OP_FLOOR TFLOOR
template <typename T0, typename T1>
TILEOP void TFloor(T0 dst, T1 src) {
    if constexpr (TileOp::IsConstContinous<T0, T1>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data;
        auto srcTile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data;
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        FloorComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcTile = PtoTile<T1>(src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                FloorComputeImpl<float>(dstTile.Data(), srcTile.Data());
            }
        }
    }
}

template <typename Ttemp, typename T0, typename T1>
TILEOP void TruncComputeImpl(T0 dst, T1 src) {
    pto::TCVT(dst, src, pto::RoundMode::CAST_TRUNC);
}
#define OP_TILE_OP_TRUNC TTRUNC
template <typename T0, typename T1>
TILEOP void TTrunc(T0 dst, T1 src) {
    if constexpr (TileOp::IsConstContinous<T0, T1>() == true) {
        auto dstTile = PtoTile<T0, pto::BLayout::RowMajor, true>().Data;
        auto srcTile = PtoTile<T1, pto::BLayout::RowMajor, true>().Data;
        pto::TASSIGN(dstTile, (uint64_t)dst.GetAddr());
        pto::TASSIGN(srcTile, (uint64_t)src.GetAddr());
        TruncComputeImpl<float>(dstTile, srcTile);
        return;
    }

    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto srcTile = PtoTile<T1>(src);

    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                srcTile.Assign(src, tileOffsets);
                TruncComputeImpl<float>(dstTile.Data(), srcTile.Data());
            }
        }
    }
}

#define OP_TILE_OP_ROUND TRound
template <typename Scalar, typename T0, typename T1, typename T2>
TILEOP void TRound(T0 dst, T1 tmp, T2 src, Scalar powDecimals) {
    const auto dstLayout = dst.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<DIM_1ST, MAX_DIMS>();
    auto shape1 = dstLayout.template GetShapeDim<DIM_2ND, MAX_DIMS>();
    auto shape2 = dstLayout.template GetShapeDim<DIM_3RD, MAX_DIMS>();

    auto dstTile = PtoTile<T0>(dst);
    auto tmpTile = PtoTile<T1>(tmp);
    auto srcTile = PtoTile<T2>(src);
    for (LoopVar n0Index = 0; n0Index < shape0; ++n0Index) {
        for (LoopVar n1Index = 0; n1Index < shape1; ++n1Index) {
            for (LoopVar n2Index = 0; n2Index < shape2; ++n2Index) {
                auto tileOffsets = TileOffset(n0Index, n1Index, n2Index);
                dstTile.Assign(dst, tileOffsets);
                tmpTile.Assign(tmp, tileOffsets);
                srcTile.Assign(src, tileOffsets);

                if constexpr (std::is_same_v<typename T2::Type, float>) {
                    pto::TMULS(srcTile.Data(), srcTile.Data(), powDecimals);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TCVT(srcTile.Data(), srcTile.Data(), pto::RoundMode::CAST_RINT);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TDIVS(dstTile.Data(), srcTile.Data(), powDecimals);
                } else {
                    pto::TCVT(tmpTile.Data(), srcTile.Data(), pto::RoundMode::CAST_NONE);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TMULS(tmpTile.Data(), tmpTile.Data(), powDecimals);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TCVT(tmpTile.Data(), tmpTile.Data(), pto::RoundMode::CAST_RINT);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TMULS(tmpTile.Data(), tmpTile.Data(), 1.0f / powDecimals);
#ifdef __DAV_V220
                    pipe_barrier(PIPE_V);
#endif
                    pto::TCVT(dstTile.Data(), tmpTile.Data(), pto::RoundMode::CAST_RINT);
                }
            }
        }
    }
}

#define OP_TILE_OP_RECIPROCAL TReciprocal
template <typename T0, typename T1>
TILEOP void TReciprocal(T0 dst, T1 src) {
    UnaryCompute<UnaryOp::RECIPROCAL>(dst, src);
}
#endif