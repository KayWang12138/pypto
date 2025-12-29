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
 * \file indexadd.h
 * \brief
 */
#ifndef TILEOP_TILE_OPERATOR_INDEXADD__H
#define TILEOP_TILE_OPERATOR_INDEXADD__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

constexpr uint32_t FP32_TO_BF16_MAN_LEN = 16;

// fp32->bf16, rint mode
INLINE bfloat16_t Fp32ToBf16R(const float fVal) {
    union Bfloat16Union {
        bfloat16_t bVal;
        uint16_t bNum;
    } bf16Union = {};
    union Float32Union {
        float fVal;
        size_t fNum;
    } fp32Union;
    fp32Union.fVal = fVal;
    size_t x = fp32Union.fNum;
    // 处理特殊值
    size_t exp = x & 0x7F800000;
    if (exp == 0x7F800000) { // NaN 或无穷大
        bf16Union.bNum = static_cast<uint16_t>((x >> FP32_TO_BF16_MAN_LEN) | 0x7F80);
        return bf16Union.bVal;
    }
    if (exp == 0) { // 0或非规格化
        bf16Union.bNum = static_cast<uint16_t>((x >> FP32_TO_BF16_MAN_LEN) & 0x8000);
        return bf16Union.bVal;
    }
    // RINT舍入
    size_t lsb = (x >> FP32_TO_BF16_MAN_LEN) & 1;
    size_t roundingBit = (x >> (FP32_TO_BF16_MAN_LEN - 1)) & 1;
    size_t sticky = x & 0x7FFF;

    size_t roundUp = 0;
    if (roundingBit) {
        roundUp = (sticky != 0) ? 1 : lsb;
    }

    size_t result = (x + (roundUp << (FP32_TO_BF16_MAN_LEN - 1))) >> FP32_TO_BF16_MAN_LEN;
    // 溢出检查
    if ((result & 0x7F80) == 0x7F80) {
        result = (result & 0x8000) | 0x7F80;
    }
    bf16Union.bNum = static_cast<uint16_t>(result);
    return bf16Union.bVal;
}

// bf16->fp32
INLINE float Bf16ToFp32(const bfloat16_t bVal) {
    union Bfloat16Union {
        bfloat16_t bVal;
        uint16_t bNum;
    } bf16Union;
    union Float32Union {
        float fVal;
        size_t fNum;
    } fp32Union = {};
    bf16Union.bVal = bVal;
    fp32Union.fNum = static_cast<size_t>(bf16Union.bNum) << FP32_TO_BF16_MAN_LEN;
    return fp32Union.fVal;
}

template <typename T0, typename T2, typename dstTileDefine, typename src1TileDefine, typename Scalar, size_t dstTileW,
    size_t src1TileW>
TILEOP void IndexAddNotLastAxisCompute(dstTileDefine dstTile, src1TileDefine src1Tile, Scalar alpha,
    __ubuf__ typename T0::Type *dstAddr, __ubuf__ typename T2::Type *src1Addr, size_t dstOffset, size_t src1Offset) {
    pto::TASSIGN(dstTile, (uint64_t)(dstAddr + dstOffset));
    pto::TASSIGN(src1Tile, (uint64_t)(src1Addr + src1Offset));

    if constexpr (Std::is_same_v<Scalar, bfloat16_t>) {
        using dstTempTile = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, dstTileW * 2, pto::BLayout::RowMajor>;
        using src1TempTile = pto::Tile<pto::TileType::Vec, bfloat16_t, 1, src1TileW * 2, pto::BLayout::RowMajor>;
        dstTempTile dstTemp;
        src1TempTile src1Temp;
        pto::TASSIGN(src1Temp, (uint64_t)(src1Addr + src1Offset));
        pto::TASSIGN(dstTemp, (uint64_t)(dstAddr + dstOffset));

        if (static_cast<float>(alpha) != 1.0f) {
            pto::TMULS(src1Tile, src1Tile, alpha);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TCVT(src1Temp, src1Tile, pto::RoundMode::CAST_RINT);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
            pto::TCVT(src1Tile, src1Temp, pto::RoundMode::CAST_NONE);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
        }
        pto::TADD(dstTile, dstTile, src1Tile);
        #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
        #endif
        pto::TCVT(dstTemp, dstTile, pto::RoundMode::CAST_RINT);
        #ifdef __DAV_V220
        pipe_barrier(PIPE_V);
        #endif
        pto::TCVT(dstTile, dstTemp, pto::RoundMode::CAST_NONE);
    } else {
        if (static_cast<float>(alpha) != 1.0f) {
            pto::TMULS(src1Tile, src1Tile, alpha);
            #ifdef __DAV_V220
            pipe_barrier(PIPE_V);
            #endif
        }
        pto::TADD(dstTile, dstTile, src1Tile);
    }
}

template <typename T0, typename T2, typename T3, typename Scalar>
TILEOP void IndexAddLastAxisCompute(T0 dst, T2 src1, T3 src2, Scalar alpha, size_t src1Shape0, size_t src1Shape1,
    size_t src1Shape2, size_t src1Shape3, size_t src1Shape4, size_t dstStride0, size_t dstStride1, size_t dstStride2,
    size_t dstStride3, size_t src1Stride0, size_t src1Stride1, size_t src1Stride2, size_t src1Stride3) {
    auto dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));
    auto src1Addr = (__ubuf__ typename T2::Type *)((uint64_t)(src1.GetAddr()));
    auto idxAddr = (__ubuf__ typename T3::Type *)((uint64_t)(src2.GetAddr()));
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    uint64_t dstOffset = 0;
    uint64_t src1Offset = 0;
    if (static_cast<float>(alpha) != 1.0f) {
        for (size_t i = 0; i < src1Shape0; ++i) {
            for (size_t j = 0; j < src1Shape1; ++j) {
                for (size_t k = 0; k < src1Shape2; ++k) {
                    for (size_t l = 0; l < src1Shape3; ++l) {
                        for (size_t idx = 0; idx < src1Shape4; ++idx) {
                            auto index = *(idxAddr + idx);
                            auto dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
                            auto src1Offset =
                                i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3 + idx;
                            if constexpr (Std::is_same_v<Scalar, half>) { // half
                                Scalar mulsResult =
                                    static_cast<float>(src1Addr[src1Offset]) * static_cast<float>(alpha);
                                src1Addr[src1Offset] = mulsResult;
                            } else if constexpr (Std::is_same_v<Scalar, bfloat16_t>) { // bf16
                                float mulsResult = src1Addr[src1Offset] * Bf16ToFp32(alpha);
                                bfloat16_t mulsResBf16 = Fp32ToBf16R(mulsResult);
                                src1Addr[src1Offset] = Bf16ToFp32(mulsResBf16);

                            } else { // int8,int16,int32,float32
                                Scalar mulsResult = static_cast<Scalar>(src1Addr[src1Offset]) * alpha;
                                src1Addr[src1Offset] = static_cast<typename T2::Type>(mulsResult);
                            }
                        }
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < src1Shape0; ++i) {
        for (size_t j = 0; j < src1Shape1; ++j) {
            for (size_t k = 0; k < src1Shape2; ++k) {
                for (size_t l = 0; l < src1Shape3; ++l) {
                    for (size_t idx = 0; idx < src1Shape4; ++idx) {
                        auto index = *(idxAddr + idx);
                        auto dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + l * dstStride3 + index;
                        auto src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3 + idx;
                        if constexpr (Std::is_same_v<Scalar, half>) {
                            float addResult =
                                static_cast<float>(dstAddr[dstOffset]) + static_cast<float>(src1Addr[src1Offset]);
                            dstAddr[dstOffset] = static_cast<typename T0::Type>(addResult);
                        } else if constexpr (Std::is_same_v<Scalar, bfloat16_t>) {
                            float addResult = dstAddr[dstOffset] + src1Addr[src1Offset];
                            bfloat16_t addResBf16 = Fp32ToBf16R(addResult);
                            dstAddr[dstOffset] = Bf16ToFp32(addResBf16);
                        } else { // int8,int16,int32,float32
                            Scalar addResult =
                                static_cast<Scalar>(dstAddr[dstOffset]) + static_cast<Scalar>(src1Addr[src1Offset]);
                            dstAddr[dstOffset] = static_cast<typename T0::Type>(addResult);
                        }
                    }
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

/*
src0:self
src1:source
src2:index
axis是泛化成5维后的值，实际值为 axis + shapeSize - 5
*/
template <int axis, typename T0, typename T1, typename T2, typename T3, typename Scalar>
TILEOP void TIndexAdd(T0 dst, T1 src0, T2 src1, T3 src2, Scalar alpha) {   // T0: tileTensor
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value; // support 2-5
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<DIM_1ST, expectSize>(); // validShape
    auto dstShape1 = dstLayout.template GetShapeDim<DIM_2ND, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<DIM_3RD, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<DIM_4TH, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<DIM_5TH, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<DIM_3RD, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<DIM_4TH, expectSize>();

    const auto src1Layout = src1.GetLayout();
    auto src1Shape0 = src1Layout.template GetShapeDim<DIM_1ST, expectSize>();
    auto src1Shape1 = src1Layout.template GetShapeDim<DIM_2ND, expectSize>();
    auto src1Shape2 = src1Layout.template GetShapeDim<DIM_3RD, expectSize>();
    auto src1Shape3 = src1Layout.template GetShapeDim<DIM_4TH, expectSize>();
    auto src1Shape4 = src1Layout.template GetShapeDim<DIM_5TH, expectSize>();

    auto src1Stride0 = src1Layout.template GetStrideDim<DIM_1ST, expectSize>();
    auto src1Stride1 = src1Layout.template GetStrideDim<DIM_2ND, expectSize>();
    auto src1Stride2 = src1Layout.template GetStrideDim<DIM_3RD, expectSize>();
    auto src1Stride3 = src1Layout.template GetStrideDim<DIM_4TH, expectSize>();

    auto dstAddr = (__ubuf__ typename T0::Type *)((uint64_t)(dst.GetAddr()));
    auto src1Addr = (__ubuf__ typename T2::Type *)((uint64_t)(src1.GetAddr()));
    auto idxAddr = (__ubuf__ typename T3::Type *)((uint64_t)(src2.GetAddr()));

    if constexpr (axis == 0) { // 从第2轴开始合轴
        constexpr auto dstTileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T0::TileShape>();
        constexpr auto src1TileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T2::TileShape>();
        using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor>;
        using src1TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, src1TileW, pto::BLayout::RowMajor>;
        dstTileDefine dstTile;
        src1TileDefine src1Tile;
        for (size_t i = 0; i < src1Shape0; ++i) {
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
            auto index = *(idxAddr + i);
            set_flag(PIPE_S, PIPE_V, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
            auto dstOffset = index * dstStride0;
            auto src1Offset = i * src1Stride0;
            IndexAddNotLastAxisCompute<T0, T2, dstTileDefine, src1TileDefine, Scalar, dstTileW, src1TileW>(
                dstTile, src1Tile, alpha, dstAddr, src1Addr, dstOffset, src1Offset);
        }
    } else if constexpr (axis == 1) { // 从第3轴开始合轴
        constexpr auto dstTileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T0::TileShape>();
        constexpr auto src1TileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T2::TileShape>();
        using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor>;
        using src1TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, src1TileW, pto::BLayout::RowMajor>;
        dstTileDefine dstTile;
        src1TileDefine src1Tile;
        for (size_t i = 0; i < src1Shape0; ++i) {
            for (size_t j = 0; j < src1Shape1; ++j) {
                set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                auto index = *(idxAddr + j);
                set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                auto dstOffset = i * dstStride0 + index * dstStride1;
                auto src1Offset = i * src1Stride0 + j * src1Stride1;
                IndexAddNotLastAxisCompute<T0, T2, dstTileDefine, src1TileDefine, Scalar, dstTileW, src1TileW>(
                    dstTile, src1Tile, alpha, dstAddr, src1Addr, dstOffset, src1Offset);
            }
        }
    } else if constexpr (axis == 2) { // 从第4轴开始合轴
        constexpr auto dstTileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T0::TileShape>();
        constexpr auto src1TileW =
            TileOp::GetAnyAxisMergeResult<axis + shapeSize - 3, shapeSize, typename T2::TileShape>();
        using dstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor>;
        using src1TileDefine = pto::Tile<pto::TileType::Vec, typename T2::Type, 1, src1TileW, pto::BLayout::RowMajor>;
        dstTileDefine dstTile;
        src1TileDefine src1Tile;
        for (size_t i = 0; i < src1Shape0; ++i) {
            for (size_t j = 0; j < src1Shape1; ++j) {
                for (size_t k = 0; k < src1Shape2; ++k) {
                    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                    auto index = *(idxAddr + k);
                    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                    auto dstOffset = i * dstStride0 + j * dstStride1 + index * dstStride2;
                    auto src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2;
                    IndexAddNotLastAxisCompute<T0, T2, dstTileDefine, src1TileDefine, Scalar, dstTileW, src1TileW>(
                        dstTile, src1Tile, alpha, dstAddr, src1Addr, dstOffset, src1Offset);
                }
            }
        }
    } else if constexpr (axis == 3) {
        constexpr auto dstTileW = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;
        constexpr auto src1TileW = Std::tuple_element<shapeSize - 1, typename T2::TileShape>::type::value;
        using dstTileDefine =
            pto::Tile<pto::TileType::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
        using src1TileDefine =
            pto::Tile<pto::TileType::Vec, typename T2::Type, 1, src1TileW, pto::BLayout::RowMajor, -1, -1>;
        dstTileDefine dstTile(1, dstShape4);
        src1TileDefine src1Tile(1, src1Shape4);
        for (size_t i = 0; i < src1Shape0; ++i) {
            for (size_t j = 0; j < src1Shape1; ++j) {
                for (size_t k = 0; k < src1Shape2; ++k) {
                    for (size_t l = 0; l < src1Shape3; ++l) {
                        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
                        auto index = *(idxAddr + l);
                        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        auto dstOffset = i * dstStride0 + j * dstStride1 + k * dstStride2 + index * dstStride3;
                        auto src1Offset = i * src1Stride0 + j * src1Stride1 + k * src1Stride2 + l * src1Stride3;
                        IndexAddNotLastAxisCompute<T0, T2, dstTileDefine, src1TileDefine, Scalar, dstTileW, src1TileW>(
                            dstTile, src1Tile, alpha, dstAddr, src1Addr, dstOffset, src1Offset);
                    }
                }
            }
        }
    } else { // 尾轴
        IndexAddLastAxisCompute(dst, src1, src2, alpha, src1Shape0, src1Shape1, src1Shape2, src1Shape3, src1Shape4,
            dstStride0, dstStride1, dstStride2, dstStride3, src1Stride0, src1Stride1, src1Stride2, src1Stride3);
    }
}

#endif
