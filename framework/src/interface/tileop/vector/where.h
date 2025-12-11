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
 * \file where.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_WHERE__H
#define TILEOP_TILE_OPERATOR_WHERE__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

TILEOP void ProcessBoolImpl(uint64_t vcmpBitResult, uint64_t condition, uint64_t castCondition, 
                            uint64_t compareCondition, unsigned repeatNum, unsigned elementsCount) {
    set_mask_count();
    set_vector_mask(0x0, (uint64_t)elementsCount);
    pipe_barrier(PIPE_V);
    vconv_u82f16((__ubuf__ half *)castCondition, (__ubuf__ unsigned char *)condition, repeatNum, 1, 1, 8, 4);
    vector_dup((__ubuf__ half *)compareCondition, (half)1.000000e+00f, repeatNum, 1, 0, 8, 0);
    set_mask_norm();
    pipe_barrier(PIPE_V);
    vcmpv_eq((__ubuf__ unsigned char *)vcmpBitResult, (__ubuf__ half *)castCondition, 
            (__ubuf__ half *)compareCondition, repeatNum, 1, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_vector_mask(-1, -1);
}

TILEOP void ProcessBool(uint64_t vcmpBitResult, uint64_t condition, 
                        uint64_t castCondition, uint64_t compareCondition, unsigned T) {
    unsigned COUNT_MAX_BYTE = 4096;
    unsigned elementsPerCount = COUNT_MAX_BYTE / sizeof(float);
    unsigned numCountPerLine = T / elementsPerCount;
    unsigned elementsRemainPerLine = T % elementsPerCount;
    unsigned repeatNum = (elementsPerCount * sizeof(half) + 255) / 256;
    unsigned repeatNumRemain = (elementsRemainPerLine * sizeof(half) + 255) / 256;

    for (int j = 0; j < numCountPerLine; j++) {
        ProcessBoolImpl(vcmpBitResult, condition + j * elementsPerCount,
                        castCondition, compareCondition, repeatNum, elementsPerCount);
    }
    if (elementsRemainPerLine) {
        ProcessBoolImpl(vcmpBitResult, condition + elementsPerCount * numCountPerLine,
                        castCondition, compareCondition, repeatNumRemain, elementsRemainPerLine);
    }
}

template <
    typename TDst,
    typename TCastCond, typename TCmpCond,
    typename TVCmpRes,
    typename TStartAddrUB,
    typename TInpTmp, typename TOtherTmp,
    typename TCond,
    typename TSrc0, typename TSrc1>
TILEOP void TWhere(
    TDst dst,
    TCastCond castCondition,
    TCmpCond compareCondition,
    TVCmpRes vcmpBitResult,
    TStartAddrUB startAddrUB,
    TInpTmp inputTempTensor,
    TOtherTmp otherTempTensor,
    TCond condition,
    TSrc0 src0,
    TSrc1 src1) {
    using ShapeValueType = typename Std::tuple_element<0, typename TDst::Shape>::type;
    constexpr auto shapeSize = Std::tuple_size<typename TDst::Shape>::value;

    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    const auto conditionLayout = condition.GetLayout();
    auto shape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto shape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto shape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto shape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto shape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto conditionShape = condition.GetLayout().template GetShapeDim<4, expectSize>();

    auto stride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto stride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto stride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto stride3 = dstLayout.template GetStrideDim<3, expectSize>();
    auto conditionStride0 = conditionLayout.template GetStrideDim<0, expectSize>();
    auto conditionStride1 = conditionLayout.template GetStrideDim<1, expectSize>();
    auto conditionStride2 = conditionLayout.template GetStrideDim<2, expectSize>();
    auto conditionStride3 = conditionLayout.template GetStrideDim<3, expectSize>();

    constexpr auto tileH = Std::tuple_element<shapeSize - 2, typename TDst::TileShape>::type::value;
    constexpr auto tileW = Std::tuple_element<shapeSize - 1, typename TDst::TileShape>::type::value;
    constexpr auto conditionTileW = Std::tuple_element<shapeSize - 1, typename TCond::TileShape>::type::value;
    constexpr auto dstTypeSize = sizeof(typename TDst::Type);
    constexpr auto conditionTypeSize = sizeof(typename TCond::Type);
    constexpr auto src0TypeSize = sizeof(typename TSrc0::Type);
    constexpr auto src1TypeSize = sizeof(typename TSrc1::Type);

    for (size_t n0Index = 0; n0Index < shape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < shape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < shape2; ++n2Index) {
                for (size_t n3Index = 0; n3Index < shape3; ++n3Index) {
                    auto conditionOffset = n0Index * conditionStride0 + n1Index * conditionStride1 + 
                                           n2Index * conditionStride2 + n3Index * conditionStride3;
                    uint64_t conditionAddr = condition.GetAddr() + conditionOffset * conditionTypeSize;
                    if constexpr (std::is_same_v<typename TCond::Type, bool>) {
                        ProcessBool(vcmpBitResult.GetAddr(), conditionAddr, 
                                    castCondition.GetAddr(), compareCondition.GetAddr(), shape4);
                        conditionAddr = vcmpBitResult.GetAddr();
                    }

                    using TileDefine =
                        pto::Tile<pto::TileType::Vec, typename TDst::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
                    using TileDefine1 =
                        pto::Tile<pto::TileType::Vec, typename TCond::Type, 1, conditionTileW, pto::BLayout::RowMajor, -1, -1>;
                    TileDefine dstTile(1, shape4), src0Tile(1, shape4), src1Tile(1, shape4);
                    TileDefine1 conditionTile(1, conditionShape);
                    auto offset = n0Index * stride0 + n1Index * stride1 + n2Index * stride2 + n3Index * stride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + offset * dstTypeSize));
                    pto::TASSIGN(conditionTile, conditionAddr);
                    pto::TASSIGN(src0Tile, (uint64_t)(src0.GetAddr() + offset * src0TypeSize));
                    pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + offset * src1TypeSize));
                    pto::TSEL(dstTile, conditionTile, src0Tile, src1Tile);
                }
            }
        }
    }
}
#endif