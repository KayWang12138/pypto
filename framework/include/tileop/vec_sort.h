/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vec_sort.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_VEC_SORT__H
#define TILEOP_TILE_OPERATOR_VEC_SORT__H
#include "a2a3/utils/layout.h"
#include "a2a3/utils/tile_tensor.h"

template <int axis, int offset, int isLargest, typename T0, typename T1>
TILEOP void TBitSort(T0 dst, T1 src) {
    using ShapeValueType = typename Std::tuple_element<0, typename T1::Shape>::type;
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto tmpShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto tmpTileH = Std::tuple_element<tmpShapeSize - 2, typename T1::TileShape>::type::value;
    constexpr auto tmpTileW = Std::tuple_element<tmpShapeSize - 1, typename T1::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<tmpShapeSize - 1, typename T0::TileShape>::type::value / 2;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    constexpr auto dstTileH = Std::tuple_element<dstShapeSize - 2, typename T0::TileShape>::type::value;

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();
    constexpr auto srcTileH = Std::tuple_element<srcShapeSize - 2, typename T1::TileShape>::type::value;
    constexpr auto srcTileW = Std::tuple_element<srcShapeSize - 1, typename T1::TileShape>::type::value;
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (size_t n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    using DstTileDefine =
                        pto::Tile<pto::Location::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileDefine =
                        pto::Tile<pto::Location::Vec, typename T1::Type, 1, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                    using IdxTileDefine =
                        pto::Tile<pto::Location::Vec, uint32_t, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    using TmpTileDefine =
                        pto::Tile<pto::Location::Vec, typename T1::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    DstTileDefine dstTile(1, dstShape4);
                    SrcTileDefine srcTile(1, srcShape4);
                    IdxTileDefine idxTile(1, srcShape4);
                    TmpTileDefine tmpTile(1, srcTileW);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 +
                     n2Index * dstStride2 + n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 +
                     n2Index * srcStride2 + n3Index * srcStride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(dst.GetAddr() + (dstOffset + srcTileW + dstTileW) * srcTypeSize));
                    pto::TASSIGN(idxTile, (uint64_t)(dst.GetAddr() + (dstOffset + dstTileW) * srcTypeSize));
                    for (uint32_t j = 0; j < srcShape4; ++j) {
                        *(idxTile.data() + j) = (j + offset);
                    }
                    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                    if constexpr (isLargest == 0) {
                        set_mask_count();
                        set_vector_mask(0, srcShape4);
                        vadds((__ubuf__ int32_t *)((uint64_t)(src.GetAddr() + srcOffset * srcTypeSize)),
                        (__ubuf__ int32_t *)((uint64_t)(src.GetAddr() + srcOffset * srcTypeSize)), 0x80000000, 1, 1, 1, 8, 8);
                        pipe_barrier(PIPE_V);
                        set_mask_norm();
                        set_vector_mask(-1, -1);
                    }
                    pto::TSORT32(dstTile, srcTile, idxTile, tmpTile);
                }
            }
        }
    }
}

template <int axis, int k, int isLargest, typename T0, typename T1>
TILEOP void MrgSort(T0 dst, T1 src) {
    constexpr auto srcShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    constexpr auto dstTileH = Std::tuple_element<dstShapeSize - 2, typename T0::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<dstShapeSize - 1, typename T0::TileShape>::type::value;

    const auto srcLayout = src.GetLayout();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();
    auto srcStride3 = srcLayout.template GetStrideDim<3, expectSize>();
    constexpr auto srcTileH = Std::tuple_element<srcShapeSize - 2, typename T1::TileShape>::type::value;
    constexpr auto srcTileW = Std::tuple_element<srcShapeSize - 1, typename T1::TileShape>::type::value / 2;
    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (size_t n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    using DstTileDefine =
                        pto::Tile<pto::Location::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    using SrcTileDefine =
                        pto::Tile<pto::Location::Vec, typename T1::Type, 1, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                    using TmpTileDefine =
                        pto::Tile<pto::Location::Vec, typename T1::Type, 1, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                    DstTileDefine dstTile(1, dstShape4);
                    SrcTileDefine srcTile(1, srcShape4 / 2);
                    TmpTileDefine tmpTile(1, srcTileW);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 +
                     n2Index * dstStride2 + n3Index * dstStride3;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 +
                     n2Index * srcStride2 + n3Index * srcStride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(src.GetAddr() + (srcOffset + srcTileW) * srcTypeSize));
                    uint32_t z = 32;
                    uint32_t totalNum = srcShape4 / 4;
                    for (; z * 4 <= totalNum; z *= 4) {
                        uint32_t repeat_mrg = totalNum / (z * 4);
                        pto::TMRGSORT(tmpTile, srcTile, z * 2);
                        pipe_barrier(PIPE_V);
                        copy_ubuf_to_ubuf(
                            (__ubuf__ void *)((uint64_t)(src.GetAddr() + srcOffset * srcTypeSize)),
                            (__ubuf__ void *)((uint64_t)(src.GetAddr() + (srcOffset + srcTileW) * srcTypeSize)),
                            0, 1, z * repeat_mrg, 0, 0);
                        pipe_barrier(PIPE_V);
                    }
                    if (z < totalNum) {
                        int32_t arrayCount = 0;
                        int32_t mrgArray[15] = {0};
                        int32_t tmpInner = totalNum;
                        for (int32_t i = z; i >= 32; i /= 4) {
                            int32_t count;
                            for (count = 0; count < tmpInner / i; count++) {
                                mrgArray[arrayCount++] = i;
                            }
                            tmpInner -= count * i;
                        }
                        uint16_t mrgSortedLen = 0;
                        for (int32_t i = 0; i < arrayCount - 1; ++i) {
                            mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
                            uint64_t tmpMrgSortedLen = mrgSortedLen;
                            uint64_t tmpMrgArray = mrgArray[i + 1];
                            if (mrgSortedLen > k) {
                                tmpMrgSortedLen = k;
                            }
                            if (mrgArray[i + 1] > k) {
                                tmpMrgArray = k;
                            }
                            SrcTileDefine src1Tile(1, tmpMrgSortedLen * 2), src2Tile(1, tmpMrgArray * 2);
                            TmpTileDefine tmp1Tile(1, (tmpMrgSortedLen + tmpMrgArray) * 2);
                            DstTileDefine dst1Tile(1, (tmpMrgSortedLen + tmpMrgArray) * 2);
                            pto::TASSIGN(src1Tile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                            pto::TASSIGN(src2Tile, (uint64_t)(src.GetAddr() + (srcOffset + mrgSortedLen * 2) * srcTypeSize));
                            pto::TASSIGN(tmp1Tile, (uint64_t)(src.GetAddr() + (srcOffset + srcTileW) * srcTypeSize));
                            pto::TASSIGN(dst1Tile, (uint64_t)(src.GetAddr() + srcOffset * srcTypeSize));
                            pto::MrgSortExecutedNumList executedNumList;
                            pto::TMRGSORT<DstTileDefine, TmpTileDefine, SrcTileDefine, SrcTileDefine, false>(dst1Tile, 
                                executedNumList, tmp1Tile, src1Tile, src2Tile);
                            pipe_barrier(PIPE_V);
                        }
                    }
                    copy_ubuf_to_ubuf(
                            (__ubuf__ void *)((uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize)),
                            (__ubuf__ void *)((uint64_t)(src.GetAddr() + srcOffset * srcTypeSize)),
                            0, 1, (k + 7) / 4, 0, 0);
                    pipe_barrier(PIPE_V);
                }
            }
        }
    }
}

template <int k, int validBit, typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
TILEOP void TiledMrgSort(T0 dst, T1 src1, T2 src2, T3 src3, T4 src4, T5 tmp) {
    constexpr auto tmpShapeSize = Std::tuple_size<typename T5::Shape>::value;
    constexpr auto dstShapeSize = Std::tuple_size<typename T0::Shape>::value;
    constexpr auto src1ShapeSize = Std::tuple_size<typename T1::Shape>::value;
    constexpr auto src4ShapeSize = Std::tuple_size<typename T4::Shape>::value;
    constexpr auto tmpTileH = Std::tuple_element<tmpShapeSize - 2, typename T5::TileShape>::type::value;
    constexpr auto tmpTileW = Std::tuple_element<tmpShapeSize - 1, typename T5::TileShape>::type::value;
    constexpr size_t expectSize = 5;
    const auto dstLayout = dst.GetLayout();
    auto dstShape0 = dstLayout.template GetShapeDim<0, expectSize>();
    auto dstShape1 = dstLayout.template GetShapeDim<1, expectSize>();
    auto dstShape2 = dstLayout.template GetShapeDim<2, expectSize>();
    auto dstShape3 = dstLayout.template GetShapeDim<3, expectSize>();
    auto dstShape4 = dstLayout.template GetShapeDim<4, expectSize>();
    auto dstStride0 = dstLayout.template GetStrideDim<0, expectSize>();
    auto dstStride1 = dstLayout.template GetStrideDim<1, expectSize>();
    auto dstStride2 = dstLayout.template GetStrideDim<2, expectSize>();
    auto dstStride3 = dstLayout.template GetStrideDim<3, expectSize>();
    constexpr auto dstTileH = Std::tuple_element<dstShapeSize - 2, typename T0::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<dstShapeSize - 1, typename T0::TileShape>::type::value;

    const auto src1Layout = src1.GetLayout();
    auto src1Shape3 = src1Layout.template GetShapeDim<3, expectSize>();
    auto src1Shape4 = src1Layout.template GetShapeDim<4, expectSize>();
    auto src1Stride0 = src1Layout.template GetStrideDim<0, expectSize>();
    auto src1Stride1 = src1Layout.template GetStrideDim<1, expectSize>();
    auto src1Stride2 = src1Layout.template GetStrideDim<2, expectSize>();
    auto src1Stride3 = src1Layout.template GetStrideDim<3, expectSize>();
    constexpr auto src1TileW = Std::tuple_element<src1ShapeSize - 1, typename T1::TileShape>::type::value;

    const auto src4Layout = src4.GetLayout();
    auto src4Shape3 = src4Layout.template GetShapeDim<3, expectSize>();
    auto src4Shape4 = src4Layout.template GetShapeDim<4, expectSize>();
    auto src4Stride0 = src4Layout.template GetStrideDim<0, expectSize>();
    auto src4Stride1 = src4Layout.template GetStrideDim<1, expectSize>();
    auto src4Stride2 = src4Layout.template GetStrideDim<2, expectSize>();
    auto src4Stride3 = src4Layout.template GetStrideDim<3, expectSize>();
    constexpr auto src4TileW = Std::tuple_element<src4ShapeSize - 1, typename T4::TileShape>::type::value;

    constexpr auto srcTypeSize = sizeof(typename T1::Type);
    for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
        for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
            for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                for (size_t n3Index = 0; n3Index < dstShape3; ++n3Index) {
                    using DstTileDefine =
                        pto::Tile<pto::Location::Vec, typename T0::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    using Src1TileDefine =
                        pto::Tile<pto::Location::Vec, typename T1::Type, 1, src1TileW, pto::BLayout::RowMajor, -1, -1>;
                    using Src4TileDefine =
                        pto::Tile<pto::Location::Vec, typename T4::Type, 1, src4TileW, pto::BLayout::RowMajor, -1, -1>;
                    using TmpTileDefine =
                        pto::Tile<pto::Location::Vec, typename T5::Type, 1, tmpTileW, pto::BLayout::RowMajor, 1, tmpTileW>;
                    DstTileDefine dstTile(1, dstShape4);
                    TmpTileDefine tmpTile;
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 +
                     n2Index * dstStride2 + n3Index * dstStride3;
                    auto src1Offset = n0Index * src1Stride0 + n1Index * src1Stride1 +
                     n2Index * src1Stride2 + n3Index * src1Stride3;
                    auto src4Offset = n0Index * src4Stride0 + n1Index * src4Stride1 +
                     n2Index * src4Stride2 + n3Index * src4Stride3;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * srcTypeSize));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));                   
                    pto::MrgSortExecutedNumList executedNumList;     
                    if constexpr (validBit == 2) {
                        Src1TileDefine src1Tile(1, src1Shape4);
                        Src4TileDefine src2Tile(1, src4Shape4);
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src2Tile, (uint64_t)(src2.GetAddr() + src4Offset * srcTypeSize));
                        pto::TMRGSORT<DstTileDefine, TmpTileDefine, Src1TileDefine, Src4TileDefine, false>(dstTile, 
                                executedNumList, tmpTile, src1Tile, src2Tile);
                    } else if constexpr (validBit == 3) {
                        Src1TileDefine src1Tile(1, src1Shape4);
                        Src1TileDefine src2Tile(1, src1Shape4);
                        Src4TileDefine src3Tile(1, src4Shape4);
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src2Tile, (uint64_t)(src2.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src3Tile, (uint64_t)(src3.GetAddr() + src4Offset * srcTypeSize));
                        pto::TMRGSORT<DstTileDefine, TmpTileDefine, Src1TileDefine, Src1TileDefine, Src4TileDefine, false>(dstTile, 
                                executedNumList, tmpTile, src1Tile, src2Tile, src3Tile);
                    } else if constexpr (validBit == 4) {
                        Src1TileDefine src1Tile(1, src1Shape4);
                        Src1TileDefine src2Tile(1, src1Shape4);
                        Src1TileDefine src3Tile(1, src1Shape4);
                        Src4TileDefine src4Tile(1, src4Shape4);
                        pto::TASSIGN(src1Tile, (uint64_t)(src1.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src2Tile, (uint64_t)(src2.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src3Tile, (uint64_t)(src3.GetAddr() + src1Offset * srcTypeSize));
                        pto::TASSIGN(src4Tile, (uint64_t)(src4.GetAddr() + src4Offset * srcTypeSize));
                        pto::TMRGSORT<DstTileDefine, TmpTileDefine, Src1TileDefine, Src1TileDefine, Src1TileDefine, Src4TileDefine, false>(dstTile, 
                                executedNumList, tmpTile, src1Tile, src2Tile, src3Tile, src4Tile);
                    }
                }
            }
        }
    }
}
#endif