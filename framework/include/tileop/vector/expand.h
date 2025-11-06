/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file expand.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_EXPAND__H
#define TILEOP_TILE_OPERATOR_EXPAND__H
#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T0, typename T1>
TILEOP void TExpand(T0 dst, T1 src, unsigned axis) {
    using ShapeValueType = typename Std::tuple_element<0, typename T0::Shape>::type;
    constexpr auto shapeSize = Std::tuple_size<typename T0::Shape>::value;

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

    const auto srcLayout = src.GetLayout();
    auto srcShape0 = srcLayout.template GetShapeDim<0, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<1, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<2, expectSize>();
    auto srcShape3 = srcLayout.template GetShapeDim<3, expectSize>();
    auto srcShape4 = srcLayout.template GetShapeDim<4, expectSize>();
    auto srcStride0 = srcLayout.template GetStrideDim<0, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<1, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<2, expectSize>();

    constexpr auto typeSize = sizeof(typename T0::Type);

    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }

    constexpr auto dstTileH = Std::tuple_element<shapeSize - 2, typename T0::TileShape>::type::value;
    constexpr auto dstTileW = Std::tuple_element<shapeSize - 1, typename T0::TileShape>::type::value;

    constexpr auto srcTileH = Std::tuple_element<shapeSize - 2, typename T1::TileShape>::type::value;
    constexpr auto srcTileW = Std::tuple_element<shapeSize - 1, typename T1::TileShape>::type::value;

    if (axis == 3) {
        for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    using dstTileDefine =
                        pto::Tile<pto::Location::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
                    using srcTileDefine =
                        pto::Tile<pto::Location::Vec, typename T1::Type, srcTileH, srcTileW, pto::BLayout::RowMajor, -1, -1>;
                    dstTileDefine dstTile(dstShape3, dstShape4);
                    srcTileDefine srcTile(srcShape3, srcShape4);
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * typeSize));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * typeSize));
                    pto::TROWEXPAND(dstTile, srcTile);
                }
            }
        }
        return;
    }

    if (axis == 2) {
        for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
                for (size_t n2Index = 0; n2Index < dstShape2; ++n2Index) {
                    uint64_t blockLen = (dstShape4 * typeSize + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE;
                    auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1 + n2Index * dstStride2;
                    auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1 + n2Index * srcStride2;
                    for (unsigned i = 0; i < dstShape3; i++) {
                        copy_ubuf_to_ubuf((__ubuf__ void*)(dst.GetAddr() + (dstOffset + i * dstTileW) * typeSize),
                        (__ubuf__ void*)(src.GetAddr() + srcOffset * typeSize), 0, 1, blockLen, 1, 1);
                    }
                }
            }
        }
        return;
    }

    if (axis == 1) {
        for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
            for (size_t n1Index = 0; n1Index < dstShape1; ++n1Index) {
                uint64_t blockLen = (dstShape4 * typeSize + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE;
                uint64_t srcGap = (srcTileW * typeSize + TileOp::BLOCK_SIZE - 1)  / TileOp::BLOCK_SIZE - blockLen;
                uint64_t dstGap = (dstTileW * typeSize + TileOp::BLOCK_SIZE - 1)  / TileOp::BLOCK_SIZE - blockLen;
                auto dstOffset = n0Index * dstStride0 + n1Index * dstStride1;
                auto srcOffset = n0Index * srcStride0 + n1Index * srcStride1;
                for (unsigned i = 0; i < dstShape2; i++) {
                    copy_ubuf_to_ubuf((__ubuf__ void*)(dst.GetAddr() + (dstOffset + i * dstTileH * dstTileW) * typeSize),
                    (__ubuf__ void*)(src.GetAddr() + srcOffset * typeSize), 0, (unsigned short)dstShape3, blockLen, srcGap, dstGap);
                }
            }
        }
        return;
    }

    if (axis == 0) {
        for (size_t n0Index = 0; n0Index < dstShape0; ++n0Index) {
            uint64_t blockLen = (dstShape4 * typeSize + TileOp::BLOCK_SIZE - 1) / TileOp::BLOCK_SIZE;
            uint64_t srcGap = (srcTileW * typeSize + TileOp::BLOCK_SIZE - 1)  / TileOp::BLOCK_SIZE - blockLen;
            uint64_t dstGap = (dstTileW * typeSize + TileOp::BLOCK_SIZE - 1)  / TileOp::BLOCK_SIZE - blockLen;
            auto dstOffset = n0Index * dstStride0;
            auto srcOffset = n0Index * srcStride0;
            if constexpr (shapeSize > 2) {
                constexpr auto dstRawShape2 = Std::tuple_element<shapeSize - 3, typename T0::TileShape>::type::value;
                for (unsigned i = 0; i < dstShape1; ++i) {
                    for (unsigned j = 0; j < dstShape2; j++) {
                        copy_ubuf_to_ubuf(
                            (__ubuf__ void*)(dst.GetAddr() + (dstOffset + i * dstRawShape2 * dstTileH * dstTileW
                            + j * dstTileH * dstTileW) * typeSize),
                            (__ubuf__ void*)(src.GetAddr() + (srcOffset + j * srcTileH * srcTileW) * typeSize),
                            0, (unsigned short)dstShape3, blockLen, srcGap, dstGap);
                    }
                }
            } else {
                for (unsigned i = 0; i < dstShape1; ++i) {
                    for (unsigned j = 0; j < dstShape2; j++) {
                        copy_ubuf_to_ubuf(
                            (__ubuf__ void*)(dst.GetAddr() + (dstOffset + i * dstShape2 * dstTileH * dstTileW
                            + j * dstTileH * dstTileW) * typeSize),
                            (__ubuf__ void*)(src.GetAddr() + (srcOffset + j * srcTileH * srcTileW) * typeSize),
                            0, (unsigned short)dstShape3, blockLen, srcGap, dstGap);
                    }
                }
            }
        }
        return;
    }
}
#endif // TILEOP_TILE_OPERATOR_VEC_EXPAND__H