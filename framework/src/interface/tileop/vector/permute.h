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
 * \file permute.h
 * \brief
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE__H
#define TILEOP_TILE_OPERATOR_PERMUTE__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"
#include "mte.h"
#include "trans.h"

namespace {

template <typename T>
__aicore__ inline void swap(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename DType, size_t tileH, size_t tileW>
TILEOP void UBTransposeAxisImpl(
    uint64_t srcUbAddr, uint64_t dstUbAddr, uint64_t tmpUbAddr,
    size_t rows, size_t cols,
    size_t srcRowStride, size_t srcColStride,
    size_t dstRowStride, size_t dstColStride) {

    using SrcTileDefine = pto::Tile<pto::TileType::Vec, DType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using DstTileDefine = pto::Tile<pto::TileType::Vec, DType, tileH, tileW, pto::BLayout::RowMajor, -1, -1>;
    using TmpTileDefine = pto::Tile<pto::TileType::Vec, DType, tileH, tileW, pto::BLayout::RowMajor, tileH, tileW>;

    SrcTileDefine srcTile(rows, cols);
    DstTileDefine dstTile(cols, rows);
    TmpTileDefine tmpTile;

    pto::TASSIGN(srcTile, srcUbAddr);
    pto::TASSIGN(dstTile, dstUbAddr);
    pto::TASSIGN(tmpTile, tmpUbAddr);

    pto::TTRANS(dstTile, srcTile, tmpTile);

#ifdef __DAV_V220
    pipe_barrier(PIPE_V);
#endif
}

template <typename DType>
TILEOP void UBTransposeAxis(
    uint64_t srcUbAddr, uint64_t dstUbAddr, uint64_t tmpUbAddr,
    const size_t srcShape[5], const size_t srcStride[5],
    unsigned axis0, unsigned axis1,
    size_t dstShape[5], size_t dstStride[5],
    size_t tileShape3, size_t tileShape4) {

    for (unsigned d = 0; d < 5; ++d) {
        dstShape[d] = srcShape[d];
        dstStride[d] = srcStride[d];
    }
    swap(dstShape[axis0], dstShape[axis1]);
    swap(dstStride[axis0], dstStride[axis1]);

    unsigned loopDims[3];
    int idx = 0;
    for (unsigned d = 0; d < 5; ++d) {
        if (d != axis0 && d != axis1) {
            loopDims[idx++] = d;
        }
    }

    size_t rows = srcShape[axis0];
    size_t cols = srcShape[axis1];
    size_t srcRowStride = srcStride[axis1];
    size_t srcColStride = srcStride[axis0];
    size_t dstRowStride = dstStride[axis1];
    size_t dstColStride = dstStride[axis0];

    int index[3] = {0};
    bool done = false;
    while (!done) {
        size_t srcOffset = 0;
        for (int k = 0; k < 3; ++k) {
            srcOffset += index[k] * srcStride[loopDims[k]];
        }
        size_t dstOffset = 0;
        for (int k = 0; k < 3; ++k) {
            dstOffset += index[k] * dstStride[loopDims[k]];
        }

        UBTransposeAxisImpl<DType, 1, 16>(
            srcUbAddr + srcOffset * sizeof(DType),
            dstUbAddr + dstOffset * sizeof(DType),
            tmpUbAddr,
            rows, cols,
            srcRowStride, srcColStride,
            dstRowStride, dstColStride);

        for (int k = 2; k >= 0; --k) {
            index[k]++;
            if (index[k] < static_cast<int>(srcShape[loopDims[k]])) {
                break;
            } else {
                index[k] = 0;
                if (k == 0) done = true;
            }
        }
    }
}

}

#define OP_TILE_OP_PERMUTE TPermute
template <typename DST, typename SRC, typename C>
TILEOP void TPermute(DST dst, SRC src, const size_t perm[], size_t n, C coordinate,
                     uint64_t ubAddr, size_t ubSize) {
    static_assert(DST::FORMAT == Hardware::GM && SRC::FORMAT == Hardware::GM);

    using DType = typename DST::Type;
    using ActualType = std::conditional_t<std::is_same_v<DType, bool>, uint8_t, DType>;
    constexpr size_t elemSize = sizeof(ActualType);
    constexpr size_t expectSize = 5;

    const auto srcLayout = src.GetLayout();
    size_t srcShape[5] = {1, 1, 1, 1, 1};
    size_t srcStride[5] = {0, 0, 0, 0, 0};

    constexpr auto srcShapeSize = Std::tuple_size<typename SRC::Shape>::value;
    for (unsigned d = 0; d < srcShapeSize; ++d) {
        srcShape[d] = srcLayout.template GetShapeDim<d, expectSize>();
        srcStride[d] = srcLayout.template GetStrideDim<d, expectSize>();
    }

    auto srcShape0 = srcLayout.template GetShapeDim<DIM_1ST, expectSize>();
    auto srcShape1 = srcLayout.template GetShapeDim<DIM_2ND, expectSize>();
    auto srcShape2 = srcLayout.template GetShapeDim<DIM_3RD, expectSize>();

    auto srcStride0 = srcLayout.template GetStrideDim<DIM_1ST, expectSize>();
    auto srcStride1 = srcLayout.template GetStrideDim<DIM_2ND, expectSize>();
    auto srcStride2 = srcLayout.template GetStrideDim<DIM_3RD, expectSize>();

    auto gmOffset = srcLayout.template GetGmOffset<C, expectSize>(coordinate);

    constexpr auto tileShape3 = TileOp::GetTensorTileShapeDim<SRC, 3, expectSize>();
    constexpr auto tileShape4 = TileOp::GetTensorTileShapeDim<SRC, 4, expectSize>();

    constexpr size_t tmpTileW = (sizeof(ActualType) == 1) ? 32 : 16;

    auto actualTileH = (srcShapeSize >= 4) ? srcShape[3] : 1;
    auto actualTileW = (srcShapeSize >= 5) ? srcShape[4] : 1;

    size_t totalElems = 1;
    for (unsigned d = 0; d < n; ++d) {
        totalElems *= srcShape[d];
    }
    size_t totalBytes = totalElems * elemSize;
    (void)ubSize;

    uint64_t ubAddrA = ubAddr;
    uint64_t ubAddrB = ubAddr + totalBytes;
    uint64_t tmpUbAddr = ubAddr + 2 * totalBytes;

    using LoadTileDefine = pto::Tile<pto::TileType::Vec, ActualType, tileShape3, tileShape4, pto::BLayout::RowMajor, -1, -1>;
    using LoadGlobalDefine = pto::GlobalTensor<ActualType, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    LoadTileDefine loadTile;
    LoadGlobalDefine loadGlobal;

    size_t tileElemsPerBlock = actualTileH * actualTileW;

    for (LoopVar index0 = 0; index0 < srcShape0; ++index0) {
        for (LoopVar index1 = 0; index1 < srcShape1; ++index1) {
            for (LoopVar index2 = 0; index2 < srcShape2; ++index2) {
                auto offset = gmOffset + index0 * srcStride0 + index1 * srcStride1 + index2 * srcStride2;
                auto ubOffset = (index0 * srcShape1 * srcShape2 + index1 * srcShape2 + index2) * tileElemsPerBlock;
                loadGlobal.Assign(src.GetAddr() + offset);
                loadTile.Assign(ubAddrA + ubOffset * elemSize);
                pto::TLOAD(loadTile.Data(), loadGlobal.Data());
            }
        }
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);

    uint64_t activeUbAddr = ubAddrA;
    size_t currentShape[5];
    size_t currentStride[5];
    for (unsigned d = 0; d < 5; ++d) {
        currentShape[d] = srcShape[d];
        currentStride[d] = srcStride[d];
    }

    size_t inv[5];
    for (size_t i = 0; i < n; ++i) {
        inv[perm[i]] = i;
    }

    for (size_t i = 0; i < n; ++i) {
        if (inv[i] != i) {
            size_t j = i;
            while (j < n && inv[j] != i) ++j;

            uint64_t otherUbAddr = (activeUbAddr == ubAddrA) ? ubAddrB : ubAddrA;
            size_t newShape[5], newStride[5];

            UBTransposeAxis<ActualType>(
                activeUbAddr, otherUbAddr, tmpUbAddr,
                currentShape, currentStride, i, j,
                newShape, newStride,
                tileShape3, tileShape4);

            activeUbAddr = otherUbAddr;
            for (unsigned d = 0; d < 5; ++d) {
                currentShape[d] = newShape[d];
                currentStride[d] = newStride[d];
            }
            swap(inv[i], inv[j]);
        }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);

    auto dstShape0 = currentShape[0];
    auto dstShape1 = currentShape[1];
    auto dstShape2 = currentShape[2];
    auto dstStride0 = currentStride[0];
    auto dstStride1 = currentStride[1];
    auto dstStride2 = currentStride[2];

    auto dstActualTileH = (n >= 4) ? currentShape[3] : 1;
    auto dstActualTileW = (n >= 5) ? currentShape[4] : 1;
    size_t dstTileElemsPerBlock = dstActualTileH * dstActualTileW;

    const auto dstLayout = dst.GetLayout();
    auto dstGmOffset = dstLayout.template GetGmOffset<C, expectSize>(coordinate);

    using StoreTileDefine = pto::Tile<pto::TileType::Vec, ActualType, tileShape3, tileShape4, pto::BLayout::RowMajor, -1, -1>;
    using StoreGlobalDefine = pto::GlobalTensor<ActualType, pto::Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    StoreTileDefine storeTile;
    StoreGlobalDefine storeGlobal;

    for (LoopVar index0 = 0; index0 < dstShape0; ++index0) {
        for (LoopVar index1 = 0; index1 < dstShape1; ++index1) {
            for (LoopVar index2 = 0; index2 < dstShape2; ++index2) {
                auto offset = dstGmOffset + index0 * dstStride0 + index1 * dstStride1 + index2 * dstStride2;
                auto ubOffset = (index0 * dstShape1 * dstShape2 + index1 * dstShape2 + index2) * dstTileElemsPerBlock;
                storeGlobal.Assign(dst.GetAddr() + offset);
                storeTile.Assign(activeUbAddr + ubOffset * elemSize);
                pto::TSTORE(storeGlobal.Data(), storeTile.Data());
            }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_PERMUTE__H
