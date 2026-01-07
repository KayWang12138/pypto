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
 * \file index_outcast.h
 * \brief
 */

template <typename T, typename T2>
TILEOP void DynTIndexoutcast(__gm__ T *dst, __ubuf__ T *src, __ubuf__ T2 *index, unsigned src1OriShape0,
    unsigned src1OriShape1, unsigned src1rawShape1, unsigned src0OriShape3, unsigned src0rawShape1, unsigned src0rawShape3) {
    unsigned b = src1OriShape0;
    unsigned s1 = src1OriShape1;
    unsigned s1_32aligned = src1rawShape1;
    unsigned nd = src0OriShape3;
    unsigned nd_32aligned = src0rawShape3;

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    __gm__ T *curDst = dst;
    __ubuf__ T2 *dstIdx = index;
    __ubuf__ T *curSrc = src;
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < s1; ++j) {
            curDst = dst +  *dstIdx * nd;                                        // dst [index[i][j]] [n][d]
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            copy_ubuf_to_gm_align_b32(
                curDst, curSrc, 0 /*sid*/, 1, nd * sizeof(T), 0, 0, (nd_32aligned - nd) * sizeof(T) / BLOCK_SIZE, 0);
            curSrc += nd_32aligned;
            dstIdx++;
        }
        curSrc += (src0rawShape1 - s1) * nd_32aligned;
        dstIdx += s1_32aligned - s1;
    }
}


template <typename T, typename T2, unsigned src0rawShape1, unsigned cacheMode, unsigned blockSize>
TILEOP void TIndexoutcastBase(__gm__ T *dst, __ubuf__ T *src0, __ubuf__ T2 *src1, unsigned src0OriShape1,
    unsigned src1OriShape1, unsigned GmShape1) {
    for (auto i = 0; i < src1OriShape1; i++) {
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7); 
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        set_flag(PIPE_V, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
        T2 curValue = *(reinterpret_cast<__ubuf__ T2 *>(src1 + i));
        if constexpr (cacheMode == 1) { // PA_NZ
            T2 blockCount = curValue / blockSize;
            T2 index = curValue % blockSize;

            __gm__ T *new_dst = dst + blockCount * blockSize * GmShape1 + index * 32 / sizeof(T);
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

            copy_ubuf_to_gm(new_dst, src0 + i * src0OriShape1, 0 /*sid*/, src0OriShape1 / 32 * sizeof(T), 1, 0, blockSize - 1);//ok
        } else {
            __gm__ T *new_dst = dst + curValue * GmShape1;
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
            TileOp::UBCopyOutBase<T, src0rawShape1>(new_dst, src0 + i * src0rawShape1, 1, src0OriShape1, GmShape1);
        }
    }
}

// src1=index [1,2] , src0: [TShape0,TShape1,TShape2,TShape3],  dst [GmShape0,GmShape1,GmShape2,GmShape3]
template <typename T, typename T2, unsigned src0rawShape1, unsigned src0rawShape2, unsigned src0rawShape3,
    unsigned src1rawShape3, unsigned cacheMode, unsigned blockSize>
TILEOP void DynTIndexoutcast(__gm__ T *dst, __ubuf__ T *src0, __ubuf__ T2 *src1, unsigned src0OriShape0,
    unsigned src0OriShape1, unsigned src0OriShape3, unsigned src1OriShape0, unsigned src1OriShape1,
    unsigned GmShape0, unsigned GmShape1, unsigned GmShape2, unsigned GmShape3,
    unsigned Offset0, unsigned Offset1, unsigned Offset2, unsigned Offset3) {
    if (src0OriShape0 == 0 || src0OriShape1 == 0 || src0OriShape3 == 0 || src1OriShape0 == 0 || src1OriShape1 == 0) {
        return;
    }
    if (cacheMode == 2) {
        DynTIndexoutcast<T, T2>(dst, src0, src1, src1OriShape0, src1OriShape1, src1rawShape3,
            src0OriShape3, src0rawShape1, src0rawShape3);
        return;
    }
    dst += CalcLinearOffset(GmShape1, GmShape2, GmShape3, Offset0, Offset1, Offset2, Offset3);

    int alignTS2TS3 = src0rawShape2 * src0rawShape3; // ub需要32B对齐
    int alignSrc1 = src1rawShape3;                   // 修改对 src1 的大小计算
    for (int i = 0; i < src0OriShape0; ++i) {
        for (int j = 0; j < src0OriShape1; ++j) {
            TileOp::TIndexoutcastBase<T, T2, src0rawShape3, cacheMode, blockSize>
                (dst, src0, src1, src0OriShape3, src1OriShape1, GmShape3);
        }
        src0 += alignTS2TS3;
        src1 += alignSrc1;
        dst += GmShape2 * GmShape3;
    }
}

#ifndef TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H
#define TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <typename T, typename T2, unsigned cacheMode, unsigned blockSize>
__aicore__ inline void TScatterUpdate(
    T dst,        // GM 
    T src,  // UB 中的数据 [S × D]
    T2 src1  // UB 中的索引 [S]
){
    const auto uLayout = src.GetLayout();
    auto uShape0 = uLayout.template GetShapeDim<0, 5>();
    auto uShape1 = uLayout.template GetShapeDim<1, 5>();
    auto uShape2 = uLayout.template GetShapeDim<2, 5>();
    auto uShape3 = uLayout.template GetShapeDim<3, 5>(); // S
    auto uShape4 = uLayout.template GetShapeDim<4, 5>(); // D

    auto uStride0 = uLayout.template GetStrideDim<0, 5>();
    auto uStride1 = uLayout.template GetStrideDim<1, 5>();
    auto uStride2 = uLayout.template GetStrideDim<2, 5>();
    auto uStride3 = uLayout.template GetStrideDim<3, 5>();
    auto uStride4 = uLayout.template GetStrideDim<4, 5>();

    // === Get indices layout info ===
    const auto iLayout = src1.GetLayout();
    auto iShape0 = iLayout.template GetShapeDim<0, 5>();
    auto iShape1 = iLayout.template GetShapeDim<1, 5>();
    auto iShape2 = iLayout.template GetShapeDim<2, 5>();
    auto iShape3 = iLayout.template GetShapeDim<3, 5>();
    auto iShape4 = iLayout.template GetShapeDim<4, 5>();

    auto iStride0 = iLayout.template GetStrideDim<0, 5>();
    auto iStride1 = iLayout.template GetStrideDim<1, 5>();
    auto iStride2 = iLayout.template GetStrideDim<2, 5>();
    auto iStride3 = iLayout.template GetStrideDim<3, 5>();
    auto iStride4 = iLayout.template GetStrideDim<4, 5>();

    // === Get dst layout info ===
    const auto dLayout = dst.GetLayout();
    auto dShape0 = dLayout.template GetShapeDim<0, 5>();
    auto dShape1 = dLayout.template GetShapeDim<1, 5>();
    auto dShape2 = dLayout.template GetShapeDim<2, 5>();
    auto dShape3 = dLayout.template GetShapeDim<3, 5>();
    auto dShape4 = dLayout.template GetShapeDim<4, 5>();

    auto dStride0 = dLayout.template GetStrideDim<0, 5>();
    auto dStride1 = dLayout.template GetStrideDim<1, 5>();
    auto dStride2 = dLayout.template GetStrideDim<2, 5>();
    auto dStride3 = dLayout.template GetStrideDim<3, 5>();
    auto dStride4 = dLayout.template GetStrideDim<4, 5>();

    if (uShape0 == 0 || uShape1 == 0 || uShape3 == 0 || iShape0 == 0 || iShape1 == 0) {
        return;
    }
    if (cacheMode == 2) {
    unsigned batch     = uShape2;   // B
    unsigned seqLen    = uShape3;   // S (number of indices per batch)
    unsigned dim       = uShape4;   // D

    constexpr unsigned ALIGN_BYTES = 32;
    constexpr unsigned ELEM_SIZE = sizeof(T);
    unsigned paddedDim = (dim + (ALIGN_BYTES / ELEM_SIZE - 1)) / (ALIGN_BYTES / ELEM_SIZE) * (ALIGN_BYTES / ELEM_SIZE);

    unsigned paddedSeqUpd = (uStride3 == 0) ? seqLen : (uStride2 / uStride3);
    unsigned paddedSeqIdx = (iStride4 == 0) ? seqLen : (iStride3 / iStride4);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);

    __gm__ T* baseDst = (__gm__ T*)((uint64_t)(dst.GetAddr()));
    __ubuf__ T2* dstIdx = (__ubuf__ T2*)((uint64_t)(src1.GetAddr()));
    __ubuf__ T* curSrc = (__ubuf__ T*)((uint64_t)(src.GetAddr()));

    using DstDtype = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
    using SrcDtype = DstDtype;

    for (unsigned b = 0; b < batch; ++b) {
        for (unsigned s = 0; s < seqLen; ++s) {
            unsigned idx_val = static_cast<unsigned>(*dstIdx);
            __gm__ DstDtype* scatterAddr = reinterpret_cast<__gm__ DstDtype*>(baseDst) + idx_val * paddedDim;

            // 构造 GlobalTensor: shape [1,1,1,1,dim], stride [0,0,0,0,1] ===
            pto::GlobalTensor<DstDtype, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<0, 0, 0, 0, 1>> 
                dstGlobal(scatterAddr, pto::Shape(1, 1, 1, 1, dim));

            pto::Tile<pto::TileType::Vec, SrcDtype, 1, 1, pto::BLayout::RowMajor, -1, -1> srcUB(1, dim);
            pto::TASSIGN(srcUB, (uint64_t)curSrc);
            pto::TSTORE(dstGlobal, srcUB);
            curSrc += paddedDim;
            dstIdx++;
        }
        curSrc += (paddedSeqUpd - seqLen) * paddedDim;
        dstIdx += (paddedSeqIdx - seqLen);
    }
    return;
    }
    dst += CalcLinearOffset(GmShape1, GmShape2, GmShape3, Offset0, Offset1, Offset2, Offset3);

    int alignTS2TS3 = src0rawShape2 * src0rawShape3; // ub需要32B对齐
    int alignSrc1 = src1rawShape3;                   // 修改对 src1 的大小计算
    for (int i = 0; i < src0OriShape0; ++i) {
        for (int j = 0; j < src0OriShape1; ++j) {
            TileOp::TIndexoutcastBase<T, T2, src0rawShape3, cacheMode, blockSize>
                (dst, src0, src1, src0OriShape3, src1OriShape1, GmShape3);
        }
        src0 += alignTS2TS3;
        src1 += alignSrc1;
        dst += GmShape2 * GmShape3;
    }
}
#endif // TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H