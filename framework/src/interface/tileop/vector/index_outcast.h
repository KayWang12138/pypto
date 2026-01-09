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

#ifndef TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H
#define TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

template <unsigned cacheMode, unsigned blockSize, typename T0, typename T1, typename T2>
TILEOP  void TIndexOutcast(T0 dst, T1 src, T2 src1)
{
   constexpr auto expectSize = 5; // 所有张量都是 5D

    // === src0: data [1, B, S, 1, D] ===
    const auto uLayout = src.GetLayout();
    auto uShape1 = uLayout.template GetShapeDim<1, expectSize>(); 
    auto uShape2 = uLayout.template GetShapeDim<2, expectSize>();
    auto uShape3 = uLayout.template GetShapeDim<3, expectSize>(); 
    auto uShape4 = uLayout.template GetShapeDim<4, expectSize>(); 

    // Stride for src0
    auto uStride1 = uLayout.template GetStrideDim<1, expectSize>(); 
    auto uStride2 = uLayout.template GetStrideDim<2, expectSize>(); 
    auto uStride4 = uLayout.template GetStrideDim<4, expectSize>(); // stride of D (should be 1)

    // === src1: indices [1, 1, 1, B, S] ===
    const auto iLayout = src1.GetLayout();
    auto iShape3 = iLayout.template GetShapeDim<3, expectSize>(); // B
    auto iShape4 = iLayout.template GetShapeDim<4, expectSize>(); // S

    auto iStride3 = iLayout.template GetStrideDim<3, expectSize>(); // stride of B
    auto iStride4 = iLayout.template GetStrideDim<4, expectSize>(); // stride of S (should be 1)

    // === dst: [1, N, K, 1, D] ===
    const auto dLayout = dst.GetLayout();
    auto dShape1 = dLayout.template GetShapeDim<1, expectSize>(); // N (logical rows)
    auto dShape2 = dLayout.template GetShapeDim<2, expectSize>(); // K = 32 or 128 (physical block size)
    auto dShape4 = dLayout.template GetShapeDim<4, expectSize>(); // D

    auto dStride1 = dLayout.template GetStrideDim<1, expectSize>();   
    auto dStride4 = dLayout.template GetStrideDim<4, expectSize>(); // should be 1

    using DstDtype = typename T0::Type;   // dst 的数据类型
    using SrcDtype = typename T1::Type;   // src (data) 的数据类型
    using IdxDtype = typename T2::Type;   // src1 (index) 的数据类型
        
    // === 提取对齐后的 raw shape（TileShape 是 5D）===
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, 5>(); // dim3 = 1
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, 5>(); // D_32aligned
    constexpr auto src0rawShape4 = dstTileW;                             // D_32aligned
    constexpr auto src0rawShape2 = TileOp::GetTensorTileShapeDim<T1, 2, 5>(); // S_32aligned from src

    constexpr auto src1rawShape4 = TileOp::GetTensorTileShapeDim<T2, 4, 5>(); // S_32aligned for index

    if (uShape1 == 0 || uShape2 == 0 || uShape4 == 0 || iShape3 == 0 || iShape4 == 0) {
        return;
    }

    if constexpr (cacheMode == 2) {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);

        __ubuf__ SrcDtype* srcBase = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
        __ubuf__ IdxDtype* idxBase = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
        __gm__ DstDtype* dstBase   = reinterpret_cast<__gm__ DstDtype*>(dst.GetAddr());

        unsigned B = iShape3;
        unsigned S = iShape4;
        unsigned D = uShape4;

        constexpr unsigned S_32aligned = src0rawShape2;
        constexpr unsigned D_32aligned = src0rawShape4;

        for (unsigned b = 0; b < B; ++b) {
            for (unsigned s = 0; s < S; ++s) {
                // 读取 index: src1[0,0,0,b,s]
                IdxDtype idx_val_raw = idxBase[b * iStride3 + s * iStride4];
                unsigned idx_val = static_cast<unsigned>(idx_val_raw);

                // 计算 src0 地址: [0, b, s, 0, 0]
                // src0 layout: [1, B, S, 1, D]
                // offset = b * stride_B + s * stride_S + 0 * stride_1 + 0 * stride_D
                // 由于 dim3=1, stride3 = D, 但我们从 [b][s][0][0] 开始，所以只需前两项
                uint64_t srcOffset = b * uStride1 + s * uStride2; 
                // 构造 src tile: w * h
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, dstTileH, dstTileW, 
                                               pto::BLayout::RowMajor, -1, -1>;
                SrcTileDefine srcTile(uShape3, uShape4);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(srcBase + srcOffset));

                // 写入 dst: dst[0, idx_val, 0, 0, 0]
                // dst layout: [1, N, K, 1, D] → 逻辑行 idx_val 的起始地址 = dstBase + idx_val * dStride1
                __gm__ DstDtype* scatterAddr = dstBase + idx_val * dStride1;

              // 使用 5D 动态 GlobalTensor
                using DstGlobalType = pto::GlobalTensor<
                    DstDtype,
                    pto::Shape<-1, -1, -1, -1, -1>,
                    pto::Stride<-1, -1, -1, -1, -1>
                >;

                DstGlobalType dstGlobal(
                    scatterAddr,
                    pto::Shape<-1, -1, -1, -1, -1>(1, 1, 1, 1, static_cast<int64_t>(D)),  
                    pto::Stride<-1, -1, -1, -1, -1>(0, 0, 0, 0, 1)                          
                );
                pto::TSTORE(dstGlobal, srcTile.Data());
            } 
            // 在 UB 中，src0 的 S 维被 padding 到 S_32aligned
            // 所以每处理完一个 b，要跳过 (S_32aligned - S) 个 S-slice
            // 每个 S-slice 占 D_32aligned 个元素（因为 D 也可能 padding）
            constexpr unsigned D_32aligned = src0rawShape4;

            // 对于 index buffer (src1): 每个 S 元素占 1 个 IdxDtype，S 维 padding 到 S_32aligned
            idxBase += (S_32aligned - S); // 因为 src1 的 D=1，stride_S=1

            // 对于 data buffer (src0): 每个 S-slice 占 D_32aligned 个元素
            srcBase += (S_32aligned - S) * D_32aligned; // 关键！不是 * D，而是 * D_32aligned
        }
        return;
    }
    unsigned B = uShape1;   // = iShape3
    unsigned S = uShape2;   // = iShape4
    unsigned D = uShape4;   // logical D

    constexpr unsigned S_32aligned = src0rawShape2; // S aligned in UB
    constexpr unsigned D_32aligned = src0rawShape4; // D aligned in UB

    // UB 起始地址
    __ubuf__ SrcDtype* src0_base = reinterpret_cast<__ubuf__ SrcDtype*>(src.GetAddr());
    __ubuf__ IdxDtype* src1_base = reinterpret_cast<__ubuf__ IdxDtype*>(src1.GetAddr());
    __gm__  DstDtype* dst_base   = reinterpret_cast<__gm__  DstDtype*>(dst.GetAddr());

    // 按 B 维度遍历
    for (unsigned b = 0; b < B; ++b) {
        // 当前 B 块在 UB 中的起始地址
        __ubuf__ SrcDtype* cur_src0 = src0_base + b * (S_32aligned * D_32aligned);
        __ubuf__ IdxDtype* cur_src1 = src1_base + b * S_32aligned;

        // 按 S 维度遍历（每个 index 对应一个 D 向量）
        for (unsigned s = 0; s < S; ++s) {
            // 插入流水屏障（按原 Base 函数）
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID7);
            set_flag(PIPE_V, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID7);

            // 读取 index: src1[b][s]
            IdxDtype curValue = cur_src1[s];

            if constexpr (cacheMode == 1) { // PA_NZ
                // PA_NZ 模式：index = blockCount * blockSize + offset
                auto blockCount = static_cast<unsigned>(curValue) / blockSize;
                auto index_in_block = static_cast<unsigned>(curValue) % blockSize;
                unsigned global_row = blockCount * blockSize + index_in_block;

                

                // 计算 dst 地址:
                // - 每个 block 占 blockSize 行
                // - 每行 stride = dStride1 (padded_D)
               __gm__ DstDtype* new_dst = dst_base + global_row * dStride1;

                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

                // 使用 TSTORE 直接拷贝 D 个元素
                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, dstTileH, dstTileW,
                                               pto::BLayout::RowMajor, -1, -1>;
                SrcTileDefine srcTile(1, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(cur_src0 + s * D_32aligned));

                using ScatterShape  = pto::Shape<1, 1, 1, 1, -1>;
                using ScatterStride = pto::Stride<0, 0, 0, 0, -1>;

                auto dstGlobal = pto::GlobalTensor<DstDtype, ScatterShape, ScatterStride>(
                    new_dst,
                    pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(D)),
                    pto::Stride<0, 0, 0, 0, -1>(0, 0, 0, 0, 1)
                );
                pto::TSTORE(dstGlobal, srcTile.Data());
            } else { // cacheMode == 0 或其他：普通 scatter
                // curValue 是逻辑行号（0 ～ N-1）
                unsigned row_id = static_cast<unsigned>(curValue);
                __gm__ DstDtype* new_dst = dst_base + row_id * dStride1;

                set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
                wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);

                using SrcTileDefine = pto::Tile<pto::TileType::Vec, SrcDtype, dstTileH, dstTileW,
                                                pto::BLayout::RowMajor, -1, -1>;
                SrcTileDefine srcTile(1, D);
                pto::TASSIGN(srcTile, reinterpret_cast<uint64_t>(cur_src0 + s * D_32aligned));

                using ScatterShape  = pto::Shape<1, 1, 1, 1, -1>;
                using ScatterStride = pto::Stride<0, 0, 0, 0, -1>;

                auto dstGlobal = pto::GlobalTensor<DstDtype, ScatterShape, ScatterStride>(
                    new_dst,
                    pto::Shape<1, 1, 1, 1, -1>(1, 1, 1, 1, static_cast<int64_t>(D)),
                    pto::Stride<0, 0, 0, 0, -1>(0, 0, 0, 0, 1)
                );
                pto::TSTORE(dstGlobal, srcTile.Data());
                }
        }
    }
}

#endif // TILEOP_TILE_OPERATOR_INDEX_OUTCAST__H