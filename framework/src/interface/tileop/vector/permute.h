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
 * \brief Permute TileOp - 外抛for循环处理二维Tile块
 * 
 * TileOp层职责：
 * - TPermuteComputeIdx: 遍历输出tile，计算每个输出位置对应的输入线性索引
 * - TPermuteGather: 用Gather获取输出的正确数据（使用TGATHER）
 * 
 * 参考架构：
 * Operation层 -> TileFunc层 -> TiledInnerPermute -> TileOp层
 * 
 * 索引计算逻辑：
 * 对于输出位置 dstPos，计算对应的输入位置 srcPos：
 * - invPerm[perm[i]] = i
 * - srcPos[i] = dstPos[invPerm[i]]
 * - srcLinearIdx = sum(srcPos[i] * srcStrides[i])
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE__H
#define TILEOP_TILE_OPERATOR_PERMUTE__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_PERMUTE_COMPUTE_IDX TPermuteComputeIdx
template <typename T0, typename T1>
TILEOP void TPermuteComputeIdx(T0 dstTile, T1 idxOut,
                               const std::vector<int64_t>& perm,
                               const std::vector<int64_t>& srcShape,
                               const std::vector<int64_t>& dstShape,
                               const std::vector<int64_t>& tileShape,
                               const std::vector<int64_t>& tileOffset,
                               int64_t dim) {
    auto idxAddr = (__ubuf__ uint32_t*)((uint64_t)(idxOut.GetAddr()));
    
    int64_t srcStrides[5] = {1, 1, 1, 1, 1};
    int64_t tileShapes[5] = {1, 1, 1, 1, 1};
    int64_t tileOffsets[5] = {0, 0, 0, 0, 0};
    
    for (int i = 0; i < dim; i++) {
        tileShapes[i] = tileShape[i];
        tileOffsets[i] = tileOffset[i];
    }
    
    for (int i = dim - 2; i >= 0; i--) {
        srcStrides[i] = srcStrides[i + 1] * srcShape[i + 1];
    }
    
    int64_t invPerm[5] = {0, 1, 2, 3, 4};
    for (int i = 0; i < dim; i++) {
        invPerm[perm[i]] = i;
    }
    
    int64_t totalElements = 1;
    for (int i = 0; i < dim; i++) {
        totalElements *= tileShapes[i];
    }
    
    constexpr int64_t tileW = 256;
    
    using IdxTileDefine = pto::Tile<pto::TileType::Vec, uint32_t, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    
    for (int64_t baseIdx = 0; baseIdx < totalElements; baseIdx += tileW) {
        int64_t curTileSize = std::min(static_cast<int64_t>(tileW), totalElements - baseIdx);
        
        IdxTileDefine idxTile(1, curTileSize);
        pto::TASSIGN(idxTile, (uint64_t)(idxAddr + baseIdx));
        set_flag(PIPE_V, PIPE_S, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
        pto::TCI<IdxTileDefine, uint32_t, 0>(idxTile, static_cast<uint32_t>(baseIdx));
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
        
        auto idxPtr = idxAddr + baseIdx;
        
        for (int64_t t = 0; t < curTileSize; t++) {
            uint32_t linearIdx = idxPtr[t];
            
            int64_t tilePos[5] = {0, 0, 0, 0, 0};
            int64_t tempIdx = linearIdx;
            for (int i = dim - 1; i >= 0; i--) {
                tilePos[i] = tempIdx % tileShapes[i];
                tempIdx /= tileShapes[i];
            }
            
            int64_t dstPos[5] = {0, 0, 0, 0, 0};
            for (int i = 0; i < dim; i++) {
                dstPos[i] = tileOffsets[i] + tilePos[i];
            }
            
            int64_t srcPos[5] = {0, 0, 0, 0, 0};
            for (int i = 0; i < dim; i++) {
                srcPos[i] = dstPos[invPerm[i]];
            }
            
            int64_t srcLinearIdx = 0;
            for (int i = 0; i < dim; i++) {
                srcLinearIdx += srcPos[i] * srcStrides[i];
            }
            
            idxPtr[t] = static_cast<uint32_t>(srcLinearIdx);
        }
        
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
    
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#define OP_TILE_OP_PERMUTE_GATHER TPermuteGather
template <typename T0, typename T1, typename T2>
TILEOP void TPermuteGather(T0 dst, T1 src, T2 idxIn) {
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
    auto dstStride4 = dstLayout.template GetStrideDim<4, expectSize>();
    
    if (dstShape3 == 0 || dstShape4 == 0) {
        return;
    }
    
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto srcAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src.GetAddr()));
    auto idxAddr = (__ubuf__ uint32_t*)((uint64_t)(idxIn.GetAddr()));
    
    constexpr auto dstTileH = TileOp::GetTensorTileShapeDim<T0, 3, expectSize>();
    constexpr auto dstTileW = TileOp::GetTensorTileShapeDim<T0, 4, expectSize>();
    constexpr auto idxTileW = TileOp::GetTensorTileShapeDim<T2, 4, expectSize>();
    
    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, dstTileH, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    using IdxTileDefine = pto::Tile<pto::TileType::Vec, uint32_t, 1, idxTileW, pto::BLayout::RowMajor, -1, -1>;
    using SrcTileDefine = pto::Tile<pto::TileType::Vec, typename T1::Type, 1, dstTileW, pto::BLayout::RowMajor, -1, -1>;
    
    DstTileDefine dstTile(dstShape3, dstShape4);
    IdxTileDefine idxTile(1, dstShape4);
    SrcTileDefine srcTile(1, dstShape4);
    
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    
    for (LoopVar i0 = 0; i0 < dstShape0; ++i0) {
        for (LoopVar i1 = 0; i1 < dstShape1; ++i1) {
            for (LoopVar i2 = 0; i2 < dstShape2; ++i2) {
                for (LoopVar i3 = 0; i3 < dstShape3; i3 += dstTileH) {
                    auto curTileH = std::min(static_cast<int64_t>(dstShape3 - i3), static_cast<int64_t>(dstTileH));
                    
                    for (LoopVar i4 = 0; i4 < dstShape4; i4 += dstTileW) {
                        auto curTileW = std::min(static_cast<int64_t>(dstShape4 - i4), static_cast<int64_t>(dstTileW));
                        
                        auto dstOffset = i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2 + 
                                        i3 * dstStride3 + i4 * dstStride4;
                        
                        pto::TASSIGN(dstTile, (uint64_t)(dstAddr + dstOffset));
                        pto::TASSIGN(idxTile, (uint64_t)(idxAddr + dstOffset));
                        pto::TASSIGN(srcTile, (uint64_t)srcAddr);
                        
                        set_flag(PIPE_V, PIPE_S, EVENT_ID6);
                        wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
                        
                        pto::TGATHER(dstTile, srcTile, idxTile);
                        
                        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
                        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
                    }
                }
            }
        }
    }
    
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#endif
