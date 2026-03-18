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
 * - TPermuteComputeIdx: 计算输出位置对应的输入位置索引（使用TCI）
 * - TPermuteGather: 用Gather获取输出的正确数据（使用TGATHER）
 * 
 * 参考架构：
 * Operation层 -> TileFunc层 -> TiledInnerPermute -> AddOperation -> TileOp层
 */

#ifndef TILEOP_TILE_OPERATOR_PERMUTE__H
#define TILEOP_TILE_OPERATOR_PERMUTE__H

#include "utils/layout.h"
#include "utils/tile_tensor.h"

#define OP_TILE_OP_PERMUTE_COMPUTE_IDX TPermuteComputeIdx
template <typename T0, typename T1>
TILEOP void TPermuteComputeIdx(T0 src, T1 idxOut,
                               const std::vector<int64_t>& perm,
                               const std::vector<int64_t>& srcShape,
                               const std::vector<int64_t>& dstShape,
                               const std::vector<int64_t>& tileShape,
                               int64_t dim) {
    constexpr size_t expectSize = 5;
    
    auto srcAddr = (__ubuf__ typename T0::Type*)((uint64_t)(src.GetAddr()));
    auto idxAddr = (__ubuf__ uint32_t*)((uint64_t)(idxOut.GetAddr()));
    
    int64_t srcStrides[5] = {1, 1, 1, 1, 1};
    int64_t tileShapes[5] = {1, 1, 1, 1, 1};
    
    for (int i = 0; i < dim; i++) {
        tileShapes[i] = tileShape[i];
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
            
            int64_t dstPos[5] = {0, 0, 0, 0, 0};
            int64_t tempIdx = linearIdx;
            for (int i = dim - 1; i >= 0; i--) {
                dstPos[i] = tempIdx % tileShapes[i];
                tempIdx /= tileShapes[i];
            }
            
            int64_t srcPos[5] = {0, 0, 0, 0, 0};
            for (int i = 0; i < dim; i++) {
                srcPos[i] = dstPos[invPerm[i]];
            }
            
            int64_t srcOffset = 0;
            for (int i = 0; i < dim; i++) {
                srcOffset += srcPos[i] * srcStrides[i];
            }
            
            idxPtr[t] = static_cast<uint32_t>(srcOffset);
        }
        
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
    
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#define OP_TILE_OP_PERMUTE_GATHER TPermuteGather
template <typename T0, typename T1, typename T2>
TILEOP void TPermuteGather(T0 dst, T1 src, T2 idxIn,
                           const std::vector<int64_t>& perm,
                           const std::vector<int64_t>& srcShape,
                           const std::vector<int64_t>& dstShape,
                           const std::vector<int64_t>& tileShape,
                           int64_t dim) {
    constexpr size_t expectSize = 5;
    const auto idxLayout = idxIn.GetLayout();
    
    auto dstAddr = (__ubuf__ typename T0::Type*)((uint64_t)(dst.GetAddr()));
    auto srcAddr = (__ubuf__ typename T1::Type*)((uint64_t)(src.GetAddr()));
    auto idxAddr = (__ubuf__ uint32_t*)((uint64_t)(idxIn.GetAddr()));
    
    auto idxShape0 = idxLayout.template GetShapeDim<0, expectSize>();
    
    if (idxShape0 == 0) {
        return;
    }
    
    constexpr int64_t tileW = 256;
    
    using DstTileDefine = pto::Tile<pto::TileType::Vec, typename T0::Type, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    using IdxTileDefine = pto::Tile<pto::TileType::Vec, uint32_t, 1, tileW, pto::BLayout::RowMajor, -1, -1>;
    
    DstTileDefine dstTile(1, tileW);
    IdxTileDefine idxTile(1, tileW);
    
    set_flag(PIPE_V, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID7);
    
    int64_t totalElements = idxShape0;
    
    for (int64_t baseIdx = 0; baseIdx < totalElements; baseIdx += tileW) {
        int64_t curTileSize = std::min(static_cast<int64_t>(tileW), totalElements - baseIdx);
        
        pto::TASSIGN(dstTile, (uint64_t)(dstAddr + baseIdx));
        pto::TASSIGN(idxTile, (uint64_t)(idxAddr + baseIdx));
        
        set_flag(PIPE_V, PIPE_S, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID6);
        
        pto::TGATHER(dstTile, srcAddr, idxTile);
        
        set_flag(PIPE_S, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
    }
    
    set_flag(PIPE_S, PIPE_V, EVENT_ID7);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID7);
}

#endif
