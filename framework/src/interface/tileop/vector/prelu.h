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
 * \file prelu.h
 * \brief
 */

#ifndef INTERFACE_TILEOP_VECTOR_PRELU_H_
#define INTERFACE_TILEOP_VECTOR_PRELU_H_

#include "tileop.h"
#include "interface/tileop/utils/vector_tile_utils.h"

namespace npu {
namespace tile_fwk {
namespace tileop {

/*!
 * \brief PReLU operation for different axes
 */
template <typename T, typename T1, typename T2, typename T3, int axis>
TILEOP void TPrelu(T1 dst, T2 src, T3 weight, T3 tmp) {
    using DstType = typename T1::Type;
    using SrcType = typename T2::Type;
    using WeightType = typename T3::Type;
    
    constexpr size_t BLOCK_NELEM = TileOp::BLOCK_NELEM;
    
    if constexpr (axis == 4) {
        // For 2D input (N, C), weight is (C,)
        for (LoopVar n0Index = 0; n0Index < dst.shape[0]; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dst.shape[1]; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dst.shape[2]; ++n2Index) {
                    auto dstOffset = n0Index * dst.stride[0] + n1Index * dst.stride[1] + n2Index * dst.stride[2];
                    auto srcOffset = n0Index * src.stride[0] + n1Index * src.stride[1] + n2Index * src.stride[2];
                    auto weightOffset = n1Index * weight.stride[0];
                    
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstType)));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcType)));
                    pto::TASSIGN(weightTile, (uint64_t)(weight.GetAddr() + weightOffset * sizeof(WeightType)));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    
                    pto::TPRELU(dstTile, srcTile, weightTile, tmpTile);
                }
            }
        }
    } else if constexpr (axis == 2 || axis == 3) {
        // For 4D input (N, C, H, W), weight is (C,)
        for (LoopVar n0Index = 0; n0Index < dst.shape[0]; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dst.shape[1]; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dst.shape[2]; ++n2Index) {
                    for (LoopVar n3Index = 0; n3Index < dst.shape[3]; ++n3Index) {
                        for (LoopVar n4Index = 0; n4Index < dst.shape[4]; ++n4Index) {
                            auto dstOffset = n0Index * dst.stride[0] + n1Index * dst.stride[1] + 
                                           n2Index * dst.stride[2] + n3Index * dst.stride[3] + n4Index * dst.stride[4];
                            auto srcOffset = n0Index * src.stride[0] + n1Index * src.stride[1] + 
                                           n2Index * src.stride[2] + n3Index * src.stride[3] + n4Index * src.stride[4];
                            auto weightOffset = n1Index * weight.stride[0];
                            
                            pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstType)));
                            pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcType)));
                            pto::TASSIGN(weightTile, (uint64_t)(weight.GetAddr() + weightOffset * sizeof(WeightType)));
                            
                            pto::TLRELU(dstTile, srcTile, weightTile);
                        }
                    }
                }
            }
        }
    } else {
        // Default case for other axes
        for (LoopVar n0Index = 0; n0Index < dst.shape[0]; ++n0Index) {
            for (LoopVar n1Index = 0; n1Index < dst.shape[1]; ++n1Index) {
                for (LoopVar n2Index = 0; n2Index < dst.shape[2]; ++n2Index) {
                    auto dstOffset = n0Index * dst.stride[0] + n1Index * dst.stride[1] + n2Index * dst.stride[2];
                    auto srcOffset = n0Index * src.stride[0] + n1Index * src.stride[1] + n2Index * src.stride[2];
                    auto weightOffset = n1Index * weight.stride[0];
                    
                    pto::TASSIGN(dstTile, (uint64_t)(dst.GetAddr() + dstOffset * sizeof(DstType)));
                    pto::TASSIGN(srcTile, (uint64_t)(src.GetAddr() + srcOffset * sizeof(SrcType)));
                    pto::TASSIGN(weightTile, (uint64_t)(weight.GetAddr() + weightOffset * sizeof(WeightType)));
                    pto::TASSIGN(tmpTile, (uint64_t)(tmp.GetAddr()));
                    
                    pto::TPRELU(dstTile, srcTile, weightTile, tmpTile);
                }
            }
        }
    }
}

#define OP_TILE_OP_PRELU TPrelu

template <typename T, typename T1, typename T2, typename T3>
TILEOP void TPrelu(T1 dst, T2 src, T3 weight, T3 tmp) {
    // Default implementation with axis=4
    TPrelu<T, T1, T2, T3, 4>(dst, src, weight, tmp);
}

} // namespace tileop
} // namespace tile_fwk
} // namespace npu

#endif // INTERFACE_TILEOP_VECTOR_PRELU_H_
