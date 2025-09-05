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
 * \file gen_Attention.h
 * \brief
 */

#pragma once
#ifndef GEN_ATTENTION
#define GEN_ATTENTION

#include "operation/tilefwk_op.h"
#include "common/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_storage.h"

namespace npu::tile_fwk {
struct GenAttenTileShapeConfig {
    int tileBSize;
    int tileS1Size;
    std::array<int, TILE_VEC_FOUR_DIMS> vec1TileShape; // vector op tileshape
    std::array<int, TILE_VEC_FOUR_DIMS> vec2TileShape; // vector op tileshape
};

void GenAttentionCompute(Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &gatingScore, Tensor &attentionOut,
    GenAttenTileShapeConfig &tileConfig);

void GenAttention(Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &gatingScore, Tensor &attentionOut,
    GenAttenTileShapeConfig &tileConfig);
} // namespace npu::tile_fwk

#endif // MLA_PROLOG