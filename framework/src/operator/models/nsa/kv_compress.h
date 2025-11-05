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
 * \file kv_compress.h
 * \brief
 */

#pragma once
#ifndef KV_COMPRESS_H
#define KV_COMPRESS_H

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "fused_compress_kv_select.h"

namespace npu::tile_fwk {

void compressKv(const Tensor &kvCache, const Tensor &krCache,
    const Tensor &cmpKvCache, const Tensor &cmpKrCache, const Tensor &blockTable, Tensor &cmpCacheIndex, 
    const Tensor &actSeqLen, const Tensor &mlpWk1, const Tensor &mlpWk2,
    const Tensor &mlpCos, const Tensor &mlpSin, 
    Tensor &cmpKvCacheOut, Tensor &cmpKrCacheOut, Tensor &auxTensor,
    const int cmpBlockSize, const int cmpStride, const int rs, CmpAttnTile &tileConfig);

} // namespace npu::tile_fwk

#endif // KV_COMPRESS_H
