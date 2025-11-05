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
 * \file gather_selected_attention.h
 * \brief
 */

#pragma once
#ifndef SELECTED_ATTENTION
#define SELECTED_ATTENTION

#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"

#include "operator/models/nsa/dynamic_nsa_common.h"

namespace npu::tile_fwk {
void SelectedAttentionComputeV2(const Tensor &qNope, const Tensor &qRope, const Tensor &kNope2D, const Tensor &kRope2D,
    const Tensor &knAuxTensor, const Tensor &scaleAuxTensor, const Tensor &kNopeScales, Tensor &offsets, const Tensor &kvSlcActSeqs,
    int nQ, int nKv, float softmaxScale, int topk, Tensor &attentionOut, SaTileShapeConfig tileConfig={});

void SelectedAttentionV2(const Tensor &qNope, const Tensor &qRope, const Tensor &kNope2D, const Tensor &kRope2D,
    const Tensor &knAuxTensor, const Tensor &scaleAuxTensor, const Tensor &kNopeScales, Tensor &offsets, const Tensor &kvSlcActSeqs,
    int nQ, int nKv, float softmaxScale, int topk, Tensor &attentionOut, SaTileShapeConfig tileConfig={});
} // namespace npu::tile_fwk

#endif // SELECTED_ATTENTION
