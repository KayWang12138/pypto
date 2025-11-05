/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file DYNAMIC_NSA_V2.h
 * \brief
 */
 
#pragma once
#ifndef DYNAMIC_NSA_V2
#define DYNAMIC_NSA_V2
 
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "interface/program/program.h"
#include "dynamic_mla_v32.h"
#include "gather_after_prolog.h"
#include "sparse_flash_attention.h"
#include "operator/models/nsa/dynamic_nsa_common.h"
#include "lightning_indexer_topk.h"
#include "lightning_indexer_prolog.h"
 
namespace npu::tile_fwk {
void DecodeIndexerAttention(const Tensor &tokenX, const Tensor &wDq, const Tensor &wUqQr, const Tensor &wUk, const Tensor &wDkvKr,
    const Tensor &gammaCq, const Tensor &gammaCkv, const Tensor &sin, const Tensor &cos, const Tensor &cacheIndex,
    Tensor &kvCache, Tensor &krCache, const MlaQuantInputs &quantInputs, Tensor &blockTable, Tensor &actSeqs,
    const Tensor &qW, const Tensor &kW, const Tensor &projW, const Tensor &lnW, const Tensor &lnBias,
    const Tensor &indexKCache, Tensor &attentionOut,
    Tensor &gatherResTmp, Tensor &tmpTopkInput, Tensor &tmpIndexerTopkRes, Tensor &tmpRowSumOut,  Tensor &rmsResOut,
    Tensor &queryOut,Tensor &weightsOut, Tensor &qNopeOut,Tensor &qRopeOut,const NSASimpleParams &params);
 
} // namespace npu::tile_fwk
 
#endif // DYNAMIC_NSA_V2