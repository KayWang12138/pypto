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
* \file gather_after_prolog.h
* \brief
*/
 
#pragma once
#ifndef GATHER_AFTER_PROLOG_H
#define GATHER_AFTER_PROLOG_H
 
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/inner/pre_def.h"
#include "tilefwk/tilefwk.h"
#include "dsia_common.h"
 
 
namespace npu::tile_fwk {
Tensor CalcOffsetsForGather(const Tensor &topKIndcies, const Tensor &blockTable, const Tensor &actSeqs,
   const DSIASimpleParams &params, SymbolicScalar b, SymbolicScalar s1);

void GatherAfterPrologCompute(Tensor &topKIndcies, Tensor &kNopeCache, Tensor &kRopeCache, Tensor &blockTable, Tensor &actSeqs,
   Tensor &gatherRes, const DSIASimpleParams &params, SymbolicScalar b, SymbolicScalar s1);
 
} // namespace npu::tile_fwk
 
#endif // GATHER_AFTER_PROLOG_H