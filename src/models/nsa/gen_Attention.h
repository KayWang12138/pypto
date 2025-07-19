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
#include "interface/configs/config.h"

namespace npu::tile_fwk {
constexpr int NUM_2 = 2;
constexpr int NUM_3 = 3;
constexpr int NUM_16 = 16;
constexpr int NUM_32 = 32;
constexpr int NUM_128 = 128;
constexpr int NUM_512 = 512;

void GenAttention(Tensor &cmpAtten, Tensor &selAtten, Tensor &winAtten, Tensor &gatingScore, Tensor &attentionOut);
} // namespace npu::tile_fwk

#endif // MLA_PROLOG