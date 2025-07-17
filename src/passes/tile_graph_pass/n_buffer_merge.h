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
 * \file n_buffer_merge.h
 * \brief
 */

#ifndef PASS_N_BUFFER_MERGE_H_
#define PASS_N_BUFFER_MERGE_H_

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "passes/pass_interface/pass.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
namespace npu::tile_fwk {
class NBufferMergePass : public Pass {
public:
    NBufferMergePass() : Pass("NBufferMergePass") {}
    ~NBufferMergePass() override = default;
private:
    Status RunOnFunction(Function &function) override;
};
}  // namespace npu::tile_fwk
#endif  // PASS_N_BUFFER_MERGE_H_