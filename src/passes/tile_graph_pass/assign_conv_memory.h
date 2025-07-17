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
 * \file assign_conv_memory.h
 * \brief
 */

#ifndef TILE_FWK_ASSIGN_CONV_MEMORY_TYPE_H
#define TILE_FWK_ASSIGN_CONV_MEMORY_TYPE_H

#include "interface/function/function.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "passes/pass_interface/pass.h"

namespace npu::tile_fwk {
class AssignConvMemoryType : public Pass {
public:
    AssignConvMemoryType() : Pass("AssignConvMemoryType") {}

    Status RunOnFunction(Function &function) override;

private:
    void RunOnOperation(Operation &operation) const;
};
}

#endif // TILE_FWK_ASSIGN_CONV_MEMORY_TYPE_H