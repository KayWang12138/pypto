/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file codegen_op.h
 * \brief
 */

#ifndef CODEGEN_OP_CLOUDNPU_H
#define CODEGEN_OP_CLOUDNPU_H

#include <utility>
#include <map>
#include <functional>
#include <unordered_set>

#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_impl.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen/npu/op_print_param_def.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/stmt_mgr/codegen_for_block.h"
#include "codegen/codegen_op.h"
#include "codegen/npu/codegen_op_npu.h"

namespace npu::tile_fwk {

class CodeGenOpCloudNPU : public CodeGenOpNPU {
public:
    explicit CodeGenOpCloudNPU(const CodeGenOpNPUCtx& ctx);

    ~CodeGenOpCloudNPU() override = default;
};
} // namespace npu::tile_fwk

#endif // CODEGEN_OP_CLOUDNPU_H
