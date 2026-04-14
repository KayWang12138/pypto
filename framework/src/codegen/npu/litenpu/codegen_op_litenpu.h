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
 * \file codegen_op_litenpu.h
 * \brief
 */

#ifndef CODEGEN_OP_LITENPU_H
#define CODEGEN_OP_LITENPU_H

#include <utility>
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

class CodeGenOpLiteNPU : public CodeGenOpNPU {
public:
    explicit CodeGenOpLiteNPU(const CodeGenOpNPUCtx& ctx);
    ~CodeGenOpLiteNPU() override = default;

private:
    TileTensor QueryTileTensorByIdx(int paramIdx) const override;

    std::string GenGmParamVar(unsigned gmParamIdx) const override;

    TileTensor BuildTileTensor(
        int paramIdx, const std::string& usingType, const ShapeInLoop& shapeInLoop = {}) override;

    void UpdateTileTensorShapeAndStride(
        int paramIdx, TileTensor& tileTensor, bool isSpillToGm, const ShapeInLoop& shapeInLoop = {}) override;
};

} // namespace npu::tile_fwk

#endif // CODEGEN_OP_LITENPU_H
