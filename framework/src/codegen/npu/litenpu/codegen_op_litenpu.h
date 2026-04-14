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
 * \file codegen_op.h
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

private:
    std::unordered_map<Opcode, std::function<std::string()>> mteFixPipeOpsLiteNPU_ = {
        // UB <-> GM
        {Opcode::OP_UB_COPY_IN, [this]() { return GenUBCopyIn(); }},
        {Opcode::OP_UB_COPY_OUT, [this]() { return GenUBCopyOut(); }},
        // L1 <-> GM/BT/L1
        {Opcode::OP_L1_COPY_IN, [this]() { return GenMemL1CopyIn(); }},

        // L0C <-> GM
        {Opcode::OP_L0C_COPY_OUT, [this]() { return GenMemL0CCopyOut(); }},

        // L1 <-> L0
        {Opcode::OP_L1_TO_L0A, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_L0B, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_L0_BT, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_L0_AT, [this]() { return GenMemL1ToL0(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> unaryOpsLiteNPU_ = {
        // cast op
        {Opcode::OP_CAST, [this]() { return GenCastOp(); }},

        // unary op
        {Opcode::OP_EXP, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_SQRT, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_EXPAND, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_RECIPROCAL, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWMAXLINE, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWMINLINE, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ABS, [this]() { return GenUnaryOp(); }},

        // unary with temp buffer
        {Opcode::OP_ROWSUM_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWMAX_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWMIN_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_TRANSPOSE_VNCHWCONV, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> binaryOpsLiteNPU_ = {
        // binary op: vector operations
        {Opcode::OP_ADD, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_SUB, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_MUL, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_DIV, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_MAXIMUM, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_MINIMUM, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_PAIRSUM, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_PAIRMAX, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_PAIRMIN, [this]() { return GenBinaryOp(); }},

        // binary op: vector scalar
        {Opcode::OP_ADDS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_SUBS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_MULS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_DIVS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_MAXS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_MINS, [this]() { return GenVectorScalarOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> compositeOpsLiteNPU_ = {
        // indexput
        {Opcode::OP_INDEX_PUT, [this]() { return GenIndexPutOp(); }},

        // vector where
        {Opcode::OP_WHERE_SS, [this]() { return GenWhereOp(); }},
        {Opcode::OP_WHERE_TS, [this]() { return GenWhereOp(); }},
        {Opcode::OP_WHERE_ST, [this]() { return GenWhereOp(); }},
        {Opcode::OP_WHERE_TT, [this]() { return GenWhereOp(); }},

        // cmp op
        {Opcode::OP_CMP, [this]() { return GenCmpOp(); }},
        {Opcode::OP_CMPS, [this]() { return GenCmpOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> sortOpsLiteNPU_ = {};

    std::unordered_map<Opcode, std::function<std::string()>> cubeOpsLiteNPU_ = {
        // matmul
        {Opcode::OP_A_MUL_B, [this]() { return GenCubeOpMatmul(); }},
        {Opcode::OP_A_MUL_BT, [this]() { return GenCubeOpMatmul(); }},
        {Opcode::OP_A_MULACC_B, [this]() { return GenCubeOpMatmulAcc(); }},
        {Opcode::OP_A_MULACC_BT, [this]() { return GenCubeOpMatmulAcc(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> syncOpsLiteNPU_ = {
        // sync
        {Opcode::OP_SYNC_SRC, [this]() { return GenSyncSetOp(); }},
        {Opcode::OP_SYNC_DST, [this]() { return GenSyncWaitOp(); }},
        {Opcode::OP_BAR_V, [this]() { return GenBarrier(); }},
        {Opcode::OP_BAR_M, [this]() { return GenBarrier(); }},
        {Opcode::OP_BAR_ALL, [this]() { return GenBarrier(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> distributeOpsLiteNPU_ = {};

    std::unordered_map<Opcode, std::function<std::string()>> gatherScatterOpsLiteNPU_ = {};

    std::unordered_map<Opcode, std::function<std::string()>> normalVecOpsLiteNPU_ = {
        // vector dup
        {Opcode::OP_VEC_DUP, [this]() { return GenDupOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> perfOpsLiteNPU_ = {
        // for performace optimization
        {Opcode::OP_PHASE1, []() { return "SUBKERNEL_PHASE1\n"; }},
        {Opcode::OP_PHASE2, []() { return "SUBKERNEL_PHASE2\n"; }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> aicpuOpsLiteNPU_ = {};
};

} // namespace npu::tile_fwk

#endif // CODEGEN_OP_LITENPU_H
