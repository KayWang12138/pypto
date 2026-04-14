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

private:
    std::unordered_map<Opcode, std::function<std::string()>> mteFixPipeOpsCloudNPU_ = {
        // UB <-> GM
        {Opcode::OP_UB_COPY_IN, [this]() { return GenUBCopyIn(); }},
        {Opcode::OP_UB_COPY_OUT, [this]() { return GenUBCopyOut(); }},
        {Opcode::OP_RESHAPE_COPY_IN, [this]() { return GenReshapeCopyIn(); }},
        {Opcode::OP_RESHAPE_COPY_OUT, [this]() { return GenReshapeCopyOut(); }},
        {Opcode::OP_L1_TO_FIX_QUANT_PRE, [this]() { return GenMemL1ToFB(); }},
        {Opcode::OP_GATHER_IN_UB, [this]() { return GenGatherInUB(); }},
        {Opcode::OP_GATHER, [this]() { return GenGatherOp(); }},
        // L1 <-> GM/BT/L1
        {Opcode::OP_L1_COPY_IN, [this]() { return GenMemL1CopyIn(); }},
        {Opcode::OP_L1_COPY_IN_A_SCALE, [this]() { return GenMemL1CopyIn(); }},
        {Opcode::OP_L1_COPY_IN_B_SCALE, [this]() { return GenMemL1CopyIn(); }},
        {Opcode::OP_L1_COPY_OUT, [this]() { return GenMemL1CopyOut(); }},
        {Opcode::OP_GATHER_IN_L1, [this]() { return GenGatherInL1(); }},
        {Opcode::OP_L1_COPY_IN_CONV, [this]() { return GenMemL1CopyInConv(); }},

        // L0C <-> GM
        {Opcode::OP_L0C_COPY_OUT, [this]() { return GenMemL0CCopyOut(); }},
        {Opcode::OP_L0C_COPY_OUT_CONV, [this]() { return GenMemL1CopyOutConv(); }},

        {Opcode::OP_L0C_TO_L1, [this]() { return GenMemL0CToL1(); }},

        // L1 <-> L0
        {Opcode::OP_L1_TO_L0A, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_L0B, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_L0_BT, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_L0_AT, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_L0A_SCALE, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_L0B_SCALE, [this]() { return GenMemL1ToL0(); }},
        {Opcode::OP_L1_TO_BT, [this]() { return GenMemL1ToBt(); }},
        {Opcode::OP_LOAD3D_CONV, [this]() { return GenMemL1ToL0Load3D(); }},
        {Opcode::OP_LOAD2D_CONV, [this]() { return GenMemL1ToL0Load2D(); }},

        // transpose with gm
        {Opcode::OP_TRANSPOSE_MOVEOUT, [this]() { return GenTransposeDataMove(); }},
        {Opcode::OP_TRANSPOSE_MOVEIN, [this]() { return GenTransposeDataMove(); }},

        // index outcast
        {Opcode::OP_INDEX_OUTCAST, [this]() { return GenIndexOutCastOp(); }},
        // lOC -> UB
        {Opcode::OP_L0C_COPY_UB, [this]() { return GenL0CToUBTileTensor(); }},

        {Opcode::OP_UB_COPY_L1, [this]() { return GenUBToL1TileTensor(); }},
        {Opcode::OP_UB_COPY_ND2NZ, [this]() { return GenUBToUBND2NZTileTensor(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> unaryOpsCloudNPU_ = {
        // cast op
        {Opcode::OP_CAST, [this]() { return GenCastOp(); }},

        // unary op
        {Opcode::OP_EXP, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_NEG, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_RSQRT, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_RELU, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_BITWISENOT, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_SQRT, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_CEIL, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_FLOOR, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_TRUNC, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_EXPAND, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ONEHOT, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_RECIPROCAL, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWSUM, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWMAX, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWEXPSUM, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWEXPMAX, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_COPY_UB_TO_UB, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWMAXLINE, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWMINLINE, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ROWPRODLINE, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_ABS, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_LN, [this]() { return GenUnaryOp(); }},
        {Opcode::OP_BRCB, [this]() { return GenUnaryOp(); }},

        // unary with temp buffer
        {Opcode::OP_COMPACT, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_EXP2, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_EXPM1, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROUND, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWSUMLINE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWARGMAXLINE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWARGMINLINE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWSUM_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWARGMAX_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWARGMIN_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWMAX_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWMIN_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ISFINITE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWPROD_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_TRANSPOSE_VNCHWCONV, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_SIGN, [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_SIGNBIT, [this]() { return GenUnaryOpWithTmpBuff(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> binaryOpsCloudNPU_ = {
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
        {Opcode::OP_PAIRPROD, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_BITWISEAND, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_BITWISEOR, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_EXPANDEXPDIF, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_GCD, [this]() { return GenBinaryOp(); }},

        // binary op: vector operations with tmp
        {Opcode::OP_MOD, [this]() { return GenBinaryOp(); }},
        {Opcode::OP_POW, [this]() { return GenBinaryOpWithTmp(); }},
        {Opcode::OP_REM, [this]() { return GenBinaryOpWithTmp(); }},
        {Opcode::OP_BITWISERIGHTSHIFT, [this]() { return GenBinaryOpWithTmp(); }},
        {Opcode::OP_BITWISELEFTSHIFT, [this]() { return GenBinaryOpWithTmp(); }},
        {Opcode::OP_BITWISEXOR, [this]() { return GenBinaryOpWithTmp(); }},
        {Opcode::OP_COPYSIGN, [this]() { return GenBinaryOpWithTmp(); }},
        {Opcode::OP_PRELU, [this]() { return GenPreluOp(); }},
        {Opcode::OP_FLOORDIV, [this]() { return GenBinaryOpWithTmp(); }},

        // binary op: broadcast associated vector
        {Opcode::OP_ADD_BRC, [this]() { return GenBinaryWithBrc(); }},
        {Opcode::OP_SUB_BRC, [this]() { return GenBinaryWithBrc(); }},
        {Opcode::OP_MUL_BRC, [this]() { return GenBinaryWithBrc(); }},
        {Opcode::OP_DIV_BRC, [this]() { return GenBinaryWithBrc(); }},
        {Opcode::OP_MAX_BRC, [this]() { return GenBinaryWithBrc(); }},
        {Opcode::OP_MIN_BRC, [this]() { return GenBinaryWithBrc(); }},
        {Opcode::OP_GCD_BRC, [this]() { return GenBinaryWithBrc(); }},

        // binary op: vector scalar
        {Opcode::OP_ADDS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_SUBS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_MULS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_DIVS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_MAXS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_MINS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_LRELU, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_BITWISEANDS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_BITWISEORS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_BITWISERIGHTSHIFTS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_BITWISELEFTSHIFTS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_GCDS, [this]() { return GenVectorScalarOp(); }},

        // binary op: vector scalar with tmp
        {Opcode::OP_MODS, [this]() { return GenVectorScalarOp(); }},
        {Opcode::OP_REMRS, [this]() { return GenRemainderSOp(); }},
        {Opcode::OP_REMS, [this]() { return GenRemainderSOp(); }},
        {Opcode::OP_SBITWISERIGHTSHIFT, [this]() { return GenVectorScalarOpWithTmp(); }},
        {Opcode::OP_SBITWISELEFTSHIFT, [this]() { return GenVectorScalarOpWithTmp(); }},
        {Opcode::OP_BITWISEXORS, [this]() { return GenVectorScalarOpWithTmp(); }},
        {Opcode::OP_FLOORDIVS, [this]() { return GenVectorScalarOpWithTmp(); }},

        // binary op: vector scalar, scalar mode
        {Opcode::OP_S_ADDS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {Opcode::OP_S_SUBS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {Opcode::OP_S_MULS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {Opcode::OP_S_DIVS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {Opcode::OP_S_MAXS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {Opcode::OP_S_MINS, [this]() { return GenVectorScalarOpScalarMode(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> compositeOpsCloudNPU_ = {
        // range op
        {Opcode::OP_RANGE, [this]() { return GenRangeOp(); }},

        // logicalnot
        {Opcode::OP_LOGICALNOT, [this]() { return GenLogicalNotOp(); }},
        // logicaland
        {Opcode::OP_LOGICALAND, [this]() { return GenLogicalAndOp(); }},

        // indexadd
        {Opcode::OP_INDEX_ADD, [this]() { return GenIndexAddOp(); }},

        // indexput
        {Opcode::OP_INDEX_PUT, [this]() { return GenIndexPutOp(); }},

        // cumOperation
        {Opcode::OP_CUM_SUM, [this]() { return GenCumOperationOp(); }},
        {Opcode::OP_CUM_PROD, [this]() { return GenCumOperationOp(); }},

        // triUL
        {Opcode::OP_TRIUL, [this]() { return GenTriULOp(); }},

        // vector where
        {Opcode::OP_WHERE_SS, [this]() { return GenWhereOp(); }},
        {Opcode::OP_WHERE_TS, [this]() { return GenWhereOp(); }},
        {Opcode::OP_WHERE_ST, [this]() { return GenWhereOp(); }},
        {Opcode::OP_WHERE_TT, [this]() { return GenWhereOp(); }},

        // cmp op
        {Opcode::OP_CMP, [this]() { return GenCmpOp(); }},
        {Opcode::OP_CMPS, [this]() { return GenCmpOp(); }},

        // hypot op
        {Opcode::OP_HYPOT, [this]() { return GenHypotOp(); }},
        {Opcode::OP_PAD, [this]() { return GenPadOp(); }},
        {Opcode::OP_FILLPAD, [this]() { return GenPadOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> sortOpsCloudNPU_ = {
        // sort
        {Opcode::OP_BITSORT, [this]() { return GenBitSortOp(); }},
        {Opcode::OP_MRGSORT, [this]() { return GenMrgSortOp(); }},
        {Opcode::OP_EXTRACT, [this]() { return GenExtractOp(); }},
        {Opcode::OP_TILEDMRGSORT, [this]() { return GenTiledMrgSortOp(); }},

        {Opcode::OP_TOPK_SORT, [this]() { return GenTopKSortOp(); }},
        {Opcode::OP_TOPK_MERGE, [this]() { return GenTopKMergeOp(); }},
        {Opcode::OP_TOPK_EXTRACT, [this]() { return GenTopKExtractOp(); }},

        // parallel sort
        {Opcode::OP_SORT, [this]() { return GenSortOp(); }},
        {Opcode::OP_COMPARE_SWAP, [this]() { return GenCompareAndSwapOp(); }},
        {Opcode::OP_MERGE, [this]() { return GenMergeOp(); }},

        {Opcode::OP_TWOTILEMRGSORT, [this]() { return GenTwoTileMrgSort(); }},
        {Opcode::OP_EXTRACT_SINGLE, [this]() { return GenExtractSingleOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> cubeOpsCloudNPU_ = {
        // matmul
        {Opcode::OP_A_MUL_B, [this]() { return GenCubeOpMatmul(); }},
        {Opcode::OP_A_MUL_BT, [this]() { return GenCubeOpMatmul(); }},
        {Opcode::OP_A_MULACC_B, [this]() { return GenCubeOpMatmulAcc(); }},
        {Opcode::OP_A_MULACC_BT, [this]() { return GenCubeOpMatmulAcc(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> syncOpsCloudNPU_ = {
        // sync
        {Opcode::OP_SYNC_SRC, [this]() { return GenSyncSetOp(); }},
        {Opcode::OP_SYNC_DST, [this]() { return GenSyncWaitOp(); }},
        {Opcode::OP_BAR_V, [this]() { return GenBarrier(); }},
        {Opcode::OP_BAR_M, [this]() { return GenBarrier(); }},
        {Opcode::OP_BAR_ALL, [this]() { return GenBarrier(); }},
        {Opcode::OP_CV_SYNC_SRC, [this]() { return GenCVSyncSetOp(); }},
        {Opcode::OP_CV_SYNC_DST, [this]() { return GenCVSyncWaitOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> distributeOpsCloudNPU_ = {
        // distribute op
        {Opcode::OP_FFN_SCHED, [this]() { return GenDistOp(); }},
        {Opcode::OP_FFN_BATCHING, [this]() { return GenDistOp(); }},
        {Opcode::OP_FFN_COMBINEINFO, [this]() { return GenDistOp(); }},
        {Opcode::OP_FFN_VALIDCNT, [this]() { return GenDistOp(); }},
        {Opcode::OP_SEND_TO_ROUTING_EXPERT, [this]() { return GenDistOp(); }},
        {Opcode::OP_SEND_TO_SHARED_EXPERT, [this]() { return GenDistOp(); }},
        {Opcode::OP_DISPATCH_SET_FLAG, [this]() { return GenDistOp(); }},
        {Opcode::OP_COPY_TO_LOCAL_EXPERT, [this]() { return GenDistOp(); }},
        {Opcode::OP_SHMEM_SET, [this]() { return GenDistOp(); }},
        {Opcode::OP_SHMEM_PUT, [this]() { return GenDistOp(); }},
        {Opcode::OP_SHMEM_PUT_UB2GM, [this]() { return GenDistOp(); }},
        {Opcode::OP_SHMEM_SIGNAL, [this]() { return GenDistOp(); }},
        {Opcode::OP_SHMEM_GET, [this]() { return GenDistOp(); }},
        {Opcode::OP_SHMEM_GET_GM2UB, [this]() { return GenDistOp(); }},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_SEND, [this]() { return GenDistOp(); }},
        {Opcode::OP_MOE_DISTRIBUTED_COMBINE_RECEIVE, [this]() { return GenDistOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> gatherScatterOpsCloudNPU_ = {
        // gather/scatter op
        {Opcode::OP_GATHER_FROM_UB, [this]() { return GenGatherFromUBOp(); }},
        {Opcode::OP_GATHER_ELEMENT, [this]() { return GenGatherElementOp(); }},
        {Opcode::OP_SCATTER_ELEMENT, [this]() { return GenScatterElementSOp(); }},
        {Opcode::OP_SCATTER, [this]() { return GenScatterOp(); }},
        {Opcode::OP_GATHER_MASK, [this]() { return GenGatherMaskOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> normalVecOpsCloudNPU_ = {
        // vector dup
        {Opcode::OP_VEC_DUP, [this]() { return GenDupOp(); }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> perfOpsCloudNPU_ = {
        // for performace optimization
        {Opcode::OP_PHASE1, []() { return "SUBKERNEL_PHASE1\n"; }},
        {Opcode::OP_PHASE2, []() { return "SUBKERNEL_PHASE2\n"; }},
    };

    std::unordered_map<Opcode, std::function<std::string()>> aicpuOpsCloudNPU_ = {
        // for aicpu call
        {Opcode::OP_AICPU_CALL_AIC, [this]() { return GenAicpuCallOp(); }},
        {Opcode::OP_AICPU_CALL_AIV, [this]() { return GenAicpuCallOp(); }},
    };
};
} // namespace npu::tile_fwk

#endif // CODEGEN_OP_CLOUDNPU_H
