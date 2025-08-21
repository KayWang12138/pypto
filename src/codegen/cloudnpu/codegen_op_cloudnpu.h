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

#ifndef CODEGEN_OP_CLOUDNPU_H
#define CODEGEN_OP_CLOUDNPU_H

#include <utility>
#include <unordered_set>

#include "codegen/codegen_common.h"
#include "tilefwk/data_type.h"
#include "interface/operation/operation.h"
#include "interface/function/function.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "codegen/symbol_mgr/codegen_symbol.h"
#include "codegen/codegen_op.h"

namespace npu::tile_fwk {
class CodeGenOpCloudNPU : public CodeGenOp {
public:
    explicit CodeGenOpCloudNPU(SymbolManager &symbolManager, FunctionType funcType,
        const std::map<int, int> &locToOffset = {}, bool isUnderDynamicFunc = false)
        : CodeGenOp(symbolManager, funcType, locToOffset, isUnderDynamicFunc){};
    ~CodeGenOpCloudNPU() override = default;

    std::string GenMemL1CopyIn() const;
    std::string GenMemL1CopyOut() const;
    std::string GenMemL0CCopyOut() const;

    std::string GenMemL1ToL0() const;

    std::string GenUBCopyIn() const;
    std::string GenUBCopyOut() const;

    std::string GenUnaryOp() const;
    std::string GenUnaryOpWithTmpBuff() const;

    std::string GenBinaryOp() const;
    std::string GenVectorScalarOp() const;

    std::string GenCubeOpMatmul() const ;
    std::string GenCubeOpMatmulAcc() const ;

    std::string GenCastOp() const ;

    std::string GenDupOp() const ;

    std::string GenTransposeDataMove() const ;

    std::string GenGatherElementOp() const ;

    std::string GenScatterElementOp() const ;

    std::string GenIndexOutCastOp() const ;

    std::string GenFusedOp() const ;

    std::string GenGatherOp() const ;

    std::string GenMemCopyCube(
        const struct OpInfo &opInfo, bool isCopyL0CToGM, bool isCopyL1ToGM, unsigned uf = 0) const;
    std::string GenMemL1SpillIntoGM(const OpInfo &opInfo, bool isCopyL0CToGM, bool isCopyL1ToGM, unsigned uf) const;

    std::string GenBinaryWithBrc() const;

    std::string GenBitSortOp() const;
    std::string GenMrgSortOp() const;
    std::string GenExtractOp() const;

    std::string GenParamsStr() const;

    std::string GenDistOp() const;
    std::string GetTemplateDType() const;

    std::string GenPoolOp() const;

    std::string GenOpCode() const override {
        auto iter = opsGenMap_.find(opCode);
        if (iter != opsGenMap_.end()) {
            return iter->second();
        }
        // To aid in testing, do not use ASSERT.
        return std::string{"CAN NOT HANDLE OP: " + opCodeStr};
    }

private:
    template <typename T>
    bool GetAttr(const std::string &key, T &value) const;

    std::vector<int> GetTileShapeForMemTransfer(
        OperandType localType, std::vector<int> gmShape, unsigned localIdx) const;
    std::string GenMemCopyVar(bool isCopyLocalToGM, OperandType localType, unsigned uf = 0) const;

    std::string GenGMAddrExprWithOffset(const std::string &addrExpr, unsigned gmIdx) const;
    std::string GenAddrExpr(const std::string &addrExpr, unsigned offsetParam) const;

    // update var offset when parent of this var is split by "view" operation. (used for ub var currently)
    void AppendLocalBufferVarOffset(
        const std::vector<std::string *> &vars, const std::vector<unsigned> &operandIdxes) const;

    std::string GenGmParamVar(unsigned gmParamIdx) const;

    bool CombineAxis(std::vector<std::vector<int> *> &shapes, bool secondLastAxis = false) const;

    std::vector<std::string> GenGetParamMacroPacked(unsigned gmParamIdx, int dim, const std::string &prefix) const;

    std::vector<std::string> GenParamIdxExprByIndex(unsigned gmParamIdx, int dim, const std::string &prefix) const;

    std::vector<std::string> GenSymbolicArgument(const std::vector<SymbolicScalar> &exprList) const;

    std::string GenMemUBTransfer(bool isCopyUBToGM) const;
    std::string GenMemUBSpillIntoGM(bool isCopyUBToGM) const;
    std::string GenVectorScalarOpByMode(VecScalMode mode) const;
    std::string GenVectorScalarOpScalarMode() const;
    std::string GenCubeOp(bool zeroC) const;

    struct PrintDupOpParam {
        const std::string &dVar;
        const std::string &dstDtypeStr;
        const std::string &dupV;
    };
    std::string PrintDupOp(const PrintDupOpParam &param) const;
    std::string PrintDupOpDynUnaligned(const PrintDupOpParam &param) const;
    std::string PrintDupOpStatic(const PrintDupOpParam &param) const;

    struct PrintUnaryParam {
        const std::string &s0Var;
        const std::string &dVar;
        const std::string &srcDtypeStr;
        const std::string &dstDtypeStr;
    };
    std::string PrintRowSumline(const PrintUnaryParam &param) const;
    std::string PrintRowSumlineDynamicUnaligned(const PrintUnaryParam &param) const;
    std::string PrintRowSumlineStatic(const PrintUnaryParam &param) const;

    std::string PrintReduceEx(const PrintUnaryParam &param) const;
    std::string PrintReduceExStatic(const PrintUnaryParam &param) const;

    std::string PrintReduceSum(const PrintUnaryParam &param) const;
    std::string PrintReduceSumStatic(const PrintUnaryParam &param) const;

    std::string PrintVcopy(const PrintUnaryParam &param) const;
    std::string PrintVcopyStatic(const PrintUnaryParam &param) const;

    struct PrintUnaryTmpBuffParam {
        const std::string &s0Var;
        const std::string &tmpVar;
        const std::string &dVar;
        const std::string &srcDtypeStr;
        const std::string &tmpDtypeStr;
        const std::string &dstDtypeStr;
    };
    std::string PrintVnchwconv(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintVnchwconvDynUnaligned(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintVnchwconvStatic(const PrintUnaryTmpBuffParam &param) const;

    std::string PrintCompact(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintCompactStatic(const PrintUnaryTmpBuffParam &param) const;

    struct PrintMemCopyWithL0CParam {
        unsigned uf;
        unsigned gmIdx;
        unsigned localIdx;
        const std::string *addrTypeHead;
        const std::string *addrExpr;
        const std::vector<int> &gmShape;
        const std::vector<int> &tileShapeForMT;
        const std::string *dataTypeExpr;
    };
    std::string PrintMemCopyWithL0C(const PrintMemCopyWithL0CParam &param) const;
    std::string PrintMemCopyWithL0CStatic(const PrintMemCopyWithL0CParam &param) const;
    std::string PrintMemCopyWithL0CDynamic(const PrintMemCopyWithL0CParam &param) const;
    std::string PrintL0CCopyOutDynamicUnalign(const PrintMemCopyWithL0CParam &param,
        std::vector<std::string> &gmShapeExpr, std::vector<std::string> &gmOffsetExpr) const;

    struct PrintMemCopyWithL1Param {
        unsigned uf;
        unsigned gmIdx;
        unsigned localIdx;
        const std::string *addrTypeHead;
        const std::string *addrExpr;
        const std::vector<int> &gmShape;
        const std::vector<int> &tileShapeForMT;
        const std::string *dataTypeExpr;
    };
    std::string PrintMemCopyWithL1(const PrintMemCopyWithL1Param &param) const;
    std::string PrintMemCopyWithL1Static(const PrintMemCopyWithL1Param &param) const;
    std::string PrintMemCopyWithL1Dynamic(const PrintMemCopyWithL1Param &param) const;

    struct PrintMemCopyWithUBParam {
        unsigned gmIdx;
        unsigned localIdx;
        const std::string *addrTypeHead;
        std::string *addrExpr;
        std::string *dataTypeExpr;
        bool isSpillIntoGM;
    };
    std::string PrintMemCopyWithUB(PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBStatic(const PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBDynamic(const PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBDynamicSupportUnaligned(const PrintMemCopyWithUBParam &param) const;

    struct PrintGatherParam {
        const std::string &s0Var;
        const std::string &s1Var;
        const std::string &dVar;
        const std::string &src0DtypeStr;
        const std::string &src1DtypeStr;
        const std::string &dstDtypeStr;
    };
    std::string PrintGather(const PrintGatherParam &param) const;
    std::string PrintGatherDynamicUnaligned(const PrintGatherParam &param) const;
    std::string PrintGatherStatic(const PrintGatherParam &param) const;

    struct PrintBinaryScalarParam {
        const std::string &s0Var;
        const std::string &dVar;
        const std::string &src0DtypeStr;
        const std::string &dstDtypeStr;
        const size_t dim;
    };
    std::string PrintBinaryScalar(const PrintBinaryScalarParam &param) const;
    std::string PrintBinaryScalarDynamicUnaligned(const PrintBinaryScalarParam &param) const;
    std::string PrintBinaryScalarStatic(const PrintBinaryScalarParam &param) const;

    std::string PrintUnary(const PrintUnaryParam &param) const;
    std::string PrintUnaryDynamicUnaligned(const PrintUnaryParam &param) const;
    std::string PrintUnaryStatic(const PrintUnaryParam &param) const;

    SortParam PrepareSortParam() const;
    std::string PrintSortDynamicUnaligned(const SortParam &param) const;
    std::string PrintSortStatic(const SortParam &param) const;
    std::string PrintBitSortDynamicUnaligned(const SortParam &param) const;
    std::string PrintBitSortStatic(const SortParam &param) const;
    std::string PrintMrgSortDynamicUnaligned(const SortParam &param) const;
    std::string PrintMrgSortStatic(const SortParam &param) const;

    struct PrintBinaryParam {
        const std::string &s0Var;
        const std::string &s1Var;
        const std::string &dVar;
        const std::string &src0DtypeStr;
        const std::string &src1DtypeStr;
        const std::string &dstDtypeStr;
    };
    std::string PrintBinaryStatic(const PrintBinaryParam &param) const;
    std::string PrintBinaryDynamicUnaligned(const PrintBinaryParam &param) const;
    std::string PrintBinary(const PrintBinaryParam &param) const;

    struct PrintBinaryBrcParam {
        const std::string &s0Var;
        const std::string &s1Var;
        const std::string &dVar;
        const std::string &tmpVar;
        const std::string &src0DtypeStr;
        const std::string &src1DtypeStr;
        const std::string &dstDtypeStr;
        const std::string &tmpDtypeStr;
    };
    std::string PrintBinaryBrcStatic(const PrintBinaryBrcParam &param) const;
    std::string PrintBinaryBrcDynamicUnaligned(const PrintBinaryBrcParam &param) const;
    std::string PrintBinaryBrc(const PrintBinaryBrcParam &param) const;

    struct PrintTransposeDataMoveParam {
        const std::string &s0Var;
        const std::vector<int> &dstShape;
        const std::string &srcDtypeStr;
        const std::string &dstDtypeStr;
    };

    std::string PrintTransposeDataMove(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveStatic(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveDynamic(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveDynamicUnaligned(const PrintTransposeDataMoveParam &param) const;

    struct PrintGatherEleParam {
        int axis;
        const std::string &dVar;
        const std::string &s0Var;
        const std::string &s1Var;
        std::vector<int> &dstOriginShape;
        std::vector<int> &dstRawShape;
        std::vector<int> &src0RawShape;
        const std::string *dataTypeExpr;
    };
    std::string PrintGatherElementDynamicUnaligned(const PrintGatherEleParam &param) const;
    std::string PrintGatherElementStatic(const PrintGatherEleParam &param) const;

    struct PrintIndexOutCastParam {
        const std::string &s0Var;
        const std::string &s1Var;
        const std::string *addrExpr;
        const std::vector<int> &gmShape;
        std::vector<int> &src0OriginShape;
        std::vector<int> &src0RawShape;
        std::vector<int> &src1OriginShape;
        std::vector<int> &src1RawShape;
        const std::string *dataTypeExpr;
        const std::string &cacheMode;
        const std::string &blockSize;
    };
    std::string PrintIndexOutCast(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastStatic(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastDynamic(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastDynamicUnaligned(const PrintIndexOutCastParam &param) const;

    std::string PrintExpandDynamicUnaligned(const PrintUnaryParam &param, int expandAxis) const;
    std::string PrintExpand(const std::string &s0Var, const std::string &dVar, const std::string &srcDtypeStr,
        const std::string &dstDtypeStr) const;

    struct DynamicParamPackMTE {
        std::vector<std::string> gmShapeExpr;
        std::vector<std::string> gmOffsetExpr;
        std::vector<std::string> paramList;
    };

    DynamicParamPackMTE PrepareDynamicShapeInfoforMTE(
        int dynShapeIdx, int ShapeDim = SHAPE_DIM4, bool isNeedGmOffset = true) const;

    std::string PrintReduceLastAxis(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintReduceLastAxisDynamicUnalign(const PrintUnaryTmpBuffParam &param) const;

    std::string PrintExtractStatic() const;
    std::string PrintExtractDynamicUnaligned() const;

    std::string PrintCastDynamicUnaligned(const PrintUnaryParam &param) const;
    std::string PrintReduceCombine(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintVectorScalarOpDynamicUnalign(const PrintUnaryParam &param) const;

    const std::unordered_map<Opcode, std::function<std::string()>> opsGenMap_ = {
        // UB <-> GM
        {                Opcode::OP_UB_COPY_IN,                 [this]() { return GenUBCopyIn(); }},
        {               Opcode::OP_UB_COPY_OUT,                [this]() { return GenUBCopyOut(); }},

        // L1 <-> GM/BT/L1
        {                Opcode::OP_L1_COPY_IN,              [this]() { return GenMemL1CopyIn(); }},
        {               Opcode::OP_L1_COPY_OUT,             [this]() { return GenMemL1CopyOut(); }},

        // L0C <-> GM
        {              Opcode::OP_L0C_COPY_OUT,            [this]() { return GenMemL0CCopyOut(); }},

        // L1 <-> L0
        {                 Opcode::OP_L1_TO_L0A,                [this]() { return GenMemL1ToL0(); }},
        {                 Opcode::OP_L1_TO_L0B,                [this]() { return GenMemL1ToL0(); }},
        {               Opcode::OP_L1_TO_L0_BT,                [this]() { return GenMemL1ToL0(); }},

        // cast op
        {                      Opcode::OP_CAST,                   [this]() { return GenCastOp(); }},

        // binary op: vector operations
        {                       Opcode::OP_ADD,                 [this]() { return GenBinaryOp(); }},
        {                       Opcode::OP_SUB,                 [this]() { return GenBinaryOp(); }},
        {                       Opcode::OP_MUL,                 [this]() { return GenBinaryOp(); }},
        {                       Opcode::OP_DIV,                 [this]() { return GenBinaryOp(); }},
        {                   Opcode::OP_MAXIMUM,                 [this]() { return GenBinaryOp(); }},
        {                   Opcode::OP_PAIRSUM,                 [this]() { return GenBinaryOp(); }},
        {                   Opcode::OP_PAIRMAX,                 [this]() { return GenBinaryOp(); }},

        // binary op: broadcast associated vector
        {                   Opcode::OP_ADD_BRC,            [this]() { return GenBinaryWithBrc(); }},
        {                   Opcode::OP_SUB_BRC,            [this]() { return GenBinaryWithBrc(); }},
        {                   Opcode::OP_MUL_BRC,            [this]() { return GenBinaryWithBrc(); }},
        {                   Opcode::OP_DIV_BRC,            [this]() { return GenBinaryWithBrc(); }},
        {                   Opcode::OP_MAX_BRC,            [this]() { return GenBinaryWithBrc(); }},

        // binary op: vector scalar
        {                      Opcode::OP_ADDS,           [this]() { return GenVectorScalarOp(); }},
        {                      Opcode::OP_SUBS,           [this]() { return GenVectorScalarOp(); }},
        {                      Opcode::OP_MULS,           [this]() { return GenVectorScalarOp(); }},
        {                      Opcode::OP_DIVS,           [this]() { return GenVectorScalarOp(); }},

        // binary op: vector scalar, scalar mode
        {                    Opcode::OP_S_ADDS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {                    Opcode::OP_S_SUBS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {                    Opcode::OP_S_MULS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {                    Opcode::OP_S_DIVS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {                    Opcode::OP_S_MAXS, [this]() { return GenVectorScalarOpScalarMode(); }},
        {                    Opcode::OP_S_MINS, [this]() { return GenVectorScalarOpScalarMode(); }},

        // unary op
        {                       Opcode::OP_EXP,                  [this]() { return GenUnaryOp(); }},
        {                      Opcode::OP_SQRT,                  [this]() { return GenUnaryOp(); }},
        {                    Opcode::OP_EXPAND,                  [this]() { return GenUnaryOp(); }},
        {                Opcode::OP_RECIPROCAL,                  [this]() { return GenUnaryOp(); }},
        {                    Opcode::OP_ROWSUM,                  [this]() { return GenUnaryOp(); }},
        {                    Opcode::OP_ROWMAX,                  [this]() { return GenUnaryOp(); }},
        {                 Opcode::OP_ROWEXPSUM,                  [this]() { return GenUnaryOp(); }},
        {                 Opcode::OP_ROWEXPMAX,                  [this]() { return GenUnaryOp(); }},
        {             Opcode::OP_COPY_UB_TO_UB,                  [this]() { return GenUnaryOp(); }},
        {                Opcode::OP_ROWSUMLINE,                  [this]() { return GenUnaryOp(); }},
        {                Opcode::OP_ROWMAXLINE,                  [this]() { return GenUnaryOp(); }},
        {                       Opcode::OP_ABS,                  [this]() { return GenUnaryOp(); }},

        // unary with temp buffer
        {                   Opcode::OP_COMPACT,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {             Opcode::OP_ROWSUM_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {             Opcode::OP_ROWMAX_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {       Opcode::OP_TRANSPOSE_VNCHWCONV,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},

        // gather/scatter op
        {                    Opcode::OP_GATHER,                 [this]() { return GenGatherOp(); }},
        {            Opcode::OP_GATHER_ELEMENT,          [this]() { return GenGatherElementOp(); }},
        {           Opcode::OP_SCATTER_ELEMENT,         [this]() { return GenScatterElementOp(); }},

        // transpose with gm
        {        Opcode::OP_TRANSPOSE_MOVEOUT,        [this]() { return GenTransposeDataMove(); }},

        // vector dup
        {                   Opcode::OP_VEC_DUP,                    [this]() { return GenDupOp(); }},

        // index outcast
        {             Opcode::OP_INDEX_OUTCAST,           [this]() { return GenIndexOutCastOp(); }},

        // sort
        {                   Opcode::OP_BITSORT,                [this]() { return GenBitSortOp(); }},
        {                   Opcode::OP_MRGSORT,                [this]() { return GenMrgSortOp(); }},
        {                   Opcode::OP_EXTRACT,                [this]() { return GenExtractOp(); }},

        // matmul
        {                   Opcode::OP_A_MUL_B,             [this]() { return GenCubeOpMatmul(); }},
        {                  Opcode::OP_A_MUL_BT,             [this]() { return GenCubeOpMatmul(); }},
        {                Opcode::OP_A_MULACC_B,          [this]() { return GenCubeOpMatmulAcc(); }},
        {               Opcode::OP_A_MULACC_BT,          [this]() { return GenCubeOpMatmulAcc(); }},

        // sync
        {                  Opcode::OP_SYNC_SRC,                [this]() { return GenSyncSetOp(); }},
        {                  Opcode::OP_SYNC_DST,               [this]() { return GenSyncWaitOp(); }},
        {                     Opcode::OP_BAR_V,                  [this]() { return GenBarrier(); }},
        {                     Opcode::OP_BAR_M,                  [this]() { return GenBarrier(); }},
        {                   Opcode::OP_BAR_ALL,                  [this]() { return GenBarrier(); }},

        // distribute op
        {              Opcode::OP_WRITE_REMOTE,                   [this]() { return GenDistOp(); }},
        {             Opcode::OP_REMOTE_REDUCE,                   [this]() { return GenDistOp(); }},
        {             Opcode::OP_REMOTE_GATHER,                   [this]() { return GenDistOp(); }},
        {            Opcode::OP_LOCAL_COPY_OUT,                   [this]() { return GenDistOp(); }},
        {           Opcode::OP_MOE_FFN_TO_ATTN,                   [this]() { return GenDistOp(); }},
        {          Opcode::OP_MOE_ATTN_COMBINE,                   [this]() { return GenDistOp(); }},
        {                 Opcode::OP_FFN_SCHED,                   [this]() { return GenDistOp(); }},
        {              Opcode::OP_FFN_BATCHING,                   [this]() { return GenDistOp(); }},
        {    Opcode::OP_SEND_TO_ROUTING_EXPERT,                   [this]() { return GenDistOp(); }},
        {     Opcode::OP_SEND_TO_SHARED_EXPERT,                   [this]() { return GenDistOp(); }},
        {         Opcode::OP_DISPATCH_SET_FLAG,                   [this]() { return GenDistOp(); }},
        {      Opcode::OP_COPY_TO_LOCAL_EXPERT,                   [this]() { return GenDistOp(); }},

        // max pool
        {                  Opcode::OP_MAX_POOL,                   [this]() { return GenPoolOp(); }},

        // fused pool
        {                  Opcode::OP_FUSED_OP,                  [this]() { return GenFusedOp(); }},

        // for performace optimization
        {                    Opcode::OP_PHASE1,              []() { return "SUBKERNEL_PHASE1\n"; }},
        {                    Opcode::OP_PHASE2,              []() { return "SUBKERNEL_PHASE2\n"; }},
    };
};

} // namespace npu::tile_fwk

#endif // CODEGEN_OP_CLOUDNPU_H
