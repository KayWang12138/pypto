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
class CodeGenOpLiteNPU : public CodeGenOp {
public:
    explicit CodeGenOpLiteNPU(const CodeGenOpCtx &ctx);
    ~CodeGenOpLiteNPU() override = default;

    std::string GenMemL1CopyIn() const;
    std::string GenMemL0CCopyOut() const;

    std::vector<std::string> GeTileOpParamForNormalCopyTileTensor(
        unsigned gmIdx, const std::string &gmVarName, bool isSpillingToGM) const;

    std::string PrintCoord(size_t dim, const std::string &coord) const;
    std::string PrintTensorForCopyBetweenGM(unsigned operandIdx, unsigned gmIdx, const std::string &gmVarName) const;

    std::string PrintMemCopyInWithL1TileTensor(const PrintMemCopyWithL1Param &param) const;
    std::string PrintMemCopyWithL1TileTensor(const PrintMemCopyWithL1Param &param) const;

    std::string PrintVectorScalarTileTensor(const PrintUnaryParam &param) const;

    std::vector<std::string> GetGmOffsetForTileTensor(unsigned gmIdx, bool isSpillingToGM = false) const;

    std::string GenMemL1ToL0() const;

    std::string GenUBCopyIn() const;
    std::string GenUBCopyOut() const;

    std::string GenUnaryOp() const;
    std::string GenUnaryOpWithTmpBuff() const;

    std::string GenBinaryOp() const;
    std::string GenVectorScalarOp() const;

    std::string PrintMemL1ToL0TileTensor() const;
    std::string PrintMatmulTileTensor(bool isAcc) const;
    std::string PrintMatmulTileTensor(
        bool isAcc, std::unordered_map<OperandType, std::string> &tensorWithMemType) const;

    std::string GenCubeOpMatmul() const ;
    std::string GenCubeOpMatmulAcc() const ;

    std::string PrintCmpTileTensor() const;

    std::string GenCmpOp() const;
    
    std::string GenCastOp() const;

    std::string GenDupOp() const ;

    std::string GenMemCopyCube(bool isLocalToGM, unsigned uf = 0) const;

    std::string GenWhereOp() const;

    std::string GenOpCode() const override {
        auto iter = opsGenMap_.find(opCode);
        if (iter != opsGenMap_.end()) {
            return iter->second();
        }
        // To aid in testing, do not use ASSERT.
        return std::string{"CAN NOT HANDLE OP: " + opCodeStr + "\n"};
    }
    void UpdateTileTensorInfo();

private:
    // <parameter index, tensor name>
    std::unordered_map<int, std::string> tensorNames_;
    template <typename T>
    bool GetAttr(const std::string &key, T &value) const {
        auto it = opAttrs.find(key);
        if (it == opAttrs.end()) {
            CODEGEN_LOGI("can not find key: %s in opAttrs", key.c_str());
            return false;
        }
        if (it->second.Type() == typeid(T)) {
            value = AnyCast<T>(it->second);
            return true;
        }
        CODEGEN_LOGE("Type of attribute %s from PASS is mismatch: %s != %s", key.c_str(), it->second.Type().name(),
            typeid(T).name());
        return false;
    }

    template <typename T = int64_t>
    std::vector<T> GetVectorIntAttribute(const std::string &key) const {
        static_assert(std::is_integral_v<T>);
        std::vector<int64_t> val;
        GetAttr(key, val);
        if constexpr (std::is_same_v<T, int64_t>) {
            return val;
        }
        std::vector<T> ret;
        for (auto &x : val) {
            ret.emplace_back(static_cast<T>(x));
        }
        return ret;
    }

    std::string PrintRowMaxline() const;
    std::string PrintRowMaxlineTileTensor() const;

    TileTensor QueryTileTensorByIdx(int paramIdx) const;

    std::string GenBarrier() const;
    std::string GenSyncSetOp() const;
    std::string GenSyncWaitOp() const;

    std::string GetLastUse() const;

    std::string QueryTileTensorNameByIdx(int paramIdx) const;

    TileTensor BuildTileTensor(int paramIdx, const std::string& usingType);
    void UpdateTileTensorShapeAndStride(
        int paramIdx, TileTensor &tileTensor, bool isSpillToGm);
    std::vector<std::string> BuildStride(const std::vector<int64_t> &input);

    std::vector<int64_t> GetTileShapeForMemTransfer(
        OperandType localType, std::vector<int64_t> gmShape, unsigned localIdx) const;
    std::string GenMemCopyVar(bool isCopyLocalToGM, bool isSpillToGm = false, unsigned uf = 0) const;

    std::string GenGMAddrExprWithOffset(const std::string& addrExpr) const;
    std::string GenAddrExpr(const std::string &addrExpr, unsigned offsetParam) const;

    // update var offset when parent of this var is split by "view" operation. (used for ub var currently)
    void AppendLocalBufferVarOffset(const std::map<unsigned, std::reference_wrapper<std::string>> &vars) const;
    SymbolicScalar GetOperandStartOffsetLite(int operandIdx) const;

    std::string GenGmParamVar(unsigned gmParamIdx) const;

    bool CombineAxis(std::vector<std::vector<int64_t> *> &shapes, bool secondLastAxis = false) const;

    std::vector<std::string> GenSymbolicArgument(const std::vector<SymbolicScalar> &exprList) const;

    std::string GenMemUBTransfer(bool isCopyUBToGM) const;
    std::string PrintMemCopyWithUBTileTensor(const PrintMemCopyWithUBParam &param) const;
    std::string GenVectorScalarOpByMode(VecScalMode mode) const;
    std::string GenVectorScalarOpScalarMode() const;
    std::string GenCubeOp(bool zeroC) const;

    std::string PrintDupOp(const PrintDupOpParam &param) const;
    std::string PrintDupTileTensor(const PrintDupOpParam &param) const;

    std::string PrintRowSumline(const PrintUnaryParam &param) const;
    std::string PrintRowSumlineDynamicUnaligned(const PrintUnaryParam &param) const;
    std::string PrintRowSumlineStatic(const PrintUnaryParam &param) const;

    std::string PrintReduceEx(const PrintUnaryParam &param) const;
    std::string PrintReduceExStatic(const PrintUnaryParam &param) const;

    std::string PrintReduceSum(const PrintUnaryParam &param) const;
    std::string PrintReduceSumStatic(const PrintUnaryParam &param) const;

    std::string PrintVcopy(const PrintUnaryParam &param) const;
    std::string PrintVcopyStatic(const PrintUnaryParam &param) const;

    std::string PrintIndexPut(const PrintIndexPutParam &param) const;
    std::string PrintIndexPutLayout(size_t indicesSize, bool accumulate) const;

    WhereParam PrepareWhereParam() const;
    void GetWhereVarAndType(std::vector<std::string> &varExpr, std::vector<std::string> &dataTypeExpr) const;
    std::string PrintWhereOp(const WhereParam &param) const;
    std::string PrintWhereOpTileTensor(const WhereParam &param) const;

    std::string PrintVnchwconv() const;
    std::string PrintVnchwconvDynUnaligned(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintVnchwconvStatic(const PrintUnaryTmpBuffParam &param) const;

    std::string PrintCompact(const PrintUnaryTmpBuffParam &param) const;
    std::string PrintCompactStatic(const PrintUnaryTmpBuffParam &param) const;

    std::string PrintMemCopyWithL0C(const PrintMemCopyWithL0CParam &param) const;
    std::string PrintMemCopyWithL0CTileTensor(const PrintMemCopyWithL0CParam &param) const;

    std::pair<std::string, std::string> GetOuterInnerValueStr(
        unsigned gmIdx, const std::vector<int64_t>& gmShape, bool isSpillingToGM = false) const;
    std::string PrintMemCopyWithL1(const PrintMemCopyWithL1Param &param) const;

    std::string PrintMemCopyWithUB(PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBStatic(const PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBDynamic(const PrintMemCopyWithUBParam &param) const;
    std::string PrintMemCopyWithUBDynamicSupportUnaligned(const PrintMemCopyWithUBParam &param) const;

    std::string PrintGather(const PrintGatherParam &param) const;
    std::string PrintGatherDynamicUnaligned(const PrintGatherParam &param) const;
    std::string PrintGatherStatic(const PrintGatherParam &param) const;

    std::string PrintBinaryScalar(const PrintBinaryScalarParam &param) const;
    std::string PrintBinaryScalarDynamicUnaligned(const PrintBinaryScalarParam &param) const;
    std::string PrintBinaryScalarStatic(const PrintBinaryScalarParam &param) const;

    std::string PrintUnary() const;
    std::string PrintUnaryTileTensor() const;

    std::string PrintCastTileTensor() const;

    std::string PrintBinary() const;
    std::string PrintBinaryTileTensor() const;
    std::string PrintUnaryWithTmpTileTensor() const;

    std::string PrintBinaryBrcStatic(const PrintBinaryBrcParam &param) const;
    std::string PrintBinaryBrcDynamicUnaligned(const PrintBinaryBrcParam &param) const;
    std::string PrintBinaryBrc(const PrintBinaryBrcParam &param) const;

    std::string GenIndexPutOp() const;

    std::string PrintTransposeDataMove(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveStatic(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveDynamic(const PrintTransposeDataMoveParam &param) const;
    std::string PrintTransposeDataMoveDynamicUnaligned(const PrintTransposeDataMoveParam &param) const;

    std::string PrintGatherElementDynamicUnaligned(const PrintGatherEleParam &param) const;
    std::string PrintGatherElementStatic(const PrintGatherEleParam &param) const;

    std::string PrintIndexOutCast(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastStatic(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastDynamic(const PrintIndexOutCastParam &param) const;
    std::string PrintIndexOutCastDynamicUnaligned(const PrintIndexOutCastParam &param) const;

    std::string PrintExpand() const;
    std::string PrintExpandLayout(int expandAxis) const;

    DynamicParamPackMTE PrepareDynamicShapeInfo(
        int dynShapeIdx, int ShapeDim = SHAPE_DIM4, bool gmOffsetCond = true) const;

    std::string PrintReduceLastAxis() const;
    std::string PrintReduceLastAxisTileTensor() const;
    const std::unordered_map<Opcode, std::function<std::string()>> opsGenMap_ = {
        // UB <-> GM
        {                Opcode::OP_UB_COPY_IN,                 [this]() { return GenUBCopyIn(); }},
        {               Opcode::OP_UB_COPY_OUT,                [this]() { return GenUBCopyOut(); }},

        // L1 <-> GM/BT/L1
        {                Opcode::OP_L1_COPY_IN,              [this]() { return GenMemL1CopyIn(); }},

        // L0C <-> GM
        {              Opcode::OP_L0C_COPY_OUT,            [this]() { return GenMemL0CCopyOut(); }},

        // L1 <-> L0
        {                 Opcode::OP_L1_TO_L0A,                [this]() { return GenMemL1ToL0(); }},
        {                 Opcode::OP_L1_TO_L0B,                [this]() { return GenMemL1ToL0(); }},
        {               Opcode::OP_L1_TO_L0_BT,                [this]() { return GenMemL1ToL0(); }},

        // // cast op
        {                      Opcode::OP_CAST,                   [this]() { return GenCastOp(); }},
        // binary op: vector operations
        {                       Opcode::OP_ADD,                 [this]() { return GenBinaryOp(); }},
        {                       Opcode::OP_SUB,                 [this]() { return GenBinaryOp(); }},
        {                       Opcode::OP_MUL,                 [this]() { return GenBinaryOp(); }},
        {                       Opcode::OP_DIV,                 [this]() { return GenBinaryOp(); }},
        {                   Opcode::OP_MAXIMUM,                 [this]() { return GenBinaryOp(); }},
        {                   Opcode::OP_MINIMUM,                 [this]() { return GenBinaryOp(); }},
        {                   Opcode::OP_PAIRSUM,                 [this]() { return GenBinaryOp(); }},
        {                   Opcode::OP_PAIRMAX,                 [this]() { return GenBinaryOp(); }},
        {                   Opcode::OP_PAIRMIN,                 [this]() { return GenBinaryOp(); }},

        // // binary op: vector scalar
        {                      Opcode::OP_ADDS,           [this]() { return GenVectorScalarOp(); }},
        {                      Opcode::OP_SUBS,           [this]() { return GenVectorScalarOp(); }},
        {                      Opcode::OP_MULS,           [this]() { return GenVectorScalarOp(); }},
        {                      Opcode::OP_DIVS,           [this]() { return GenVectorScalarOp(); }},
        {                      Opcode::OP_MAXS,           [this]() { return GenVectorScalarOp(); }},
        {                      Opcode::OP_MINS,           [this]() { return GenVectorScalarOp(); }},

        // // unary op
        {                       Opcode::OP_EXP,                  [this]() { return GenUnaryOp(); }},
        {                      Opcode::OP_SQRT,                  [this]() { return GenUnaryOp(); }},
        {                    Opcode::OP_EXPAND,                  [this]() { return GenUnaryOp(); }},
        {                Opcode::OP_RECIPROCAL,                  [this]() { return GenUnaryOp(); }},
        {                       Opcode::OP_ABS,                  [this]() { return GenUnaryOp(); }},
        {                       Opcode::OP_ROWMAXLINE,          [this]() { return GenUnaryOp(); }},
          {                     Opcode::OP_ROWMINLINE,          [this]() { return GenUnaryOp(); }},

        // unary with temp buffer
        {             Opcode::OP_ROWSUM_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {             Opcode::OP_ROWMAX_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {             Opcode::OP_ROWMIN_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {       Opcode::OP_TRANSPOSE_VNCHWCONV,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},
        {Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE,       [this]() { return GenUnaryOpWithTmpBuff(); }},

        // // vector dup
        {                   Opcode::OP_VEC_DUP,                    [this]() { return GenDupOp(); }},

        // // matmul
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

        // indexput
        {Opcode::OP_INDEX_PUT, [this]() { return GenIndexPutOp(); }},

        // for performace optimization
        {                    Opcode::OP_PHASE1,              []() { return "SUBKERNEL_PHASE1\n"; }},
        {                    Opcode::OP_PHASE2,              []() { return "SUBKERNEL_PHASE2\n"; }},

        // vector where
        {                    Opcode::OP_WHERE_SS,              [this]() { return GenWhereOp(); }},
        {                    Opcode::OP_WHERE_TS,              [this]() { return GenWhereOp(); }},
        {                    Opcode::OP_WHERE_ST,              [this]() { return GenWhereOp(); }},
        {                    Opcode::OP_WHERE_TT,              [this]() { return GenWhereOp(); }},

        // cmp op
        {               Opcode::OP_CMP,                   [this]() { return GenCmpOp(); }},
        {               Opcode::OP_CMPS,                   [this]() { return GenCmpOp(); }},
    };
};

} // namespace npu::tile_fwk

#endif // CODEGEN_OP_LITENPU_H
