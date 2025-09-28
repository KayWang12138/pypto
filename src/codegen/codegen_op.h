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

#ifndef CODEGEN_OP_H
#define CODEGEN_OP_H

#include <map>
#include <tuple>
#include <cstdint>
#include <string>
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

namespace npu::tile_fwk {
const std::unordered_set<Opcode> SKIP_OPCODE = {
    Opcode::OP_VIEW,
    Opcode::OP_ASSEMBLE,
    Opcode::OP_RESHAPE,
    Opcode::OP_UB_ALLOC,
    Opcode::OP_L1_ALLOC,
    Opcode::OP_L0A_ALLOC,
    Opcode::OP_L0B_ALLOC,
    Opcode::OP_L0C_ALLOC,
    Opcode::OP_FIX_ALLOC,
    Opcode::OP_BT_ALLOC,
    Opcode::OP_BIND_TENSOR,
};

const int MAX_OPERANDS = 11;
const int NULL_OPERAND = 0;

struct OpInfo {
    std::string op;
    std::vector<int> operands;       // d, s0, s1
    std::vector<DataType> dataTypes; // dstDType, src0DType, src1DType
    int bufferId{0};
    OpInfo(std::string opArg, const std::vector<int> &operandsArg, const std::vector<DataType> &dataTypesArg)
        : op(std::move(opArg)), operands(operandsArg), dataTypes(dataTypesArg) {}
};

class CodeGenOp {
public:
    explicit CodeGenOp(SymbolManager &symbolManager, FunctionType funcType, const std::map<int, int> &locToOffset = {},
        bool isUnderDynamicFunc = false)
        : functionType(funcType), paramLocToParamListOffset(locToOffset), isUnderDynamicFunction(isUnderDynamicFunc) {
        for (size_t i = 0; i < MAX_OPERANDS; i++) {
            operand[i] = NULL_OPERAND;
            operandType[i] = BUF_UNKNOWN;
        }
        sm = &symbolManager;
    }
    virtual ~CodeGenOp() = default;

    virtual bool Init(const Operation &ops);

    virtual std::string GenBarrier() const;
    virtual std::string GenSyncSetOp() const;
    virtual std::string GenSyncWaitOp() const;

    virtual std::string GenOpCode() const = 0;

protected:
    std::string GenOpAttr() const;

    // NEXTNEXT: list of all primitives:
    // [ NOP, UB_ALLOC, L1_ALLOC, L0A_ALLOC, L0B_ALLOC, L0C_ALLOC,
    //   UB_ADD, UB_MUL, UB_COPY_IN, UB_COPY_OUT, L1_COPY_IN, UB_TO_L1,
    //   UB_PAIRMAX, UB_PAIRSUM, UB_ROWEXPMAX, UB_ROWEXPSUM, UB_SUB, UB_DIV,
    //   UB_EXP SYNC_SRC, SYNC_DST ]
    std::string opCodeStr;
    Opcode opCode;
    std::string aliasOp; // alias op name

    int operand[MAX_OPERANDS] = {}; // buffer id
    int operandWithMagic[MAX_OPERANDS] = {};
    OperandType operandType[MAX_OPERANDS] = {BUF_UNKNOWN, BUF_UNKNOWN, BUF_UNKNOWN, BUF_UNKNOWN};
    DataType operandDtype[MAX_OPERANDS] = {
        DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM, DataType::DT_BOTTOM};
    Element extOperandVal;
    std::vector<int64_t> offset[MAX_OPERANDS] = {};
    std::vector<int64_t> shape[MAX_OPERANDS] = {};
    std::vector<int64_t> rawShape[MAX_OPERANDS] = {};
    std::vector<OpImmediate> dynShapeFromAttr[MAX_OPERANDS] = {}; // used for gm spilling scene
    // need adapt unaligned scene
    // Used for unaligned scene. In AST 1.0 it was padded in LogicalTensor constructor
    std::vector<int64_t> originShape[MAX_OPERANDS] = {};
    std::vector<SymbolicScalar> dynamicValidShape[MAX_OPERANDS] = {}; // valid shape
    std::vector<SymbolicScalar> offsetGmSymbolic[MAX_OPERANDS] = {};  // for spilling into GM scene
    // if operand is an variable, record its related argument location
    // In COA(Call Operation Attribute), 0-index is the callee's cce info. So the tensor list starts from 1.
    int paramLocation[MAX_OPERANDS] = {1, 1, 1, 1, 1, 1};
    int GmTensorParamIdxInCallFunc{0};
    OpSyncQueue syncQueue;

    // add for ooo sched
    int addrOffset[MAX_OPERANDS] = {};
    std::vector<long> convParams;
    std::vector<int> poolParams;
    RawTensor::ShmemInfo shmemInfo[MAX_OPERANDS] = {};

    std::map<std::string, npu::tile_fwk::Any> opAttrs;

    SymbolManager *sm{nullptr};

    const FunctionType functionType;
    std::string tileOpName;
    bool isInputForceCombineAxis{false};
    bool isSupportDynamicUnaligned{false};
    bool isSupportLayout{false};
    const std::map<int, int> &paramLocToParamListOffset{};
    bool isUnderDynamicFunction;
    int operandCnt{0};

private:
    void UpdateCodegenOpInfoByTensor(
        const Operation &ops, bool isInput, const std::shared_ptr<LogicalTensor> &tensor, int &operandIdx);

    void UpdateTileOpInfo(const Operation &ops);

    void GetGmParamIdx(const Operation &oper);

    void ConvertPoolAttribute(const Operation &operation);
    void ConvertAttribute(const Operation &operation);

    void UpdateShape(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx);
    void UpdateOffsetForInput(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx);
    void UpdateOffsetForOutput(const Operation &oper, const LogicalTensor &logicalTensor, int operandIdx);
    void UpdateOffsetValueForGM(const std::vector<OpImmediate> &offsets, int operandIdx);
    void UpdateOpAttribute(const npu::tile_fwk::Operation &ops);
};
} // namespace npu::tile_fwk

#endif // CODEGEN_OP_H
