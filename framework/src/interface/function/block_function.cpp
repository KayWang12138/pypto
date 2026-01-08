/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_function.cpp
 * \brief
 */

#include "interface/function/block_function.h"

#include <unordered_map>

#include "function/function.h"
#include "interface/operation/attribute.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/operation/operation_impl.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/tensor/tensor_offset.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
BlockFunction::BlockFunction(const Program &belongTo, const std::string &funcMagicName,
    const std::string &funcRawName, Function *parentFunc)
    : Function(belongTo, funcMagicName, funcRawName, parentFunc),
      programId_(-1),
      leafFuncAttr_(nullptr) {
}

std::vector<OperationPtr> &BlockFunction::GetProgramOp() {
    return operations_;
}

void BlockFunction::SetProgramOp(const std::vector<OperationPtr> &operations) {
    operations_ = operations;

    RefreshOpPosition();
    sorted_ = true;
}

void BlockFunction::UpdateBelongToThis() {
    for (auto &ele : operations_) {
         ele->SetBelongingFunction(this);
    }
}

void BlockFunction::ScheduleBy(const std::vector<Operation *> &newList, bool needRefresh) {
    if (needRefresh) {
        RefreshOpPosition();
    }
    ASSERT(newList.size() == operations_.size());
    std::vector<std::shared_ptr<Operation>> newOperations;
    for (auto op : newList) {
        ASSERT(opPosition_.count(op) > 0);
        newOperations.emplace_back(operations_[opPosition_.at(op)]);   
    }
    operations_ = newOperations;
    RefreshOpPosition();

    sorted_ = true;
}

static const SymbolicScalar RUNTIME_COA_GetOffset = AddRuntimeCoaPrefix("GET_PARAM_OFFSET");
static const SymbolicScalar RUNTIME_COA_GetValidShape = AddRuntimeCoaPrefix("GET_PARAM_VALID_SHAPE");
static const SymbolicScalar RUNTIME_COA_GetParam = AddRuntimeCoaPrefix("GET_PARAM");

static void MaybeNormalizeValue(
        const SymbolicScalar &coaFunc,
        std::vector<SymbolicScalar> &operandCoaList,
        int operandCoaIndex,
        std::vector<OpImmediate> &opImmList,
        int coaIndex,
        bool valueToIndex) {
    for (size_t dimIndex = 0; dimIndex < opImmList.size(); dimIndex++) {
        auto &opImm = opImmList[dimIndex];
        SymbolicScalar scalar = opImm.GetSpecifiedValue();
        auto getTensorDataDict = GetTensorDataDict(scalar);
        if (getTensorDataDict.size() == 0) {
            OpImmediate::NormalizeValue(operandCoaList[operandCoaIndex + dimIndex], opImm, coaFunc(opImmList.size(), coaIndex, dimIndex), valueToIndex);
        }
    }
};

static void MaybeNormalizeValue(
    std::vector<SymbolicScalar> &valueCoa,
    SymbolicScalar &value,
    int coaIndex,
    bool valueToIndex) {
    auto getTensorDataDict = GetTensorDataDict(value);
    if (getTensorDataDict.size() == 0) {
        valueCoa.push_back(value);
        if (valueToIndex) {
            value = RUNTIME_COA_GetParam(coaIndex);
        }
    }
}

static std::vector<SymbolicScalar> NormalizeCopyIn(Operation *op, int coaIndexBase, bool valueToIndex) {
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    int dim = copyAttr->GetShape().size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    auto opImmList = copyAttr->GetFromOffset();
    MaybeNormalizeValue(RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex, opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetFromOffset(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    // shape to normal
    opImmList = copyAttr->GetShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    copyAttr->SetShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetRawShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    copyAttr->SetRawShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetToDynValidShape();
    MaybeNormalizeValue(RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex, opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetToDynValidShape(opImmList);

    return operandCoaList;
}

static std::vector<SymbolicScalar> NormalizeCopyOut(Operation *op, int coaIndexBase, bool valueToIndex) {
    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op->GetOpAttribute());
    int dim = copyAttr->GetShape().size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    auto opImmList = copyAttr->GetToOffset();
    MaybeNormalizeValue(RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex, opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetToOffset(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    // shape to normals
    opImmList = copyAttr->GetShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    copyAttr->SetShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetRawShape();
    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, opImmList, coaIndex, valueToIndex);
    copyAttr->SetRawShape(opImmList);
    operandCoaIndex += dim;
    coaIndex += dim;

    opImmList = copyAttr->GetFromDynValidShape();
    MaybeNormalizeValue(RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex, opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetFromDynValidShape(opImmList);

    return operandCoaList;
}

static std::vector<SymbolicScalar> NormalizeTensor(
    LogicalTensorPtr operand, int coaIndexBase, bool valueToIndex, bool isNop = false) {
    auto offset = OpImmediate::Specified(operand->GetOffset());
    auto dynOffset = OpImmediate::Specified(operand->GetDynOffset());
    auto shape = OpImmediate::Specified(operand->GetShape());
    auto rawshape = OpImmediate::Specified(operand->GetRawTensor()->GetRawShape());
    auto dynRawshape = OpImmediate::Specified(operand->GetRawTensor()->GetDynRawShape());
    auto dynValidShape = OpImmediate::Specified(operand->GetDynValidShape());
    if (isNop) {
        offset = OpImmediate::Specified(Offset(operand->GetShape().size()));
        dynOffset = OpImmediate::Specified(Offset(operand->GetShape().size()));
        shape = OpImmediate::Specified(Shape(operand->GetShape().size()));
        dynValidShape = OpImmediate::Specified(Shape(operand->GetShape().size()));
    }

    int dim = shape.size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    if (!dynOffset.empty()) {
        MaybeNormalizeValue(
            RUNTIME_COA_GetOffset, operandCoaList, operandCoaIndex, dynOffset, coaIndexBase, valueToIndex);
        operand->UpdateOffset(TensorOffset{operand->GetOffset(), OpImmediate::ToSpecified(dynOffset)});
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, offset, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, shape, coaIndex, false);
    operandCoaIndex += dim;
    coaIndex += dim;

    if (!dynRawshape.empty()) {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, dynRawshape, coaIndex, valueToIndex);
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, rawshape, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    if (!dynValidShape.empty()) {
        MaybeNormalizeValue(
            RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex, dynValidShape, coaIndexBase, valueToIndex);
        operand->UpdateDynValidShape(OpImmediate::ToSpecified(dynValidShape));
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, shape, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    return operandCoaList;
}

void BlockFunction::GetOutcastSymbolicExpr(std::map<int, SymbolicScalar>& tabel) {
    for (size_t i = 0; i< outCasts_.size(); i++) {
        auto op = *outCasts_[i]->GetProducers().begin();
        if (op->GetOpcode() == Opcode::OP_BIND_TENSOR) {
            if (op->HasAttr(OpAttributeKey::bindTensor) && (op->GetOOperands().size() == 1UL)) {
                tabel[i] = op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
            }
        }
    }
}

std::vector<std::vector<SymbolicScalar>> BlockFunction::NormalizeCoa(std::vector<int> &iOffset, std::vector<int> &oOffset) {
    std::unordered_map<int, Operation *> opmagicToOp;
    std::unordered_map<LogicalTensorPtr, int> processedOperands;

    opmagicToOp.reserve(operations_.size());
    for (auto &op : operations_) {
        opmagicToOp[op->GetOpMagic()] = op.get();
    }

    int coaIndex = COA_INDEX_BASE;
    std::vector<std::vector<SymbolicScalar>> coaLists;
    coaLists.reserve(incastPosition.size() + outcastPosition.size());
    NormalizeCoaForInCasts(iOffset, coaLists, coaIndex, processedOperands, opmagicToOp);
    NormalizeCoaForOutCasts(oOffset, coaLists, coaIndex, processedOperands, opmagicToOp);
    NormalizeCoaForNormalOperands(coaLists, coaIndex, processedOperands);
    NormalizeCoaForSpecialInfo(coaLists, coaIndex);

    return coaLists;
}

void BlockFunction::NormalizeCoaForInCasts(std::vector<int> &iOffset, std::vector<std::vector<SymbolicScalar>> &coaLists,
    int &coaIndex, std::unordered_map<LogicalTensorPtr, int> &processedOperands,
    const std::unordered_map<int, Operation *> &opmagicToOp) {
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    iOffset.clear();
    iOffset.reserve(incastPosition.size());
    for (auto [opmagic, k] : incastPosition) {
        auto op = opmagicToOp.at(opmagic);
        if (op->GetIOpAttrOffset(k) != -1) {
            continue;
        }
        std::vector<SymbolicScalar> operandCoaList;
        if (IsCopyIn(op->GetOpcode()) && k == 0) {
            operandCoaList = NormalizeCopyIn(op, coaIndex, valueToIndex);
            if (CheckEmuOpcode(op, EMUOP_TENSOR_GETDATA_DEPEND)) {
                GetTensorDataSetCoaIndex(op, coaIndex);
            }
        } else {
            auto iOperand = op->GetInputOperand(k);
            auto it = processedOperands.find(iOperand);
            if (it != processedOperands.end()) {
                op->SetIOpAttrOffset(k, it->second);
                iOffset.emplace_back(it->second);
                continue;
            }
            operandCoaList = NormalizeTensor(iOperand, coaIndex, false, op->GetOpcode() == Opcode::OP_NOP);
            processedOperands.emplace(iOperand, coaIndex);
        }
        op->SetIOpAttrOffset(k, coaIndex);
        iOffset.emplace_back(coaIndex);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }
}

void BlockFunction::NormalizeCoaForOutCasts(std::vector<int> &oOffset, std::vector<std::vector<SymbolicScalar>> &coaLists,
    int &coaIndex, std::unordered_map<LogicalTensorPtr, int> &processedOperands,
    const std::unordered_map<int, Operation *> &opmagicToOp) {
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    oOffset.reserve(outcastPosition.size());
    for (auto [opmagic, k] : outcastPosition) {
        auto op = opmagicToOp.at(opmagic);
        if (op->GetOOpAttrOffset(k) != -1) {
            continue;
        }
        std::vector<SymbolicScalar> operandCoaList;
        if (IsCopyOut(op->GetOpcode()) && k == 0) {
            operandCoaList = NormalizeCopyOut(op, coaIndex, valueToIndex);
        } else {
            auto oOperand = op->GetOutputOperand(k);
            auto it = processedOperands.find(oOperand);
            if (it != processedOperands.end()) {
                op->SetOOpAttrOffset(k, it->second);
                oOffset.emplace_back(it->second);
                continue;
            }
            operandCoaList = NormalizeTensor(oOperand, coaIndex, false);
            processedOperands.emplace(oOperand, coaIndex);
        }
        op->SetOOpAttrOffset(k, coaIndex);
        oOffset.emplace_back(coaIndex);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }
}

void BlockFunction::NormalizeCoaForNormalOperands(std::vector<std::vector<SymbolicScalar>> &coaLists, int &coaIndex,
    std::unordered_map<LogicalTensorPtr, int> &processedOperands) {
    std::unordered_set<LogicalTensorPtr> inOutCasts;
    inOutCasts.insert(inCasts_.begin(), inCasts_.end());
    inOutCasts.insert(outCasts_.begin(), outCasts_.end());
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    for (auto &op : operations_) {
        if (op->GetOpcode() == Opcode::OP_NOP) {
            continue;
        }
        for (size_t i = 0; i < op->GetInputOperandSize(); i++) {
            auto iOperand = op->GetInputOperand(i);
            if ((op->GetIOpAttrOffset(i) != -1) || (inOutCasts.count(iOperand) > 0)) {
                continue;
            }
            auto it = processedOperands.find(iOperand);
            if (it != processedOperands.end()) {
                op->SetIOpAttrOffset(i, it->second);
                continue;
            }
            if (!iOperand->GetDynOffset().empty() || !iOperand->GetDynValidShape().empty()) {
                auto operandCoaList = NormalizeTensor(iOperand, coaIndex, valueToIndex);
                processedOperands.emplace(iOperand, coaIndex);
                coaIndex += operandCoaList.size();
                coaLists.emplace_back(std::move(operandCoaList));
            }
        }
        for (size_t i = 0; i < op->GetOutputOperandSize(); i++) {
            auto oOperand = op->GetOutputOperand(i);
            if ((op->GetOOpAttrOffset(i) != -1) || (inOutCasts.count(oOperand) > 0)) {
                continue;
            }
            if (oOperand->GetConsumers().empty()) {
                continue;
            }
            auto it = processedOperands.find(oOperand);
            if (it != processedOperands.end()) {
                op->SetOOpAttrOffset(i, it->second);
                continue;
            }
            if (!oOperand->GetDynOffset().empty() || !oOperand->GetDynValidShape().empty()) {
                auto operandCoaList = NormalizeTensor(oOperand, coaIndex, valueToIndex);
                processedOperands.emplace(oOperand, coaIndex);
                op->SetOOpAttrOffset(i, coaIndex);
                coaIndex += operandCoaList.size();
                coaLists.emplace_back(std::move(operandCoaList));
            }
        }
    }
}

void BlockFunction::NormalizeCoaForSpecialInfo(std::vector<std::vector<SymbolicScalar>> &coaLists, int &coaIndex) {
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    for (auto &op : operations_) {
        if (op->GetOpcode() == Opcode::OP_VEC_DUP || op->GetOpcode() == Opcode::OP_RANGE) {
            if (op->HasAttr(OpAttributeKey::dynScalar)) {
                SymbolicScalar dynScalar = op->GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
                std::vector<SymbolicScalar> valueCoaList;
                MaybeNormalizeValue(valueCoaList, dynScalar, coaIndex, valueToIndex);
                op->SetAttribute(OpAttributeKey::dynScalar, dynScalar);
                coaLists.emplace_back(valueCoaList);
                coaIndex += 1;
            }
        } else if (op->GetOpcode() == Opcode::OP_BIND_TENSOR) {
            if (op->HasAttr(OpAttributeKey::bindTensor) && (op->GetOOperands().size() == 1UL)) {
                SymbolicScalar bindTensor = op->GetSymbolicScalarAttribute(OpAttributeKey::bindTensor);
                std::vector<SymbolicScalar> valueCoaList;
                MaybeNormalizeValue(valueCoaList, bindTensor, coaIndex, valueToIndex);
                coaLists.emplace_back(valueCoaList);
                coaIndex += 1;
            }
        }
    }
}

void BlockFunction::CreateLeafInAndOutCast(const LogicalTensorPtr &inOrOut,
    LogicalTensors &inOrOutList) const {
inOrOutList.emplace_back(inOrOut->Clone(*parent_));
}

GetTensorDataIODescDict BlockFunction::GetTensorDataForLeafGraph() {
    GetTensorDataIODescDict iodescDict;
    for (auto &op : Operations(false)) {
        if (!CheckEmuOpcode(&op, EMUOP_TENSOR_GETDATA_IMPORT)) {
            continue;
        }
        int getTensorDataIndex = GetTensorDataGetIndex(&op);
        ASSERT(getTensorDataIndex != -1)
            << "Failed to get tensor data index for operation";
        auto tensor = op.GetIOperands()[0];
        auto incastIndex = GetIncastIndex(tensor);
        if (incastIndex != INVALID_IOINDEX) {
            iodescDict[getTensorDataIndex] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIndex, 0);
        }
    }
    return iodescDict;
}

} // namespace npu::tile_fwk