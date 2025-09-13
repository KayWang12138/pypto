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
 * \file infer_param_index.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include "infer_param_index.h"
#include "interface/operation/op_infer_shape_impl.h"
#include "interface/operation/opcode.h"

namespace npu {
namespace tile_fwk {
std::string InferParamIndex::DumpParamIndex(const std::map<std::string, DynParamInfo>& dynParamTable)
{
    std::ostringstream ss;
    for (auto paramInfo : dynParamTable) {
        ss << "param: " << paramInfo.first << " ( ";
        ss << "tensorIdx: " << paramInfo.second.tensorIndex << ", ";
        ss << "dimsize: " << paramInfo.second.dimSize << ", ";
        ss << "type: " << static_cast<int>(paramInfo.second.type) << ", ";
        ss << "addrCoaIdx: " << paramInfo.second.tensorBaseAddrCoaIndex << ", ";
        ss << "dimIdx: " << paramInfo.second.dimIndex << " )" << std::endl;
    }
    return ss.str();
}

Status InferParamIndex::ResetOutputDynValidShape(const Operation &op) {
    std::vector<SymbolicScalar> validShape;
    const std::set<Opcode> specifiedOps = {Opcode::OP_VEC_DUP, Opcode::OP_EXPAND, Opcode::OP_RESHAPE};
    for (auto outOperand : op.GetOOperands()) {
        if (OpcodeManager::Inst().IsCopyInOrOut(op.GetOpcode()) || specifiedOps.count(op.GetOpcode())) {
            for (size_t dimIdx = 0U; dimIdx < outOperand->GetShape().size(); ++dimIdx) {
                validShape.push_back(SymbolicScalar("sym_" +  std::to_string(outOperand->GetMagic()) +
                                                    "_dim_" + std::to_string(dimIdx)));
            }
        }
        outOperand->UpdateDynValidShape(validShape);
    }
    return SUCCESS;
}

Status InferParamIndex::ResetViewDynValidShape(const Operation &op) {
    auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
    if (viewOpAttribute == nullptr) {
        return SUCCESS;
    }
    auto newDynValidShape = viewOpAttribute->GetToDynValidShape();
    std::vector<int> newValidShape;
    for (auto validSym : newDynValidShape) {
        if (validSym.ConcreteValid()) {
            newValidShape.push_back(validSym.Concrete());
        }
    }
    if (newValidShape.size() == newDynValidShape.size()) {
        op.GetOOperands()[0]->UpdateDynValidShape(newDynValidShape);
        return SUCCESS;
    }
    viewOpAttribute->SetToDynValidShape(std::vector<SymbolicScalar>());
    return SUCCESS;
}

Status InferParamIndex::ResetAssembleDynValidShape(const Operation &op) {
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get());
    if (assembleOpAttribute != nullptr) {
        auto emptyValidShape = std::vector<SymbolicScalar>();
        assembleOpAttribute->SetFromDynValidShape(emptyValidShape);
    }
    return SUCCESS;
}

Status InferParamIndex::ResetDynValidShape(Function& function) {
    for (auto &op : function.Operations()) {
        if (ResetOutputDynValidShape(op) != SUCCESS) {
            ALOG_ERROR_F("ResetOutputDynValidShape failed.");
            return FAILED;
        }
        // 清空view和assemble的属性中的dynvalidshape，以便后续重新推导符号化的dynvalidshape
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            if (ResetViewDynValidShape(op) != SUCCESS) {
                ALOG_ERROR_F("ResetViewDynValidShape failed.");
                return FAILED;
            }
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            if (ResetAssembleDynValidShape(op) != SUCCESS) {
                ALOG_ERROR_F("ResetAssembleDynValidShape failed.");
                return FAILED;
            }
        }
    }
    return SUCCESS;
}

Status InferParamIndex::InferShape(Function &function)
{
    size_t i = 0U;
    std::map<int, size_t> opMagic2Idx;
    std::vector<Operation*> opList = function.Operations().DuplicatedOpList();
    if (opList.empty()) {
        ALOG_ERROR_F("InferShape: opList is Empty.");
        return FAILED;
    }
    for (auto op : opList) {
        opMagic2Idx[op->GetOpMagic()] = i;
        i++;
    }
    std::vector<std::vector<size_t>> opInGraph(opList.size());
    std::vector<std::vector<size_t>> opOutGraph(opList.size());
    for (auto op : opList) {
        for (auto producer : op->ProducerOps()) {
            opInGraph[opMagic2Idx[op->GetOpMagic()]].push_back(opMagic2Idx[producer->GetOpMagic()]);
            opOutGraph[opMagic2Idx[producer->GetOpMagic()]].push_back(opMagic2Idx[op->GetOpMagic()]);
        }
    }
    bool isParamIndex = true;
    TopoProgramUtils::TopoProgram(opList, opInGraph, opOutGraph, isParamIndex);
    return SUCCESS;
}

Status InferParamIndex::UpdateValidShape(Function &subFunc, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShape, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShapeSpecified) {
    for (auto &op : subFunc.Operations()) {
        int tensorBaseAddrCoaIndex = IsCopyIn(op.GetOpcode()) ? op.GetIOpAttrOffset(0) : op.GetOOpAttrOffset(0);
        if (tensorBaseAddrCoaIndex == -1) {
            continue;
        }
        if (addr2ValidShape.find(tensorBaseAddrCoaIndex) == addr2ValidShape.end()) {
            addr2ValidShape[tensorBaseAddrCoaIndex] = op.GetOOperands()[0]->GetDynValidShape();
            if (IsCopyIn(op.GetOpcode())) {
                auto attr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
                if (attr->GetToDynValidShape().size() != 0 && attr->GetToDynValidShape()[0].IsSpecified()) {
                    addr2ValidShapeSpecified[tensorBaseAddrCoaIndex] = OpImmediate::ToSpecified(attr->GetToDynValidShape());
                }
            }
        }
    }
    return SUCCESS;
}

Status InferParamIndex::SetSubValidShape(Function &subFunc, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShape, std::map<int, std::vector<SymbolicScalar>> &addr2ValidShapeSpecified) {
    std::set<std::string> visitedSymbol;
    int tensorIndex{0};
    for (auto validShape : addr2ValidShape) {
        int dimIdx{0};
        for (auto dim : validShape.second) {
            if (!dim.IsSymbol()) {
                continue;
            }
            if (visitedSymbol.count(dim.Dump()) > 0) {
                continue;
            }
            auto tensorBaseAddrCoaIndex = validShape.first;
            SymbolicScalar dynDim;
            if (addr2ValidShapeSpecified.count(tensorBaseAddrCoaIndex)) {
                dynDim = addr2ValidShapeSpecified[tensorBaseAddrCoaIndex][dimIdx];
            }
            auto paramInfo = DynParamInfo{
                static_cast<int>(validShape.second.size()), tensorIndex, tensorBaseAddrCoaIndex, DynParamInfoType::VALID_SHAPE,
                dimIdx, dynDim};
            subFunc.InsertDynParam(dim.Dump(), paramInfo);
            dimIdx++;
        }
        tensorIndex++;
    }
    return SUCCESS;
}

Status InferParamIndex::UpdateParamIndex(Function &function) {
    for (auto &subProgram : function.rootFunc_->programs_) {
        auto &subFunc = *subProgram.second;
        if (ResetDynValidShape(subFunc) != SUCCESS) {
            return FAILED;
        }
        if (InferShape(subFunc) != SUCCESS) {
            return FAILED;
        }
        ALOG_INFO(subFunc.Dump());
        std::map<int, std::vector<SymbolicScalar>> addr2ValidShape;
        std::map<int, std::vector<SymbolicScalar>> addr2ValidShapeSpecified;
        if (UpdateValidShape(subFunc, addr2ValidShape, addr2ValidShapeSpecified) != SUCCESS) {
            ALOG_ERROR_F("UpdateValidShape failed.");
            return FAILED;
        }
        if (SetSubValidShape(subFunc, addr2ValidShape, addr2ValidShapeSpecified) != SUCCESS) {
            ALOG_ERROR_F("SetSubValidShape failed.");
            return FAILED;
        }
        ALOG_DEBUG(DumpParamIndex(subFunc.GetDynParamTable()));
    }
    return SUCCESS;
}

Status InferParamIndex::RunOnFunction(Function &function)
{
    ALOG_INFO_F("===> Start InferParamIndex.");
    if (UpdateParamIndex(function) != SUCCESS) {
        ALOG_ERROR_F("UpdateParamIndex failed.");
        return FAILED;
    }
    ALOG_INFO_F("===> End InferParamIndex By Sequential Execution.");
    return SUCCESS;
}
}  // namespace tile_fwk
}  // namespace npu