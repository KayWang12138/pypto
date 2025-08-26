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
 * \file function.cpp
 * \brief
 */

#include "interface/function/function.h"
#include <queue>
#include <algorithm>
#include <unordered_map>
#include "common/pre_def.h"
#include "interface/cache/hash.h"
#include "interface/operation/opcode.h"
#include "interface/operation/operation.h"
#include "interface/tensor/tensor_offset.h"
#include "interface/utils/id_gen.h"
#include "interface/utils/log.h"
#include "tilefwk/symbolic_scalar.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/logical_tensor.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/configs/config_manager.h"
#include "interface/operation/operation_impl.h"
#include "interface/utils/serialization.h"
#include "interface/interpreter/flow_verifier.h"

using namespace npu::tile_fwk;

namespace {
const std::string PREFIX = "  ";
const int SPACE_NUM_THREE = 3;
const int LAST_TWO = -2;

const std::set<Opcode> SPECIAL_OPCODE_SET = {
    Opcode::OP_INDEX_OUTCAST, Opcode::OP_VIEW, Opcode::OP_ASSEMBLE, Opcode::OP_CALL, Opcode::OP_CONVERT,
    Opcode::OP_COPY_IN, Opcode::OP_COPY_OUT
};
struct ViewKey {
    ViewKey(const int magic, const std::vector<int> &newShape, const std::vector<int> &newOffset,
        const std::vector<SymbolicScalar> &tmpDynOffset)
        : rawMagic(magic), shape(newShape), offset(newOffset), dynOffset(tmpDynOffset) {}

    bool operator<(const ViewKey &x) const {
        if (shape != x.shape) {
            return shape < x.shape;
        } else if (offset != x.offset) {
            return offset < x.offset;
        }
        if (dynOffset.size() != x.dynOffset.size()) {
            return dynOffset.size() < x.dynOffset.size();
        }
        for (size_t i = 0; i < dynOffset.size(); i++) {
            if (dynOffset[i].Raw() != x.dynOffset[i].Raw()) {
                return dynOffset[i].Raw() < x.dynOffset[i].Raw();
            }
        }

        return rawMagic < x.rawMagic;
    }

    int rawMagic;
    std::vector<int> shape;
    std::vector<int> offset;
    std::vector<SymbolicScalar> dynOffset;
};
} // namespace

std::vector<Operation *> OperationsViewer::DuplicatedOpList() const {
    std::vector<Operation *> opList;
    opList.reserve(operations_.size());
    for (auto op : operations_) {
        opList.emplace_back(op.get());
    }
    return opList;
}

struct CompareTensorPtr {
    bool operator()(const std::shared_ptr<LogicalTensor> &a, const std::shared_ptr<LogicalTensor> &b) const {
        return a->offset < b->offset;
    }
};

std::string DynloopFunctionPathNode::Dump() const
{
    int indent = 2;
    std::ostringstream oss;
    std::function<void(const DynloopFunctionPathNode *, int)> dump = [&oss, &indent, &dump](const DynloopFunctionPathNode *node, int level) {
        if (!node->cond.IsValid()) {
            oss << std::setw(level * indent) << ' ' << node->root->GetRawName() << "(" << node->root->GetFunctionHash() << ")\n";
        } else {
            oss << std::setw(level * indent) << ' ' << node->cond.Dump() << "\n";
            if (node->branchNodeList[0] != nullptr) {
                dump(node->branchNodeList[0].get(), level + 1);
            }
            if (node->branchNodeList[1] != nullptr) {
                dump(node->branchNodeList[1].get(), level + 1);
            }
        }
    };
    dump(this, 0);
    return oss.str();
}

std::shared_ptr<DynloopFunctionPathNode> DynloopFunctionAttribute::BuildPathNode() {
    std::shared_ptr<DynloopFunctionPathNode> root = std::make_shared<DynloopFunctionPathNode>();
    if (pathList.size() == 1) {
        // No branch
        ASSERT(pathList[0].pathCondList.size() == 0);
        root->root = pathList[0].root;
    } else {
        for (size_t i = 0; i < pathList.size(); i++) {
            auto node = root;
            for (size_t j = 0; j < pathList[i].pathCondList.size(); j++) {
                auto &pathCond = pathList[i].pathCondList[j];
                if (!node->cond.IsValid()) {
                    node->cond = pathCond.GetCond();
                }
                if (node->branchNodeList[pathCond.IsSat()] == nullptr) {
                    node->branchNodeList[pathCond.IsSat()] = std::make_shared<DynloopFunctionPathNode>();
                }
                node = node->branchNodeList[pathCond.IsSat()];
            }
            node->root = pathList[i].root;
        }
    }
    return root;
}

std::string DynloopFunctionAttribute::DumpBranch() const {
    std::ostringstream oss;
    for (size_t i = 0; i < pathList.size(); i++) {
        auto &path = pathList[i];
        oss << "Branch-" << i << ": " << "\n";
        for (size_t j = 0; j < path.pathCondList.size(); j++) {
            auto &cond = path.pathCondList[j];
            oss << "  " << cond.GetFile() << ":" << cond.GetLine() << "] " << cond.GetCond().Dump() << ":" << cond.IsSat() << "\n";
        }
    }
    oss << "current:" << currIndex << "\n";
    for (size_t i = 0; i < currPathCond.size(); i++) {
        oss << "  " << currPathCond[i].GetFile() << ":" << currPathCond[i].GetLine() << "] " << currPathCond[i].GetCond().Dump() << ":" << currPathCond[i].IsSat() << "\n";
    }
    return oss.str();
}

std::vector<DynloopFunctionPathCondition> DynloopFunctionAttribute::GenCondWithBeginEnd(const std::vector<DynloopFunctionPathCondition> &conds) const {
    std::vector<DynloopFunctionPathCondition> resultPathCond = conds;
    for (auto &cond : resultPathCond) {
        if (!cond.cond_.IsExpression()) {
            continue;
        }
        auto expr = std::static_pointer_cast<RawSymbolicExpression>(cond.cond_.Raw());
        if (expr->IsLoopEndCall()) {
            std::vector<RawSymbolicScalarPtr> operandList{expr->OperandList()[0], expr->OperandList()[1], RawSymbolicExpression::CreateBopSub(expr->OperandList()[2], loopRange.Step().Raw())};
            auto newExpr = std::make_shared<RawSymbolicExpression>(SymbolicOpcode::T_MOP_CALL, operandList);
            cond.cond_ = SymbolicScalar(newExpr);
        }
    }
    return resultPathCond;
}

bool DynloopFunctionAttribute::IterationEnd(int unroll, Function *pathFunc, Operation *operation) {
    auto resultPathCond = GenCondWithBeginEnd(currPathCond);
    unrollTimes = unroll;
    pathList.emplace_back(pathFunc, resultPathCond, operation);

    bool finished = true;
    for (size_t i = 0; i < currPathCond.size(); i++) {
        if (!currPathCond[i].IsSat()) {
            const auto &cond = currPathCond[i].cond_;
            if (IsLoopBeginOrEndExpr(cond)) {
                if (!cond.IsLoopBegin() && !cond.IsLoopEnd()) {
                    continue;
                }
                if (std::static_pointer_cast<RawSymbolicExpression>(cond.Raw())->IsLoopBeginCall() && !cond.IsLoopBegin()) {
                    continue;
                }
                if (std::static_pointer_cast<RawSymbolicExpression>(cond.Raw())->IsLoopEndCall() && !cond.IsLoopEnd()) {
                    continue;
                }
            }
            finished = false;
            break;
        }
    }
    return finished;
}

bool DynloopFunctionAttribute::AppendCond(const SymbolicScalar &cond, const std::string &file, int line) {
    bool result = false;
    if (currIndex < currPathCond.size()) {
        ASSERT(cond.Dump() == currPathCond[currIndex].GetCond().Dump());
        ASSERT(file == currPathCond[currIndex].GetFile());
        ASSERT(line == currPathCond[currIndex].GetLine());
        result = currPathCond[currIndex].IsSat();
    } else {
        currPathCond.emplace_back(result, cond, file, line);
    }
    currIndex++;
    return result;
}

void DynloopFunctionAttribute::CreateCurrCond() {
    if (pathList.size() == 0) {
        currPathCond.clear();
        currIndex = 0;
        return;
    }
    bool found = false;
    for (size_t i = currPathCond.size() - 1; i != static_cast<size_t>(-1); i--) {
        if (!currPathCond[i].IsSat()) {
            const auto &cond = currPathCond[i].cond_;
            if (IsLoopBeginOrEndExpr(cond)) {
                if (!cond.IsLoopBegin() && !cond.IsLoopEnd()) {
                    continue;
                }
                if (std::static_pointer_cast<RawSymbolicExpression>(cond.Raw())->IsLoopBeginCall() && !cond.IsLoopBegin()) {
                    continue;
                }
                if (std::static_pointer_cast<RawSymbolicExpression>(cond.Raw())->IsLoopEndCall() && !cond.IsLoopEnd()) {
                    continue;
                }
            }
            currPathCond[i].IsSat() = true;
            currPathCond.erase(currPathCond.begin() + i + 1, currPathCond.end());
            found = true;
            break;
        }
    }
    ASSERT(found);
    currIndex = 0;
}

Function::Function(const Program &belongTo, const std::string &funcMagicName,
    const std::string &funcRawName, Function *parentFunc)
    : funcMagicName_(funcMagicName), funcRawName_(funcRawName), tensorMap_(*this), belongTo_(belongTo) {
    parent_ = parentFunc;
    functionMagic_ = IdGen<IdType::FUNCTION>::Inst().NewId();

    magicSeed_ = 0;
    opSeed_ = FUNCTION_MAX_INCASTS;
}

OperationsViewer Function::Operations(bool sorted) {
    if (!sorted_ && sorted) {
        sorted_ = true;
        SortOperations();
    }
    return OperationsViewer(operations_, opPosition_);
}

OperationsViewer Function::OperationsAfterOOO()
{
    return OperationsViewer(operationsAfterOOO_, opPositionAfterOOO_);
}

void Function::RecordOOOSeq()
{
    operationsAfterOOO_ = operations_;
    opPositionAfterOOO_ = opPosition_;
}

std::vector<OperationPtr> &Function::GetProgramOp() {
    ASSERT(graphType_ == GraphType::LEAF_GRAPH);
    return operations_;
}

void Function::SetProgramOp(const std::vector<OperationPtr> &operations) {
    ASSERT(graphType_ == GraphType::LEAF_GRAPH);
    operations_ = operations;

    RefreshOpPosition();
    sorted_ = true;
}

void Function::UpdateBelongToThis() {
    ASSERT(graphType_ == GraphType::LEAF_GRAPH);
    for (auto &ele : operations_) {
        ele->function_ = this;
    }
}

const SubfuncInvokeInfoTy &Function::GetSubFuncInvokeInfo(const size_t i) const {
    auto callAttr = dynamic_cast<CallOpAttribute *>(operations_[i]->GetOpAttribute().get());
    ASSERT(callAttr != nullptr);
    return *(callAttr->invokeInfo_);
}

size_t Function::GetParamIndex(const RawTensor& rawTensor) {
    if (rawTensor.GetTensorInfo().subscript == -1) {
        return INVALID_IN_OUT_INDEX;
    }
    return rawTensor.GetTensorInfo().subscript;
}

bool Function::HasCallOperation() {
    for (const auto &op : Operations()) {
        if (op.GetOpcode() == Opcode::OP_CALL) {
            return true;
        }
    }
    return false;
}

void Function::CreateLeafInAndOutCast(const LogicalTensorPtr &inOrOut, LogicalTensors &inOrOutList) const {
    inOrOutList.emplace_back(inOrOut->Clone(*parent_));
}

std::unordered_map<int, GetTensorDataIODesc> Function::GetTensorDataForTensorGraph() {
    std::unordered_map<int, GetTensorDataIODesc> iodescDict;
    for (auto &op : Operations(false)) {
        if (!op.HasAttr(OP_EMUOP_PREFIX + "GetTensorData_tensor_to_scalar")) {
            continue;
        }
        int getTensorDataIndex = *op.GetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_tensor_to_scalar");
        auto tensor = op.GetIOperands()[0];
        for (auto cons : tensor->GetConsumers()) {
            if (cons != &op) {
                auto outcast = cons->GetOOperands()[0];
                auto outcastIndex = GetOutcastIndex(outcast);
                if (outcastIndex != INVALID_IOINDEX) {
                    iodescDict[getTensorDataIndex] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_OUTCAST, outcastIndex, 0);
                }
            }
        }
    }
    return iodescDict;
}

std::unordered_map<int, GetTensorDataIODesc> Function::GetTensorDataForLeafGraph() {
    std::unordered_map<int, GetTensorDataIODesc> iodescDict;
    for (auto &op : Operations(false)) {
        if (!op.HasAttr(OP_EMUOP_PREFIX + "GetTensorData_tensor_to_scalar")) {
            continue;
        }
        int getTensorDataIndex = *op.GetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_tensor_to_scalar");
        auto tensor = op.GetIOperands()[0];
        auto incastIndex = GetIncastIndex(tensor);
        if (incastIndex != INVALID_IOINDEX) {
            iodescDict[getTensorDataIndex] = GetTensorDataIODesc(GET_TENSOR_DATA_OPERAND_IOTYPE_INCAST, incastIndex, 0);
        }
    }
    return iodescDict;
}

void Function::GetTensorDataRefreshIO(std::unordered_map<int, GetTensorDataIODesc> &iodescDict) {
    for (auto &op : Operations(false)) {
        switch (op.GetOpcode()) {
            case Opcode::OP_VIEW:
                {
                    auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
                    if (viewAttr != nullptr) {
                        std::vector<SymbolicScalar> &viewFromDynOffset = viewAttr->GetFromDynOffset();
                        std::for_each(viewFromDynOffset.begin(), viewFromDynOffset.end(),
                            [&](SymbolicScalar &offset) { offset = GetTensorDataFillIO(iodescDict, offset); });
                    }
                } break;
            case Opcode::OP_ASSEMBLE:
                {
                    auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
                    if (assembleAttr != nullptr) {
                        std::vector<SymbolicScalar> &assembleToDynOffset = assembleAttr->GetToDynOffset();
                        std::for_each(assembleToDynOffset.begin(), assembleToDynOffset.end(), [&](SymbolicScalar &offset) {
                            offset = GetTensorDataFillIO(iodescDict, offset);
                        });
                    }
                } break;
            case Opcode::OP_COPY_IN: [[fallthrough]];
            case Opcode::OP_UB_COPY_IN:
                {
                    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
                    std::vector<OpImmediate> copyFromOffset = copyAttr->GetFromOffset();
                    if (copyFromOffset[0].IsSpecified()) {
                        std::for_each(copyFromOffset.begin(), copyFromOffset.end(), [&](OpImmediate &opimm) {
                            opimm = OpImmediate::Specified(GetTensorDataFillIO(iodescDict, opimm.GetSpecifiedValue()));
                        });
                        copyAttr->SetFromOffset(copyFromOffset);
                    }
                } break;
            case Opcode::OP_COPY_OUT: [[fallthrough]];
            case Opcode::OP_UB_COPY_OUT:
                {
                    auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
                    std::vector<OpImmediate> copyToOffset = copyAttr->GetToOffset();
                    if (copyToOffset[0].IsSpecified()) {
                        std::for_each(copyToOffset.begin(), copyToOffset.end(), [&](OpImmediate &opimm) {
                            opimm = OpImmediate::Specified(GetTensorDataFillIO(iodescDict, opimm.GetSpecifiedValue()));
                        });
                    }
                    copyAttr->SetToOffset(copyToOffset);
                } break;
            default:
                break;
        }
    }
}

void Function::BeginFunction(const std::vector<std::reference_wrapper<Tensor>> &explicitOpArgs) {
    if (!IsGraphType(GraphType::TENSOR_GRAPH) && !IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        return;
    }
    std::unordered_set<LogicalTensorPtr> used;
    if (explicitOpArgs.empty()) {
        for (const auto &tensor : BelongTo().GetAliveTensors()) {
            auto logicalTensor = tensor->GetStorage(false);
            functionParamInfos_.emplace_back(
                FunctionParamInfo{.key = tensor, .beginValue = logicalTensor, .endValue = logicalTensor});
        }
        return;
    }
    ASSERT(GetFunctionType() != FunctionType::DYNAMIC); // 只有静态function才能走进这个逻辑
    SetExplicit();
    for (const Tensor &tensor : explicitOpArgs) {
        auto logicalTensor = tensor.GetStorage(false);
        ASSERT(logicalTensor != nullptr);
        ASSERT(used.count(logicalTensor) == 0);
        used.emplace(logicalTensor);
        functionParamInfos_.emplace_back(
            FunctionParamInfo{.key = &tensor, .beginValue = logicalTensor, .endValue = logicalTensor});
    }
}

FunctionCallArgs Function::EndFunction(const std::shared_ptr<TensorSlotScope> &scope) {
    // Deduce Incast and Outcast here, need by TENSOR_GRAPH & STATIC_TILE_GRAPH
    if (IsGraphType(GraphType::TENSOR_GRAPH) || IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        OrderedSet<LogicalTensorPtr> incasts;
        OrderedSet<LogicalTensorPtr> outcasts;
        // update functionParamInfo endValue
        for (auto &functionParamInfo : functionParamInfos_) {
            functionParamInfo.endValue = functionParamInfo.key->GetStorage(false);
            if (functionParamInfo.beginValue != nullptr) {
                ASSERT(functionParamInfo.endValue != nullptr);
                functionParamInfo.endValue->GetRawTensor()->SetRawDataPtr(functionParamInfo.beginValue->GetRawTensor()->GetRawDataPtr());
            }
        }
        for (auto &op : Operations()) {
            for (auto &iOperand : op.iOperand) {
                if (tensorMap_.tensorMap_.count(iOperand->tensor->rawmagic) == 0) {
                    incasts.Insert(iOperand);
                }
            }
            for (auto &oOperand : op.oOperand) {
                if (oOperand->tensor->GetRefCount() > 0) {
                    outcasts.Insert(oOperand);
                    ASSERT(incasts.count(oOperand) == 0);
                }
            }
        }
        if (isExplicit_) {
            for (const auto &functionParamInfo : functionParamInfos_) {
                auto beginTensor = functionParamInfo.beginValue;
                auto endTensor = functionParamInfo.beginValue;
            }
            int subscript = 0;
            for (const auto &functionParamInfo : functionParamInfos_) {
                auto beginTensor = functionParamInfo.beginValue;
                auto endTensor = functionParamInfo.endValue;
                ASSERT(beginTensor->GetRawTensor()->GetTensorInfo().subscript == -1);
                RawTensor::TensorInfo tensorInfo{functionParamInfo.key->Id(), subscript};
                beginTensor->GetRawTensor()->SetTensorInfo(tensorInfo);
                endTensor->GetRawTensor()->SetTensorInfo(tensorInfo);
                if (incasts.count(beginTensor) > 0) {
                    AddOriginIncast(beginTensor);
                    incasts.Remove({beginTensor});
                    beginTensor->GetRawTensor()->SetRawDataPtr(functionParamInfo.key->GetData());
                    beginTensor->GetRawTensor()->SetTensorInfo(tensorInfo);
                }
                if (outcasts.count(endTensor) > 0) {
                    AddOriginOutcast(endTensor);
                    outcasts.Remove({endTensor});
                    endTensor->GetRawTensor()->SetRawDataPtr(functionParamInfo.key->GetData());
                    endTensor->GetRawTensor()->SetTensorInfo(tensorInfo);
                }
                subscript++;
            }
            for (const auto &incast : incasts) {
                AddOriginIncast(incast);
                ASSERT(incast->GetRawTensor()->GetRawDataPtr() == nullptr);
                ASSERT(incast->GetRawTensor()->GetTensorInfo().tensorIndex == -1);
                ASSERT(incast->GetRawTensor()->GetTensorInfo().subscript == -1);
            }
            for (const auto &outcast : outcasts) {
                AddOriginOutcast(outcast);
                ASSERT(outcast->GetRawTensor()->GetRawDataPtr() == nullptr);
                ASSERT(outcast->GetRawTensor()->GetTensorInfo().tensorIndex == -1);
                ASSERT(outcast->GetRawTensor()->GetTensorInfo().subscript == -1);
            }
        } else { // !isExplicit_ or dynamicFunction
            std::map<LogicalTensorPtr, int> beginMapping;
            std::map<LogicalTensorPtr, int> endMapping;
            for (size_t i = 0; i < functionParamInfos_.size(); i++) {
                beginMapping.emplace(functionParamInfos_[i].beginValue, i);
                endMapping.emplace(functionParamInfos_[i].endValue, i);
            }

            int subscript = 0;
            std::unordered_map<std::shared_ptr<RawTensor>, std::shared_ptr<LogicalTensor>> updated;
            for (const auto &incast : incasts) {
                AddOriginIncast(incast);
                ASSERT(incast->GetShape() == incast->GetRawTensor()->GetRawShape());
                if (beginMapping.count(incast) == 0) {
                    continue;
                }
                auto &functionParamInfo = functionParamInfos_[beginMapping.at(incast)];
                if (functionParamInfo.key->GetData() == nullptr) {
                    continue;
                }
                if (updated.count(incast->GetRawTensor()) > 0) {
                    ASSERT(incast->GetShape() == updated.at(incast->GetRawTensor())->GetShape());
                    continue;
                }
                ASSERT(incast->GetRawTensor()->GetTensorInfo().subscript == -1);
                incast->GetRawTensor()->SetRawDataPtr(functionParamInfo.key->GetData());
                incast->GetRawTensor()->SetTensorInfo({functionParamInfo.key->Id(), subscript++});
                updated.emplace(incast->GetRawTensor(), incast);
            }
            for (const auto &outcast : outcasts) {
                AddOriginOutcast(outcast);
                ASSERT(outcast->GetShape() == outcast->GetRawTensor()->GetRawShape());
                if (endMapping.count(outcast) == 0) {
                    continue;
                }
                auto &functionParamInfo = functionParamInfos_[endMapping.at(outcast)];
                if (functionParamInfo.key->GetData() == nullptr) {
                    continue;
                }
                if (updated.count(outcast->GetRawTensor()) > 0) {
                    ASSERT(outcast->GetShape() == updated.at(outcast->GetRawTensor())->GetShape());
                    continue;
                }
                ASSERT(outcast->GetRawTensor()->GetTensorInfo().subscript == -1);
                outcast->GetRawTensor()->SetRawDataPtr(functionParamInfo.key->GetData());
                outcast->GetRawTensor()->SetTensorInfo({functionParamInfo.key->Id(), subscript++});
                updated.emplace(outcast->GetRawTensor(), outcast);
            }
        }
    }
    functionParamInfos_.clear();

    LogicalTensors inArgumentList, outArgumentList;
    if (IsGraphType(GraphType::TENSOR_GRAPH) || IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        inArgumentList = MakeIncasts(scope);
        outArgumentList = MakeOutcasts(scope);
        if (!isExplicit_) {
            for (auto arg : inArgumentList) {
                arg->GetRawTensor()->SetTensorSubScript(-1);
            }
            for (auto arg : outArgumentList) {
                arg->GetRawTensor()->SetTensorSubScript(-1);
            }
        }
        auto iodescDict = GetTensorDataForTensorGraph();
        GetTensorDataRefreshIO(iodescDict);
    } else if (graphType_ == GraphType::ROOT_GRAPH) {
    } else if (graphType_ == GraphType::LEAF_GRAPH) {
        for (auto &out : outCasts_) {
            CreateLeafInAndOutCast(out, outArgumentList);
        }
        for (auto &in : inCasts_) {
            /* actualRawmagic存在的场景下，前序 LogicalTensor 应该已经创建好，直接获取 */
            CreateLeafInAndOutCast(in, inArgumentList);
        }
    } else {
        ASSERT(false) << "Not support connecting other type of function currently";
    }
    std::vector<int> iOffset;
    std::vector<int> oOffset;
    std::vector< std::vector<SymbolicScalar>> argList;
    if (graphType_ == GraphType::LEAF_GRAPH) {
        argList = NormalizeCoa(iOffset, oOffset);
    }
    ComputeHash();
    return {std::move(inArgumentList), std::move(outArgumentList), std::move(iOffset), std::move(oOffset),
        std::move(argList)};
}

void Function::AddWhenNotExistOrAssert(const std::shared_ptr<LogicalTensor> &tensor,
                                             std::map<int, int> &magicToRawMagic,
                                             std::map<int, std::shared_ptr<LogicalTensor>> &magicToLogicalTensor) {
    if (auto it = magicToRawMagic.find(tensor->magic); it != magicToRawMagic.end()) {
        if (it->second != tensor->tensor->GetRawMagic()) {
            ALOG_INFO("Diff Magic Same RawMagic: ", it->second, magicToLogicalTensor[tensor->magic]->Dump(),
                      tensor->tensor->GetRawMagic(), tensor->Dump());
        }
    }
    magicToRawMagic[tensor->magic] = tensor->tensor->GetRawMagic();
    magicToLogicalTensor[tensor->magic] = tensor;
}

// Tensor magic should be only in the function, so same magic have same rawmagic
void Function::TensorMagicCheck() const {
    std::map<int, int> magicToRawMagic;
    std::map<int, std::shared_ptr<LogicalTensor>> magicToLogicalTensor;
    for (const auto &op : operations_) {
        std::map<int, int> subGraphIDCount;
        // Count subGraphID occurrences in iOperand
        for (const auto &tensor : op->iOperand) {
            AddWhenNotExistOrAssert(tensor, magicToRawMagic, magicToLogicalTensor);
        }

        // Count subGraphID occurrences in oOperand
        for (const auto &tensor : op->oOperand) {
            AddWhenNotExistOrAssert(tensor, magicToRawMagic, magicToLogicalTensor);
        }
    }
}

void Function::OperationLoopCheck(const std::string &errorMsg) {
    std::map<LogicalTensor *, std::vector<Operation *>> producers;
    std::map<LogicalTensor *, std::vector<Operation *>> consumers;

    for (auto &&op : operations_) {
        for (auto &&iop : op->GetIOperands()) {
            consumers[iop.get()].emplace_back(op.get());
        }
        for (auto &&oop : op->GetOOperands()) {
            producers[oop.get()].emplace_back(op.get());
        }
    }

    enum class DfsState {
        TODO = 0,
        IN_STACK,
        DONE,
    };

    std::map<int, DfsState> states;
    for (auto &&op : operations_) {
        int dupOpMagic = -1;
        auto cycleDetection = [&states, &dupOpMagic, &consumers](Operation *curr, auto self) -> bool {
            int magic = curr->GetOpMagic();
            if (states[magic] == DfsState::DONE) {
                return false;
            }

            if (states[magic] == DfsState::IN_STACK) {
                dupOpMagic = magic;
                ALOG_ERROR("[OperationLoopCheck] Cycle detected: ");
                ALOG_ERROR("[OperationLoopCheck]     Operation: ", curr->Dump());
                return true;
            }

            states[magic] = DfsState::IN_STACK;

            for (auto &&oop : curr->GetOOperands()) {
                for (auto *consumer : consumers[oop.get()]) {
                    if (self(consumer, self)) {
                        if (dupOpMagic != -1) {
                            ALOG_ERROR("[OperationLoopCheck]     Tensor:    ", oop->Dump());
                            ALOG_ERROR("[OperationLoopCheck]     Operation: ", curr->Dump());
                            if (magic == dupOpMagic) {
                                dupOpMagic = -1; // stop dumpping
                            }
                        }
                        return true;
                    }
                }
            }

            states[magic] = DfsState::DONE;

            return false;
        };

        ASSERT(!cycleDetection(op.get(), cycleDetection)) << errorMsg;
    }
}

bool Function::OperationLoopCheck()
{
    std::unordered_map<Operation*, int> inLinkNum;
    std::unordered_set<Operation*> visitedOp;
    std::vector<Operation*> visitStack;
    for (std::shared_ptr<Operation> op : operations_) {
        inLinkNum[op.get()] = op->ProducerOps().size();
        if (inLinkNum[op.get()] == 0) {
            visitStack.push_back(op.get());
        }
    }
    while (!visitStack.empty()) {
        Operation* currOp = visitStack.back();
        visitStack.pop_back();
        visitedOp.insert(currOp);
        for (Operation* nextOp : currOp->ConsumerOps()) {
            inLinkNum[nextOp] -= 1;
            if (inLinkNum[nextOp] == 0) {
                visitStack.push_back(nextOp);
            }
            if (inLinkNum[nextOp] < 0) {
                ALOG_ERROR_F("[OperationLoopCheck]     Operation:", nextOp->Dump());
                return false;
            }
        }
    }
    if (visitedOp.size() != operations_.size()) {
        ALOG_ERROR_F("[OperationLoopCheck]     Loop Detected.");
        return false;
    }
    return true;
}

void Function::GetAnIslandIncastsOutcasts(const std::map<int, int> &opToSubgraph, const int subgraphID,
    const std::vector<Operation *> &operations, std::vector<std::shared_ptr<LogicalTensor>> &iOperands,
    std::vector<std::shared_ptr<LogicalTensor>> &oOperands) const {
    std::set<std::shared_ptr<LogicalTensor>> allLogicalTensors;
    std::set<std::shared_ptr<LogicalTensor>> notOutcasts;
    std::set<std::shared_ptr<LogicalTensor>> notIncasts;
    for (const auto &opPtr : operations) {
        const auto &op = *opPtr;
        for (auto &&operand : op.GetIOperands()) {
            allLogicalTensors.insert(operand);
            bool usedbyotherfunction = false;
            for (auto &consumer : operand->GetConsumers()) {
                auto magic = consumer->GetOpMagic();
                if (consumer->GetOpcode() == Opcode::OP_CALL) {
                    continue;
                }
                ASSERT(opToSubgraph.find(magic) != opToSubgraph.end());
                if (opToSubgraph.at(magic) != subgraphID) {
                    usedbyotherfunction = true;
                    break;
                }
            }
            if (!usedbyotherfunction) {
                notOutcasts.insert(operand);
            }
        }

        for (auto &&operand : op.GetOOperands()) {
            allLogicalTensors.insert(operand);
            notIncasts.insert(operand);
        }
    }
    std::set_difference(allLogicalTensors.begin(), allLogicalTensors.end(), notIncasts.begin(), notIncasts.end(),
        std::inserter(iOperands, iOperands.begin()));
    std::set<std::shared_ptr<LogicalTensor>> tryOOperands;
    std::set_difference(allLogicalTensors.begin(), allLogicalTensors.end(), notOutcasts.begin(), notOutcasts.end(),
        std::inserter(tryOOperands, tryOOperands.begin()));
    std::set_difference(tryOOperands.begin(), tryOOperands.end(), iOperands.begin(), iOperands.end(),
        std::inserter(oOperands, oOperands.begin()));

    std::sort(iOperands.begin(), iOperands.end(), TensorPtrComparator());
    std::sort(oOperands.begin(), oOperands.end(), TensorPtrComparator());
}

auto Function::AnnotateOperation() {
    std::map<int, std::vector<Operation *>> subgraphs;
    std::map<int, int> opToSubgraph;
    for (auto &&op : Operations()) {
        // same op magic shall only appear once
        ASSERT(opToSubgraph.find(op.GetOpMagic()) == opToSubgraph.end());
        if (op.GetSubgraphID() < 0) {
            ALOG_DEBUG("Op magic: ", op.GetOpMagic(), "less than 0 graph: ", op.GetSubgraphID());
            continue;
        }
        subgraphs[op.GetSubgraphID()].emplace_back(&op);
        opToSubgraph[op.GetOpMagic()] = op.GetSubgraphID();
        ALOG_DEBUG("Operation: ", op.GetOpMagic(), "Belong To subgraph: ", op.GetSubgraphID());
    }

    for (const auto &pair : subgraphs) {
        ALOG_DEBUG("Subgraph ID: ", pair.first);
        for (const auto &op : pair.second) {
            ALOG_DEBUG("Operation: ", op->Dump());
        }
    }
    return std::make_pair(std::move(subgraphs), std::move(opToSubgraph));
}

std::unordered_set<int> Function::LoopCheck() {
    if (totalSubGraphCount_ == 0) {
        return {};
    }
    ALOG_INFO("LoopCheck begin.");

    auto [subgraphs, opToSubgraph] = AnnotateOperation();
    std::map<LogicalTensor *, std::vector<int>> producers;
    std::map<LogicalTensor *, std::vector<int>> consumers;

    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> iOperands;
    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> oOperands;

    for (auto &&[subgraphID, operations] : subgraphs) {
        if (subgraphID == NOT_IN_SUBGRAPH) {
            continue;
        }

        GetAnIslandIncastsOutcasts(opToSubgraph, subgraphID, operations, iOperands[subgraphID], oOperands[subgraphID]);

        for (auto &&iop : iOperands[subgraphID]) {
            consumers[iop.get()].push_back(subgraphID);
        }
        for (auto &&oop : oOperands[subgraphID]) {
            producers[oop.get()].push_back(subgraphID);
        }
    }

    enum class DfsState {
        TODO = 0,
        IN_STACK,
        DONE,
    };

    std::map<int, DfsState> states;
    std::unordered_set<int> subGraphInCycle;
    for (auto &&[subgraphID, operations] : subgraphs) {
        (void)operations;
        if (subgraphID == NOT_IN_SUBGRAPH) {
            continue;
        }

        int duplicatedSubgraphID = -2;
        auto cycleDetection = [&states, &duplicatedSubgraphID, &oOperands, &consumers, &subGraphInCycle](int currSubgraph, auto self) -> bool {
            if (states[currSubgraph] == DfsState::DONE) {
                return false;
            }

            if (states[currSubgraph] == DfsState::IN_STACK) {
                duplicatedSubgraphID = currSubgraph;
                ALOG_ERROR("[Cycle Detection] Cycle detected: ");
                ALOG_ERROR("[Cycle Detection]     subgraph id: ", currSubgraph);
                subGraphInCycle.emplace(currSubgraph);
                return true;
            }

            states[currSubgraph] = DfsState::IN_STACK;

            for (auto &&oop : oOperands[currSubgraph]) {
                for (int consumer : consumers[oop.get()]) {
                    if (self(consumer, self)) {
                        if (duplicatedSubgraphID != -2) {
                            ALOG_ERROR("[Cycle Detection]     tensor:      ", oop->Dump());
                            ALOG_ERROR("[producer]=");
                            for (const auto &producer : oop->GetProducers()) {
                                ALOG_ERROR(producer->GetOpMagic());
                            }
                            ALOG_ERROR("[Cycle Detection]     subgraph id: ", currSubgraph);
                            subGraphInCycle.emplace(currSubgraph);
                            if (currSubgraph == duplicatedSubgraphID) {
                                duplicatedSubgraphID = -2; // stop dumpping
                            }
                        }
                        return true;
                    }
                }
            }

            states[currSubgraph] = DfsState::DONE;
            return false;
        };
        if (cycleDetection(subgraphID, cycleDetection)) {
            return subGraphInCycle;
        }
    }
    return std::unordered_set<int>{};
}

void Function::SortOperations() {
    std::unordered_map<const Operation *, int> opMagicToIndex;
    for (size_t i = 0; i < operations_.size(); i++) {
        ASSERT(opMagicToIndex.count(operations_[i].get()) == 0);
        opMagicToIndex.emplace(operations_[i].get(), i);
    }
    std::vector<int> outDegree(operations_.size(), 0);
    std::vector<int> groupOutDegree(operationGroups_.size(), 0);

    for (auto &operation : operations_) {
        for (auto &iOperand : operation->iOperand) {
            for (const auto &producer : iOperand->GetProducers()) {
                if (producer->BelongTo() != this || producer == operation.get()) {
                    continue;
                }
                ASSERT(opMagicToIndex.count(producer) != 0);
                if (producer->GroupID() == NON_GROUP) {
                    outDegree[opMagicToIndex[producer]]++;
                } else if (operation->GroupID() != producer->GroupID()) {
                    groupOutDegree[producer->GroupID()]++;
                }
            }
        }
    }

    std::queue<int> q;
    for (size_t i = 0; i < operations_.size(); i++) {
        if (outDegree[i] == 0 && operations_[i]->GroupID() == NON_GROUP) {
            q.emplace(i);
        }
    }
    for (size_t i = 0; i < operationGroups_.size(); i++) {
        if (groupOutDegree[i] == 0) {
            auto &group = operationGroups_[i];
            for (auto riter = group.rbegin(); riter != group.rend(); ++riter) {
                ASSERT(opMagicToIndex.count(*riter) != 0);
                q.emplace(opMagicToIndex[*riter]);
            }
        }
    }

    std::vector<std::shared_ptr<Operation>> sortedOperations;
    while (!q.empty()) {
        const auto &op = operations_[q.front()];
        q.pop();
        sortedOperations.emplace_back(op);
        for (auto &iOperand : op->iOperand) {
            for (const auto &producer : iOperand->GetProducers()) {
                if (producer->BelongTo() != this || producer == op.get()) {
                    continue;
                }
                if (producer->GroupID() == NON_GROUP) {
                    auto nxtOpIndex = opMagicToIndex[producer];
                    if (--outDegree[nxtOpIndex] == 0) {
                        q.emplace(nxtOpIndex);
                    }
                } else if (op->GroupID() != producer->GroupID()) {
                    if (--groupOutDegree[producer->GroupID()] == 0) {
                        auto &group = operationGroups_[producer->GroupID()];
                        for (auto riter = group.rbegin(); riter != group.rend(); ++riter) {
                            ASSERT(opMagicToIndex.count(*riter) != 0);
                            q.emplace(opMagicToIndex[*riter]);
                        }
                    }
                }
            }
        }
    }
    for (const auto &operation : operations_) {
        if (operation->GroupID() == NON_GROUP) {
            ASSERT(outDegree[opMagicToIndex[operation.get()]] == 0);
        } else {
            ASSERT(groupOutDegree[operation->GroupID()] == 0);
        }
    }
    ASSERT(operations_.size() == sortedOperations.size());
    std::reverse_copy(sortedOperations.begin(), sortedOperations.end(), operations_.begin());
    RefreshOpPosition();

    sorted_ = true;
}

void Function::ScheduleBy(const std::vector<Operation *> &newList, bool needRefresh) {
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

void Function::AddOperationGroup(std::vector<Operation *> operationGroup) {
    size_t groupID = operationGroups_.size();
    for (const auto &operation : operationGroup) {
        ASSERT(operation->GroupID() == NON_GROUP);
        operation->SetGroupID(groupID);
    }
    operationGroups_.emplace_back(std::move(operationGroup));
    sorted_ = false;
}

void Function::CheckGroupValid() const {
    std::unordered_set<const Operation *> inGroupOp;
    for (size_t i = 0; i < operationGroups_.size(); i++) {
        for (auto &operation : operationGroups_[i]) {
            ASSERT(operation->GroupID() == i);
            ASSERT(inGroupOp.count(operation) == 0);
            inGroupOp.emplace(operation);
        }
    }
    for (const auto &operation : operations_) {
        ASSERT(inGroupOp.count(operation.get()) == (operation->GroupID() != NON_GROUP));
    }
}

void Function::RefreshOpPosition() {
    opPosition_.clear();
    for (size_t i = 0; i < operations_.size(); ++i) {
        ASSERT(opPosition_.count(operations_[i].get()) == 0);
        opPosition_.emplace(operations_[i].get(), i);
    }
}

bool Function::enableMagicLookupRecord_{false};
std::map<std::pair<int, int>, std::set<Operation *, LogicalTensor::CompareOp>> Function::tensorAndSubgraphToProducer_;

void Function::ProducerMagicLookup(const Function *function, const std::set<Operation *, LogicalTensor::CompareOp> &producers,
        const int subGraphId, int &index, std::unordered_map<int, int> &magic2index, std::stringstream &ss)
{
    for (auto &op : producers) {
        if (subGraphId != INT32_MIN && op->GetSubgraphID() != subGraphId) {
            continue;
        }
        bool isInBoundary = OpcodeManager::Inst().IsBoundaryIn(op->GetOpcode());
        if (isInBoundary) {
            /* 除了最高轴之外的所有内轴都纳入到hash的计算中 */
            for (size_t i = 1; i < op->iOperand[0]->tensor->rawshape.size(); i++) {
                ss << op->iOperand[0]->tensor->rawshape[i] << " ";
            }
        }
        bool isOutBoundary = OpcodeManager::Inst().IsBoundaryOut(op->GetOpcode());
        if (isOutBoundary) {
            /* 除了最高轴之外的所有内轴都纳入到hash的计算中 */
            for (size_t i = 1; i < op->oOperand[0]->tensor->rawshape.size(); i++) {
                ss << op->oOperand[0]->tensor->rawshape[i] << " ";
            }
        }
        ss << " " << op->GetOpcodeStr(true);
        for (const auto &attr : OpcodeManager::Inst().GetAttrs(op->GetOpcode())) {
            ss << " attr: [" << attr << " : " << op->DumpAttr(attr) << "]";
        }
        if (function->GetGraphType() != GraphType::LEAF_GRAPH) {
            ss << " tile shape: [" << op->GetTileShape().Dump() << "]";
        }
        if (op->GetOpAttribute() != nullptr) {
            if (op->GetOpcode() == Opcode::OP_ASSEMBLE) {
                if (!op->oOperand[0]->isSubGraphBoundary) {
                    ss << " " << op->GetOpAttribute()->Dump();
                }
            } else if ((!IsCopyIn(op->GetOpcode()) && !IsCopyOut(op->GetOpcode())) ||
                function->GetGraphType() != GraphType::LEAF_GRAPH) {
                ss << " " << op->GetOpAttribute()->Dump();
            }
        }
        MagicLookup(function, op->iOperand, subGraphId, index, magic2index, ss);
    }
}

void Function::MagicLookup(const Function *function, const std::vector<LogicalTensorPtr> &operand, const int subGraphId,
                           int &index, std::unordered_map<int, int> &magic2index, std::stringstream &ss)
{
    for (auto &t : operand) {
        if (magic2index.count(t->GetMagic()) && (function->inCastsSet_.count(t) == 0) &&
            t->GetProducers().size() != 0) {
            continue;
        }
        magic2index[t->GetMagic()] = index++;
        ss << "(" << " " << static_cast<int>(t->tensor->datatype) << " ";
        // Add shape information
        for (const auto &dim : t->shape) {
            ss << dim << " ";
        }
        if (function->IsFunctionType(FunctionType::STATIC)) {
            for (const auto &dim : t->oriShape) {
                ss << dim << " ";
            }
        }
        if (t->GetMemoryTypeOriginal() != MemoryType::MEM_DEVICE_DDR) {
            for (const auto &dim : t->offset) {
                ss << dim << " ";
            }
        }
        if (!enableMagicLookupRecord_) {
            ProducerMagicLookup(function, t->GetProducers(), subGraphId, index, magic2index, ss);
        } else if (tensorAndSubgraphToProducer_.count({t->GetMagic(), subGraphId}) > 0) {
            ProducerMagicLookup(function, tensorAndSubgraphToProducer_[{t->GetMagic(), subGraphId}], subGraphId,
                                index, magic2index, ss);
        }
        ss << ")";
    }
}

unsigned long Function::ComputeHashOrderless() const {
    std::stringstream ss;
    ss << std::to_string(static_cast<int>(functionType_)) << " ";
    ss << std::to_string(static_cast<int>(graphType_)) << " ";
    if (!IsGraphType({GraphType::LEAF_GRAPH, GraphType::LEAF_VF_GRAPH}) &&
        !IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TENSOR_GRAPH)) {
        ss << GetMagicName() << " ";
    }

    // Build using Polish Notation
    int index = 0;
    std::unordered_map<int, int> magic2index;
    // 只有leaf graph需要判断边界
    if (graphType_ == GraphType::LEAF_GRAPH) {
        MagicLookup(this, GetOutcast(), operations_[operations_.size() - 1]->GetSubgraphID(), index, magic2index, ss);
    } else {
        MagicLookup(this, GetOutcast(), INT32_MIN, index, magic2index, ss);
    }

    // 补充一些没有输出的Op的hash
    for (size_t i = 0; i < operations_.size(); i++) {
        if (operations_[i]->oOperand.empty()) {
            ss << " " << operations_[i]->GetOpcodeStr(true);
            for (const auto &attr : OpcodeManager::Inst().GetAttrs(operations_[i]->GetOpcode())) {
                ss << " attr: [" << attr << " : " << operations_[i]->DumpAttr(attr) << "]";
            }
            ss << " tile shape: [" << operations_[i]->GetTileShape().Dump() << "]";
            MagicLookup(this, operations_[i]->GetIOperands(), operations_[0]->GetSubgraphID(), index, magic2index, ss);
        }
    }
    index = 0;
    for (auto &i : inCasts_) {
        ss << "(i" << index++ << ")";
        bool isGlobal = (globalTensors_.count(i) != 0);
        if (isGlobal) {
            ss << "(Global)";
        }
    }
    for (auto &o : outCasts_) {
        ss << "(o" << index++ << ")";
        bool isGlobal = (globalTensors_.count(o) != 0);
        if (isGlobal) {
            ss << "(Global)";
        }
    }

    // fill symbol and loop range attr of dyndev tensor graph for dynamic binary reuse
    if (functionType_ == FunctionType::DYNAMIC_LOOP && dynloopAttr_ != nullptr) {
        ss << "symbol name:[" << dynloopAttr_->iterSymbolName << "]";
        ss << "loop range:[" << dynloopAttr_->loopRange.Dump() << "]";
    }
    // temporary avoidance, switch SUPPORT_DYNAMIC_UNALIGNED has an unexpected effect on dynamic binary reuse
    if (functionType_ == FunctionType::DYNAMIC) {
        ss << "dynamic unaligned:" << ConfigManager::Instance().GetCodeGenConfig(KEY_SUPPORT_DYNAMIC_UNALIGNED, false);
    }
    std::hash<std::string> hasher;
    auto result = hasher(ss.str());
    ALOG_DEBUG_F("Hash for function %d %s is %s hash value is %lu\n", functionMagic_, GetMagicName().c_str(),
        ss.str().c_str(), result);
    return result;
}

void Function::EraseOperations(bool eraseRelatedTensor, bool sorted) {
    std::vector<std::shared_ptr<Operation>> operations;
    for (auto &op : operations_) {
        if (!op->IsDeleted()) {
            operations.emplace_back(op);
            continue;
        }
        ASSERT(op->IsDeleted());
        for (auto &input : op->GetIOperands()) {
            input->RemoveConsumer(op.get());
            if (input->GetConsumers().empty() && eraseRelatedTensor && input->nodetype == NodeType::LOCAL) {
                GetTensorMap().Erase(input);
                for (auto &producer : input->GetProducers()) {
                    if (producer->BelongTo() == this) {
                        producer->GetOOperands().clear();
                    }
                }
            }
        }

        for (auto &output : op->GetOOperands()) {
            output->RemoveProducer(op.get());
            if (output->GetProducers().empty() && eraseRelatedTensor && output->nodetype == NodeType::LOCAL) {
                GetTensorMap().Erase(output);
                for (auto &consumer : output->GetConsumers()) {
                    if (consumer->BelongTo() == this) {
                        consumer->EraseInput(output);
                    }
                }
            }
        }
    }
    operations_ = operations;

    if (sorted) {
        SortOperations();
    }
}

void Function::EraseOperations(const OperationDeleter &deleter) {
    if (!sorted_) {
        SortOperations();
    }
    for (auto &op : operations_) {
        if (deleter(op, *this)) {
            op->SetAsDeleted();
        }
    }

    EraseOperations();
}

FunctionHash Function::ComputeHash() {
    if (functionHash_.GetHash() != 0 &&
        (functionType_ != FunctionType::DYNAMIC_LOOP && functionType_ != FunctionType::DYNAMIC)) {
        /* 动态类型的graph里面的op和tensor会随着循环的展开而变化，每次都需要刷新 */
        return functionHash_;
    }
    for (auto &ele : inCasts_) {
        inCastsSet_.emplace(ele);
    }
    functionHash_ = ComputeHashOrderless();
    return functionHash_;
}

void Function::AddOriginIncast(const std::shared_ptr<LogicalTensor> tensor) {
    originInCasts_.push_back(tensor);
}

void Function::AddOriginOutcast(const std::shared_ptr<LogicalTensor> tensor) {
    originOutCasts_.push_back(tensor);
}

Operation &Function::AddOperation(const std::string &opName, LogicalTensors iOperands,
    const LogicalTensors &oOperands, const bool updateTensorMap) {
    return AddOperation(FindOpcode(opName), iOperands, oOperands, updateTensorMap);
}

Operation &Function::AddOperation(const Opcode opCode, LogicalTensors iOperands, const LogicalTensors &oOperands,
    const bool updateTensorMap) {
    for (auto &iOperand : iOperands) {
        iOperand = ConnectWithOverlap(iOperand);
    }
    return AddRawOperation(opCode, iOperands, oOperands, updateTensorMap);
}

Operation &Function::AddRawOperation(
const Opcode opCode, const LogicalTensors &iOperands, const LogicalTensors &oOperands, bool updateTensorMap) {
    if (IsFunctionTypeAndGraphType(FunctionType::STATIC, {GraphType::ROOT_GRAPH, GraphType::LEAF_GRAPH})) {
        updateTensorMap = false;
        sorted_ = true;
    } else {
        sorted_ = functionType_ == FunctionType::DYNAMIC;
    }
    auto &op =
        operations_.emplace_back(std::make_shared<Operation>(*this, opCode, iOperands, oOperands, updateTensorMap));
    opPosition_.emplace(op.get(), operations_.size() - 1);
    return *operations_.back();
}

void Function::SetSameMemId(const Tensor &operand, Tensor &dst) {
    ASSERT(operand.GetDataType() == dst.GetDataType()) << " Check Dtype failed!";

    auto dstRaw = dst.GetStorage()->GetRawTensor();
    auto operandRaw = operand.GetStorage()->GetRawTensor();
    dstRaw->memoryId = operandRaw->memoryId;
    outIncastLinkMap[dstRaw] = operandRaw;
}

std::vector<Operation *> Function::GetAllInputOperations(const Operation &op) const {
    std::vector<Operation *> retOps;
    if (op.BelongTo() != this) {
        return retOps;
    }
    for (const LogicalTensorPtr &inTensor : op.GetIOperands()) {
        if (inTensor == nullptr) {
            continue;
        }
        for (auto &producer : inTensor->GetProducers()) {
            retOps.push_back(producer);
        }
    }
    return retOps;
}

std::vector<Operation *> Function::GetAllOutputOperations(const Operation &op) const {
    std::vector<Operation *> retOps;
    if (op.BelongTo() != this) {
        return retOps;
    }
    for (const LogicalTensorPtr &outTensor : op.GetOOperands()) {
        if (outTensor == nullptr) {
            continue;
        }
        for (auto &consumer : outTensor->GetConsumers()) {
            retOps.push_back(consumer);
        }
    }
    return retOps;
}

std::vector<Operation *> Function::GetCallopList() const {
    std::vector<Operation *> callopList;
    for (auto &op : operations_) {
        if (op->GetOpcode() != Opcode::OP_CALL) {
            continue;
        }
        callopList.push_back(op.get());
    }
    return callopList;
}

std::vector<std::shared_ptr<CallOpAttribute>> Function::GetCallopAttrList() const {
    std::vector<Operation *> callopList = GetCallopList();
    std::vector<std::shared_ptr<CallOpAttribute>> callopAttrList;
    for (auto callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        callopAttrList.push_back(callopAttr);
    }
    return callopAttrList;
}

std::vector<Function *> Function::GetCalleeFunctionList() const {
    std::vector<Operation *> callopList = GetCallopList();
    std::vector<Function *> calleeFuncList;
    for (auto callop : callopList) {
        auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
        auto calleeFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
        ASSERT(calleeFunc) << callopAttr->GetCalleeMagicName() << " is not in functionmap!";
        calleeFuncList.push_back(calleeFunc);
    }
    return calleeFuncList;
}

void Function::SubstituteIn(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor) {
    for (auto &operation : operations_) {
        auto &cur = *operation;
        std::unordered_set<std::shared_ptr<LogicalTensor>> replaced;
        for (size_t i = 0; i < cur.GetInputOperandSize(); i++) {
            LogicalTensorPtr inputTensor = cur.GetInputOperand(i);
            if (inputTensor != oldTensor) {
                continue;
            }
            if (replaced.count(inputTensor) == 0) {
                ASSERT(inputTensor->HasConsumer(cur));
                replaced.emplace(inputTensor);
            }
            cur.ReplaceIOperand(i, newTensor);
            if (cur.GetOpAttribute() == nullptr) {
                continue;
            }
            if (auto viewOpAttribute = dynamic_cast<ViewOpAttribute *>(cur.GetOpAttribute().get())) {
                // VIEW操作的offset要相应被修改。
                auto &fromOffset = viewOpAttribute->GetFrom();
                for (size_t j = 0; j < fromOffset.size(); j++) {
                    fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
                }
            } else if (auto copyOpAttribute = dynamic_cast<CopyOpAttribute *>(cur.GetOpAttribute().get())) {
                // CopyIn操作的offset要相应被修改。
                if (!copyOpAttribute->IsCopyOut()) {
                    auto [fromOffset, memType] = copyOpAttribute->GetCopyInAttr();
                    (void)memType;
                    for (size_t j = 0; j < fromOffset.size(); j++) {
                        fromOffset[j] -= oldTensor->offset[j] - newTensor->offset[j];
                    }
                    copyOpAttribute->SetFromOffset(fromOffset);
                }
            }
        }
    }
}

void Function::SubstituteOut(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor) {
    for (auto &operation : operations_) {
        auto &cur = *operation;
        for (size_t i = 0; i < cur.GetOOperands().size(); i++) {
            if (cur.GetOOperands()[i] == oldTensor) {
                ASSERT(cur.GetOOperands()[i]->shape == newTensor->shape);
                ASSERT(cur.GetOOperands()[i]->HasProducer(cur));
                cur.ReplaceOOperand(i, newTensor);
            }
        }
    }
}

void Function::Substitute(std::shared_ptr<LogicalTensor> oldTensor, std::shared_ptr<LogicalTensor> newTensor) {
    SubstituteIn(oldTensor, newTensor);
    SubstituteOut(oldTensor, newTensor);
}

void Function::RemoveOriginIncastConsumer(const std::shared_ptr<LogicalTensor> &originIncast) const {
    // 因为originIncast不输出本function，因此originIncast的producer的oOperand在本function外，需要移除对function内的消费者
    for (const auto &producer : originIncast->GetProducers()) {
        auto targetFunc = this;
        while (targetFunc != producer->BelongTo()) {
            if (!targetFunc->HasParent()) {
                targetFunc = nullptr;
                break;
            }
            targetFunc = &targetFunc->Parent();
        }
        ASSERT(targetFunc != nullptr);

        for (auto &oOperandForProducerOp : producer->oOperand) {
            auto &consumers = oOperandForProducerOp->GetConsumers();
            for (auto it = consumers.begin(); it != consumers.end();) {
                if ((*it)->BelongTo() == this) {
                    it = consumers.erase(it);
                } else {
                    it++;
                }
            }
        }
    }
}

void Function::UpdateLinkMap(const std::shared_ptr<LogicalTensor> &oriLogicalTensor, const std::shared_ptr<LogicalTensor> &newLogicalTensor, const bool isOutCast) {
    if (isOutCast) {
        //  update outcast
        auto it = outIncastLinkMap.find(oriLogicalTensor->tensor);
        if (it != outIncastLinkMap.end()) {
            outIncastLinkMap[newLogicalTensor->tensor] = it->second;
            newLogicalTensor->tensor->memoryId = it->second->memoryId;
            ALOG_DEBUG_F("UpdateLinkMap memoryId to %d  \n", it->second->memoryId);
            outIncastLinkMap.erase(it);
        }
    } else {
        //  update incast
        for (auto &ele : outIncastLinkMap) {
            if (ele.second == oriLogicalTensor->tensor) {
                ele.second = newLogicalTensor->tensor;
            }
        }
    }
}

std::pair<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensor>> Function::CreateIncastTensor(const std::shared_ptr<LogicalTensor> &inArgument) {
    auto idx = inCasts_.size();
    auto newSymbol = inArgument->tensor->GetSymbol();
    if (newSymbol == "") {
        newSymbol = "INCAST_SYMBOL" + std::to_string(idx);
    }
    auto incastSymbol = std::make_shared<LogicalTensor>(*this, inArgument->tensor->datatype, inArgument->shape,
        inArgument->tensor->GetDynRawShape(), newSymbol, NodeType::INCAST, inArgument->tensorfmt);
    incastSymbol->tensor->SetRawDataPtr(inArgument->tensor->GetRawDataPtr());
    incastSymbol->tensor->SetTensorInfo(inArgument->tensor->GetTensorInfo());
    tensorMap_.Insert(incastSymbol);
    inCasts_.push_back(incastSymbol);
    incastToInArgumentDict[incastSymbol] = inArgument;

    UpdateLinkMap(inArgument, incastSymbol);
    return std::pair<std::shared_ptr<LogicalTensor>, std::shared_ptr<LogicalTensor>>{incastSymbol, nullptr};
}

void Function::CreateFromIncast(const std::shared_ptr<LogicalTensor> &symbol,
                                      const std::shared_ptr<LogicalTensor> &newIncast,
                                      const std::shared_ptr<LogicalTensor> &originIncast) {
    auto &incastOp = AddOperation(Opcode::OP_VIEW, {symbol}, {newIncast});
    incastOp.SetAttr(OpAttributeKey::isGlobalInput, true);

    auto validShape = originIncast->GetDynValidShape();
    if (validShape.empty()) {
        validShape = GetViewValidShape(symbol->GetDynValidShape(), originIncast->GetOffset(),
            originIncast->GetDynOffset(), newIncast->GetShape());
    }
    incastOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(originIncast->GetOffset(),
        originIncast->GetDynOffset(), validShape));
    newIncast->UpdateDynValidShape(validShape);
}

void Function::ReplaceMaybeParams(const std::shared_ptr<LogicalTensor> &newIncast,
                                        const std::shared_ptr<LogicalTensor> &originIncast) {
    auto it = std::find(originInCasts_.begin(), originInCasts_.end(), originIncast);
    ASSERT(it != originInCasts_.end());
    *it = newIncast;
}

LogicalTensors Function::MakeIncasts(const std::shared_ptr<TensorSlotScope> &scope) {
    ASSERT(IsGraphType(GraphType::TENSOR_GRAPH) || IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH));
    ASSERT(HasParent());
    LogicalTensors inArgumentList;
    std::unordered_set<int> appearedRawIncasts;
    std::vector<std::shared_ptr<RawTensor>> rawIncasts;
    std::map<int, std::vector<std::shared_ptr<LogicalTensor>>> incastWithSameRaw;
    std::map<std::shared_ptr<RawTensor>, std::shared_ptr<LogicalTensor>> rawToIncast;
    for (auto &originIncast : originInCasts_) {
        incastWithSameRaw[originIncast->tensor->rawmagic].emplace_back(originIncast);
        if (appearedRawIncasts.count(originIncast->tensor->rawmagic) != 0) {
            continue;
        }
        appearedRawIncasts.emplace(originIncast->tensor->rawmagic);
        rawIncasts.emplace_back(originIncast->tensor);
        rawToIncast[originIncast->tensor] = originIncast;
    }

    for (auto &[rawMagic, sameRawIncasts] : incastWithSameRaw) {
        (void)rawMagic;
        sort(sameRawIncasts.begin(), sameRawIncasts.end(),
            [](const auto &a, const auto &b) -> bool { return a->shape < b->shape; });
    }

    int idx = 0;
    for (auto &rawIncast : rawIncasts) {
        const auto &sameRawIncasts = incastWithSameRaw[rawIncast->rawmagic];

        std::vector<int> zeroOffset(rawIncast->rawshape.size(), 0);
        auto inArgument = std::make_shared<LogicalTensor>(Parent(), rawIncast, zeroOffset, rawIncast->rawshape,
            NodeType::LOCAL, rawToIncast[rawIncast]->tensorfmt);
        inArgumentList.push_back(inArgument);

        auto [incastSymbol, incastLocalBuf] = CreateIncastTensor(inArgument);
        if (scope) {
            scope->incastToInArgumentDict[incastSymbol] = inArgument;
            scope->incastToInOriginalDict[incastSymbol].insert(sameRawIncasts.begin(), sameRawIncasts.end());
        }
        (void)incastLocalBuf;

        std::map<ViewKey, std::shared_ptr<LogicalTensor>> newincastMap;
        for (auto &originIncast : sameRawIncasts) {
            auto viewKey = ViewKey(originIncast->tensor->rawmagic, originIncast->shape, originIncast->offset, originIncast->GetDynOffset());
            std::shared_ptr<LogicalTensor> newIncast;
            if (newincastMap.count(viewKey) != 0) {
                newIncast = newincastMap[viewKey];
            } else {
                newIncast = std::make_shared<LogicalTensor>(*this, originIncast->tensor->datatype, originIncast->shape,
                    "INCAST_LOCAL_BUF" + std::to_string(idx++), NodeType::LOCAL, originIncast->tensorfmt);
                ASSERT(originIncast->conflicterTensors.empty());
                newIncast->CopyMemoryType(originIncast);

                newincastMap.emplace(viewKey, newIncast);
                CreateFromIncast(incastSymbol, newIncast, originIncast);
            }

            Substitute(originIncast, newIncast);

            ReplaceMaybeParams(newIncast, originIncast);

            for (const auto &producer : inArgument->GetProducers()) {
                ASSERT(producer->BelongTo() != this)
                    << inArgument->magic << "-> producer. funcMagic = " << producer->BelongTo()->GetFuncMagic()
                    << " producer = " << producer->GetOpMagic() << "inArgument = " << inArgument->Dump() << std::endl;
            }
            for (const auto &consumer : inArgument->GetConsumers()) {
                ASSERT(consumer->BelongTo() != this);
            }
        }
    }

    for (auto &rawIncast : rawIncasts) {
        const auto &sameRawIncasts = incastWithSameRaw[rawIncast->rawmagic];
        for (const auto &originIncast : sameRawIncasts) {
            RemoveOriginIncastConsumer(originIncast);
        }
    }

    return inArgumentList;
}

LogicalTensors Function::MakeOutcasts(const std::shared_ptr<TensorSlotScope> &scope) {
    ASSERT(IsGraphType(GraphType::TENSOR_GRAPH) || IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH));
    ASSERT(HasParent());
    LogicalTensors outArgumentList;

    std::unordered_set<int> appearedRawOutcasts;
    std::vector<std::shared_ptr<RawTensor>> rawOutcasts;
    std::map<int, std::set<std::shared_ptr<LogicalTensor>, CompareTensorPtr>> outcastWithSameRaw;
    std::map<std::shared_ptr<RawTensor>, std::shared_ptr<LogicalTensor>> rawToOutcast;
    for (const auto &originOutcast : originOutCasts_) {
        ASLOGI("originOut cast name %d %d", originOutcast->magic, originOutcast->GetRawMagic());
        outcastWithSameRaw[originOutcast->tensor->rawmagic].emplace(originOutcast);
        if (appearedRawOutcasts.count(originOutcast->tensor->rawmagic) != 0) {
            continue;
        }
        appearedRawOutcasts.emplace(originOutcast->tensor->rawmagic);
        rawOutcasts.emplace_back(originOutcast->tensor);
        rawToOutcast[originOutcast->tensor] = originOutcast;
    }

    ASLOGI("raw out cast number %zu", rawOutcasts.size());
    for (const auto &rawOutcast : rawOutcasts) {
        const auto &sameRawOutcasts = outcastWithSameRaw[rawOutcast->rawmagic];
        std::vector<int> nonOffsets(rawOutcast->rawshape.size(), 0);

        auto idx = outCasts_.size();
        auto newSymbol = rawOutcast->GetSymbol();
        if (newSymbol == "") {
            newSymbol = "OUTCAST_SYMBOL" + std::to_string(idx);
        }
        auto rawSymbol = std::make_shared<LogicalTensor>(*this, rawOutcast->datatype, rawOutcast->rawshape,
            rawOutcast->GetDynRawShape(), newSymbol, NodeType::OUTCAST, rawToOutcast[rawOutcast]->tensorfmt);
        auto rawBuf = std::make_shared<LogicalTensor>(*this, rawOutcast->datatype, rawOutcast->rawshape,
            rawOutcast->GetDynRawShape(), "OUTCAST_LOCAL_BUF" + std::to_string(idx), NodeType::LOCAL, rawToOutcast[rawOutcast]->tensorfmt);
        auto outArgument = std::make_shared<LogicalTensor>(Parent(), rawOutcast, nonOffsets, rawOutcast->rawshape, NodeType::LOCAL, rawToOutcast[rawOutcast]->tensorfmt);
        rawSymbol->tensor->UpdateDynRawShape(rawOutcast->GetDynRawShape());
        rawBuf->tensor->UpdateDynRawShape(rawOutcast->GetDynRawShape());
        rawSymbol->tensor->SetRawDataPtr(rawOutcast->GetRawDataPtr());
        rawSymbol->tensor->SetTensorInfo(rawOutcast->GetTensorInfo());
        Parent().tensorMap_.Insert(outArgument);
        outArgumentList.push_back(outArgument);
        UpdateLinkMap(outArgument, rawSymbol, true);
        outCasts_.emplace_back(rawSymbol);
        if (scope) {
            scope->outcastToOutArgumentDict[rawSymbol] = outArgument;
            scope->outcastToOutOriginalDict[rawSymbol].insert(sameRawOutcasts.begin(), sameRawOutcasts.end());
        }

        std::vector<std::vector<int>> newOutcastOffsets;
        std::vector<std::shared_ptr<LogicalTensor>> iOperand;
        std::vector<std::shared_ptr<LogicalTensor>> oOperand = {rawSymbol};
        ASLOGI("same raw out cast number %zu", sameRawOutcasts.size());
        for (auto &originOutcast : sameRawOutcasts) {
            auto newOutcast = rawBuf->View(*this, originOutcast->shape, originOutcast->offset);
            auto oldConsumers = originOutcast->GetConsumers(); // only for check
            auto oldProducers = originOutcast->GetProducers(); // only for check
            tensorMap_.Insert(newOutcast);
            Substitute(originOutcast, newOutcast);
            ASSERT(newOutcast->GetProducers() == oldProducers);
            ASSERT(newOutcast->GetConsumers() == oldConsumers);
            ASSERT(originOutcast->GetProducers().empty());
            auto it = std::find(originOutCasts_.begin(), originOutCasts_.end(), originOutcast);
            ASSERT(it != originOutCasts_.end());
            *it = newOutcast;
            newOutcastOffsets.emplace_back(originOutcast->offset);
            iOperand.emplace_back(newOutcast);
        }
        ASSERT(rawSymbol->GetProducers().empty());
        ASSERT(iOperand.size() == newOutcastOffsets.size());
        for (size_t i = 0; i < iOperand.size(); i++) {
            auto producerSet = iOperand[i]->GetProducers();
            auto anyDAssemble = std::any_of(producerSet.begin(), producerSet.end(), [](Operation *op){
                return op->GetOpcode() == Opcode::OP_ASSEMBLE && op->HasAttribute("dassemble");
            });
            if (anyDAssemble) {
                for (auto producer : producerSet) {
                    auto producerAttr = std::static_pointer_cast<AssembleOpAttribute>(producer->GetOpAttribute());
                    auto [offset, dynOffset] = TensorOffset::Add(iOperand[i]->GetOffset(), iOperand[i]->GetDynOffset(), producerAttr->GetToOffset(), producerAttr->GetToDynOffset());
                    auto &assembleOp = AddOperation(Opcode::OP_ASSEMBLE, {producer->GetIOperands()[0]}, oOperand);
                    assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offset, dynOffset));
                    producer->SetAsDeleted();
                }
                if (scope) {
                    scope->partialUpdateOutcastSet.insert(rawSymbol);
                }
            } else {
                auto &assembleOp = AddOperation(Opcode::OP_ASSEMBLE, {iOperand[i]}, oOperand);
                assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(newOutcastOffsets[i]));
            }
        }
        // Substitute alive tensor magics
        Program::GetInstance().UpdateAliveTensorsParent(rawOutcast->rawmagic, Parent());
    }

    for (auto it = tensorMap_.inverseMap_.begin(); it != tensorMap_.inverseMap_.end();) {
        if (appearedRawOutcasts.count(it->second->tensor->rawmagic) > 0) {
            it = tensorMap_.inverseMap_.erase(it);
        } else {
            it++;
        }
    }

    std::vector<int> magicToRemove;
    for (auto it = tensorMap_.tensorMap_.begin(); it != tensorMap_.tensorMap_.end();) {
        if (appearedRawOutcasts.count(it->first) > 0) {
            magicToRemove.push_back(it->first);
            it++;
        } else {
            it++;
        }
    }
    for(auto rmagic: magicToRemove){
        tensorMap_.EraseRawMagic(rmagic);
    }

    for (const auto &tensor : outArgumentList) {
        ASSERT(tensor->GetProducers().empty());
    }

    return outArgumentList;
}

bool Function::IsFlattening() const {
    return IsFunctionTypeAndGraphType(FunctionType::STATIC, {GraphType::TENSOR_GRAPH, GraphType::TILE_GRAPH});
}

FunctionType Function::GetFunctionType() const {
    return functionType_;
}

void Function::SetFunctionType(FunctionType type) {
    functionType_ = type;
}

std::string Function::GetFunctionTypeStr() const{
    return GetFunctionTypeNameDict().Find(functionType_);
}

GraphType Function::GetGraphType() const {
    return graphType_;
}

void Function::SetGraphType(GraphType type) {
    graphType_ = type;
}

bool Function::IsFunctionType(FunctionType type) const {
    return functionType_ == type;
}

bool Function::IsFunctionType(std::set<FunctionType> types) const {
    return types.count(functionType_) != 0;
}

bool Function::IsGraphType(GraphType type) const {
    return graphType_ == type;
}

bool Function::IsGraphType(std::set<GraphType> types) const {
    return types.count(graphType_) != 0;
}

bool Function::IsFunctionTypeAndGraphType(FunctionType funcType, GraphType graphType) const {
    return functionType_ == funcType && graphType_ == graphType;
}

bool Function::IsFunctionTypeAndGraphType(FunctionType funcType, std::set<GraphType> graphTypes) const {
    return IsFunctionType(funcType) && IsGraphType(graphTypes);
}

bool Function::IsFunctionTypeAndGraphType(std::set<FunctionType> funcTypes, GraphType graphType) const {
    return IsFunctionType(funcTypes) && IsGraphType(graphType);
}

bool Function::IsFunctionTypeAndGraphType(std::set<FunctionType> funcTypes, std::set<GraphType> graphTypes) const {
    return IsFunctionType(funcTypes) && IsGraphType(graphTypes);
}

void Function::DumpJsonFile(std::string fileName) {
    auto filePath = config::LogTopFolder() + "/" + funcRawName_ + ".json";
    if (!fileName.empty()) {
        filePath = fileName;
    }
    std::ofstream file(filePath);
    Json progDump;
    progDump["version"] = T_VERSION;
    progDump["functions"].push_back(DumpJson());
    progDump["entryhash"] = this->GetFunctionHash().Data();
    file << progDump.dump(1) << std::endl;
    file.close();
}

struct RawTensorCompare {
    bool operator()(const std::shared_ptr<RawTensor>& a, const std::shared_ptr<RawTensor>& b) const {
        return a->rawmagic < b->rawmagic;
    }
};

struct TensorCompare {
    bool operator()(const std::shared_ptr<LogicalTensor>& a, const std::shared_ptr<LogicalTensor>& b) const {
        return a->magic < b->magic;
    }
};

Json Function::DumpJson(bool useTable) {
    Json funcDump;
    funcDump[T_FIELD_KIND] = static_cast<int>(Kind::T_KIND_FUNCTION);
    funcDump["rawname"] = funcRawName_;
    funcDump["funcmagic"] = GetFuncMagic();
    if (parent_ != nullptr) {
        funcDump["parent_funcmagic"] = parent_->GetFuncMagic();
    }
    funcDump["functype"] = functionType_;
    funcDump["graphtype"] = graphType_;
    funcDump["func_magicname"] = funcMagicName_;
    funcDump["_opseed"] = opSeed_;
    funcDump["_magicseed"] = magicSeed_;
    funcDump["_rawid"] = IdGen<IdType::RAW_TENSOR>::Inst().CurId();
    funcDump["_funcid"] = IdGen<IdType::FUNCTION>::Inst().CurId();
    funcDump["_l1_reuse_num"] = paramConfigs_.l1ReuseNum;
    funcDump["_cube_nbuffer_num"] = paramConfigs_.cubeNBufferNum;
    funcDump["_sg_cycle_upperbound"] = paramConfigs_.sgCycleUpperBound;
    funcDump["_sg_cycle_lowerbound"] = paramConfigs_.sgCycleLowerBound;
    funcDump["_sg_parallel_num"] = paramConfigs_.sgParallelNum;
    funcDump["_sg_copyin_threshold"] = paramConfigs_.sgCopyInThreshold;
    funcDump["_dbtype"] = paramConfigs_.dbType;
    funcDump["_nbuffer_num"] = paramConfigs_.NbufferNum;
    funcDump["_total_subgraph_count"] = totalSubGraphCount_;
    funcDump["_ooo_preschedule_method_default"] = paramConfigs_.OoOPreScheduleMethodDefault;

    if (useTable) {
        std::vector<std::pair<int, std::vector<int>>> incasts;
        std::vector<std::pair<int, std::vector<int>>> outcasts;
        size_t inSize = inCasts_.size();
        for (size_t i = 0; i < inSize; i++) {
            std::pair<int, std::vector<int>> incast;
            incast.first = inCasts_[i]->GetMagic();
            if (slotScope_ != nullptr) {
                incast.second = slotScope_->ioslot.incastSlot[i];
            } else {
                std::vector<int> emptyIncast;
                incast.second = emptyIncast;
            }
            incasts.push_back(incast);
        }
        size_t outSize = outCasts_.size();
        for (size_t i = 0; i < outSize; i++) {
            std::pair<int, std::vector<int>> outcast;
            outcast.first = outCasts_[i]->GetMagic();
            if (slotScope_ != nullptr) {
                outcast.second = slotScope_->ioslot.outcastSlot[i];
            } else {
                std::vector<int> emptyOutcast;
                outcast.second = emptyOutcast;
            }
            outcasts.push_back(outcast);
        }

        funcDump["incasts"] = incasts;
        funcDump["outcasts"] = outcasts;
    } else {
        Json incasts = Json::array();
        Json outcasts = Json::array();
        Json globalTensors = Json::array();
        for (auto &i : inCasts_) {
            incasts.push_back(i->DumpJson(true));
        }
        for (auto &o : outCasts_) {
            outcasts.push_back(o->DumpJson(true));
        }
        funcDump["incasts"] = incasts;
        funcDump["outcasts"] = outcasts;
    }

    std::set<int> globalTensorSet;
    for (auto &t : globalTensors_) {
        globalTensorSet.emplace(t->GetMagic());
    }
    std::vector<int> globalTensorVec;
    for (auto &tMagic : globalTensorSet) {
        globalTensorVec.emplace_back(tMagic);
    }
    funcDump["global_tensors"] = globalTensorVec;
    funcDump["static"]["global_tensors"] = funcDump["global_tensors"];
    Json operations = Json::array();
    if (useTable) {
        for (const auto &op : Operations()) {
            operations.push_back(op.DumpJson(false));
        }
    } else {
        for (const auto &op : Operations()) {
            operations.push_back(op.DumpJson(true));
        }
    }
    funcDump["operations"] = operations;
    funcDump["hash"] = functionHash_.Data();

    std::vector<std::string> resultSemanticLabels(semanticLabels_.begin(), semanticLabels_.end());
    std::sort(resultSemanticLabels.begin(), resultSemanticLabels.end());
    funcDump["semantic_label"] = resultSemanticLabels;

    if (leafFuncAttr_ != nullptr && leafFuncAttr_->coreType != CoreType::INVALID) {
        funcDump["leaf_func_attr"]["coretype"] = leafFuncAttr_->coreType;
    }

    if (rootFunc_ != nullptr) {
        funcDump["root_func_magic"] = rootFunc_->GetFuncMagic();
    }
    if (!programs_.empty()) {
        Json programsJson;
        for (auto &ele : programs_) {
            programsJson[ele.first] = ele.second->GetFuncMagic();
        }
        funcDump["programs"] = programsJson;
        funcDump["topo"] = topoInfo_.DumpJson();
        funcDump["static"]["topo"] = funcDump["topo"];
    }
    if (graphType_ == GraphType::LEAF_GRAPH) {
        funcDump["subfunc_param"] = parameter_.ToJson();
        funcDump["static"]["subfunc_param"] = funcDump["subfunc_param"];
    }

    auto aicIt = readySubGraphIds_.find(CoreType::AIC);
    if (aicIt != readySubGraphIds_.end() && !aicIt->second.empty()) {
        funcDump["aic_ready_subgraph_ids"] = aicIt->second;
        funcDump["static"]["aic_ready_subgraph_ids"] = funcDump["aic_ready_subgraph_ids"];
    }

    auto aivIt = readySubGraphIds_.find(CoreType::AIV);
    if (aivIt != readySubGraphIds_.end() && !aivIt->second.empty()) {
        funcDump["aiv_ready_subgraph_ids"] = aivIt->second;
        funcDump["static"]["aiv_ready_subgraph_ids"] = funcDump["aiv_ready_subgraph_ids"];
    }

    auto aicpuIt = readySubGraphIds_.find(CoreType::AICPU);
    if (aicpuIt != readySubGraphIds_.end() && !aicpuIt->second.empty()) {
        funcDump["aicpu_ready_subgraph_ids"] = aicpuIt->second;
        funcDump["static"]["aicpu_ready_subgraph_ids"] = funcDump["aicpu_ready_subgraph_ids"];
    }

    if (useTable) {
        std::set<std::shared_ptr<RawTensor>, RawTensorCompare> rawTensorSet;
        std::set<std::shared_ptr<LogicalTensor>, TensorCompare> tensorSet;
        for (const auto &incast : inCasts_) {
            tensorSet.insert(incast);
            rawTensorSet.insert(incast->tensor);
        }
        for (const auto &outcast : outCasts_) {
            tensorSet.insert(outcast);
            rawTensorSet.insert(outcast->tensor);
        }
        for (const auto &op : Operations()) {
            for (auto &i : op.GetIOperands()) {
                tensorSet.insert(i);
                rawTensorSet.insert(i->tensor);
            }
            for (auto &o : op.GetOOperands()) {
                tensorSet.insert(o);
                rawTensorSet.insert(o->tensor);
            }
        }
        std::vector<std::shared_ptr<RawTensor>> rawTensorList(rawTensorSet.begin(), rawTensorSet.end());
        std::vector<std::shared_ptr<LogicalTensor>> tensorList(tensorSet.begin(), tensorSet.end());
        std::sort(rawTensorList.begin(), rawTensorList.end(),
            [](auto l, auto r) { return l->GetRawMagic() < r->GetRawMagic(); });
        std::sort(tensorList.begin(), tensorList.end(), [](auto l, auto r) { return l->GetMagic() < r->GetMagic(); });

        Json rawtensors = Json::array();
        Json tensors = Json::array();
        for (auto &rawTensor : rawTensorList) {
            rawtensors.push_back(rawTensor->DumpJson());
        }
        for (auto &tensor : tensorList) {
            tensors.push_back(tensor->DumpJson(false));
        }
        funcDump["rawtensors"] = rawtensors;
        funcDump["tensors"] = tensors;
    }
    if (functionType_ == FunctionType::DYNAMIC_LOOP) {
        auto loopAttr = GetDynloopAttribute();
        if (loopAttr != nullptr) {
            //itername
            std::string itername = loopAttr->iterSymbolName;
            funcDump["dynamic"]["itername"] = itername;

            // begin
            SymbolicScalar begin = loopAttr->Begin();
            auto jbegin = ToJson(begin);
            if (jbegin.size() > 0) {
                funcDump["dynamic"]["begin"] = jbegin;
            }

            // end
            SymbolicScalar end = loopAttr->End();
            auto jend = ToJson(end);
            if (jend.size() > 0) {
                funcDump["dynamic"]["end"] = jend;
            }

            // step
            SymbolicScalar step = loopAttr->Step();
            auto jstep = ToJson(step);
            if (jstep.size() > 0) {
                funcDump["dynamic"]["step"] = jstep;
            }

            // originalBegin
            SymbolicScalar originalBegin = loopAttr->originalRange.Begin();
            auto jOriBegin = ToJson(originalBegin);
            if (jOriBegin.size() > 0) {
                funcDump["dynamic"]["originalBegin"] = jOriBegin;
            }

            // originalEnd
            SymbolicScalar originalEnd = loopAttr->originalRange.End();
            auto jOriEnd = ToJson(originalEnd);
            if (jOriEnd.size() > 0) {
                funcDump["dynamic"]["originalEnd"] = jOriEnd;
            }

            // unrollTimes
            int unrollTimes = loopAttr->unrollTimes;
            funcDump["dynamic"]["unrollTimes"] = unrollTimes;

            // pathList
            Json loopFuncPathList = Json::array();
            for (auto &path : loopAttr->pathList) {
                Json pathJson = Json::array();
                auto opmagic = path.callop->GetOpMagic();
                for (auto &pathCond : path.pathCondList) {
                    Json pathCondJson = Json::array();
                    SymbolicScalar cond = pathCond.GetCond();
                    auto jcond = ToJson(cond);
                    pathCondJson.push_back(jcond);
                    pathCondJson.push_back(pathCond.IsSat());
                    pathJson.push_back(pathCondJson);
                }
                if (pathJson.size() > 0) {
                    loopFuncPathList.push_back({opmagic, pathJson});
                }
            }
            if (loopFuncPathList.size() > 0) {
                funcDump["dynamic"]["paths"] = loopFuncPathList;
            }
        }
    }
    return funcDump;
}

void Function::LoadTensorJson(const std::shared_ptr<Function> &func, const Json &funcDump,
                                    const std::unordered_map<int, std::shared_ptr<RawTensor>> &rawTensorDict,
                                    std::unordered_map<int, std::shared_ptr<LogicalTensor>> &tensorDict) {
    if (funcDump.count("tensors") != 0) {
        for (auto &tensorDump : funcDump["tensors"]) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, tensorDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }
    for (auto &iDump : funcDump["incasts"]) {
        if (!iDump[0].is_number()) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, iDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }
    for (auto &oDump : funcDump["outcasts"]) {
        if (!oDump[0].is_number()) {
            std::shared_ptr<LogicalTensor> tensor = LogicalTensor::LoadJson(*func, rawTensorDict, oDump);
            tensorDict[tensor->GetMagic()] = tensor;
        }
    }

    for (auto &tDump : funcDump["global_tensors"]) {
        int magic = tDump.get<int>();
        auto &t = tensorDict[magic];
        func->globalTensors_.emplace(t);
    }

    for (auto &iDump : funcDump["incasts"]) {
        int magic = iDump[0].is_number() ? iDump[0].get<int>() : iDump["magic"].get<int>();
        auto &in = tensorDict[magic];
        func->inCasts_.push_back(in);
        func->GetTensorMap().Insert(in);
    }
    for (auto &oDump : funcDump["outcasts"]) {
        int magic = oDump[0].is_number() ? oDump[0].get<int>() : oDump["magic"].get<int>();
        func->outCasts_.push_back(tensorDict[magic]);
    }
    /* 补充对于临时workspace的tensor添加 */
    for (auto &ele : tensorDict) {
        func->GetTensorMap().Insert(ele.second, false);
    }

    for (auto &opDump : funcDump["operations"]) {
        auto op = Operation::LoadJson(*func, tensorDict, opDump);
        func->operations_.push_back(op);
    }
}

std::shared_ptr<Function> Function::LoadJson(Program &belongTo, const Json &funcDump) {
    ASSERT(funcDump[T_FIELD_KIND].get<int>() == static_cast<int>(Kind::T_KIND_FUNCTION));

    int funcmagic = funcDump["funcmagic"].get<int>();
    std::string rawname = funcDump["rawname"].get<std::string>();
    std::shared_ptr<Function> func =
        std::make_shared<Function>(belongTo, rawname + "_" + std::to_string(funcmagic), rawname, nullptr);
    func->funcMagicName_ = funcDump["func_magicname"];
    func->functionMagic_ = funcmagic;
    func->functionType_ = static_cast<FunctionType>(funcDump["functype"].get<int>());
    func->graphType_ = static_cast<GraphType>(funcDump["graphtype"].get<int>());
    func->sorted_ = true;
    std::unordered_map<int, std::shared_ptr<RawTensor>> rawTensorDict;
    if (funcDump.count("rawtensors") != 0) {
        for (auto &rawTensorDump : funcDump["rawtensors"]) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(rawTensorDump);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }
    for (auto &iDump : funcDump["incasts"]) {
        if (!iDump[0].is_number() && !iDump[T_FIELD_RAWTENSOR].is_number()) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(iDump[T_FIELD_RAWTENSOR]);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }
    for (auto &oDump : funcDump["outcasts"]) {
        if (!oDump[0].is_number() && !oDump[T_FIELD_RAWTENSOR].is_number()) {
            std::shared_ptr<RawTensor> rawTensor = RawTensor::LoadJson(oDump[T_FIELD_RAWTENSOR]);
            rawTensorDict[rawTensor->rawmagic] = rawTensor;
        }
    }

    std::unordered_map<int, std::shared_ptr<LogicalTensor>> tensorDict;
    LoadTensorJson(func, funcDump, rawTensorDict, tensorDict);
    func->opSeed_ = funcDump["_opseed"].get<int>();
    func->magicSeed_ = funcDump["_magicseed"].get<int>();
    int rawid = funcDump["_rawid"].get<int>();
    IdGen<IdType::RAW_TENSOR>::Inst().SetId(rawid);
    int funcid = funcDump["_funcid"].get<int>();
    IdGen<IdType::FUNCTION>::Inst().SetId(funcid);
    func->paramConfigs_.l1ReuseNum = funcDump["_l1_reuse_num"].get<int>();
    func->paramConfigs_.cubeNBufferNum = funcDump["_cube_nbuffer_num"].get<int>();
    func->paramConfigs_.sgCycleUpperBound = funcDump["_sg_cycle_upperbound"].get<int>();
    func->paramConfigs_.sgCycleLowerBound = funcDump["_sg_cycle_lowerbound"].get<int>();
    func->paramConfigs_.sgParallelNum = funcDump["_sg_parallel_num"].get<int>();
    func->paramConfigs_.sgCopyInThreshold = funcDump["_sg_copyin_threshold"].get<int>();
    func->paramConfigs_.dbType = funcDump["_dbtype"].get<int>();
    func->paramConfigs_.NbufferNum = funcDump["_nbuffer_num"].get<int>();
    auto subGraphCount = funcDump["_total_subgraph_count"].get<size_t>();
    func->SetTotalSubGraphCount(subGraphCount);
    func->paramConfigs_.OoOPreScheduleMethodDefault = funcDump["_ooo_preschedule_method_default"].get<std::string>();

    std::vector<std::vector<int>> incastSlot;
    for (auto &iDump : funcDump["incasts"]) {
        std::vector<int> iSlot;
        if (iDump[0].is_number()) {
            for (auto &slot : iDump[1]) {
                iSlot.push_back(slot.get<int>());
            }
            incastSlot.push_back(iSlot);
        }
    }
    std::vector<std::vector<int>> outcastSlot;
    for (auto &oDump : funcDump["outcasts"]) {
        std::vector<int> oSlot;
        if (oDump[0].is_number()) {
            for (auto &slot : oDump[1]) {
                oSlot.push_back(slot.get<int>());
            }
            outcastSlot.push_back(oSlot);
        }
    }
    IncastOutcastSlot ioSlot;
    ioSlot.incastSlot = incastSlot;
    ioSlot.outcastSlot = outcastSlot;
    std::shared_ptr<TensorSlotScope> tensorSlotScope = std::make_shared<TensorSlotScope>(func.get());
    tensorSlotScope->ioslot = ioSlot;
    func->slotScope_ = tensorSlotScope;

    func->ComputeHashOrderless();
    func->functionHash_ = std::stoull(funcDump["hash"].get<std::string>());

    std::vector<std::string> semanticLabelData = funcDump["semantic_label"].get<std::vector<std::string>>();
    func->semanticLabels_.insert(semanticLabelData.begin(), semanticLabelData.end());

    if (funcDump.count("leaf_func_attr") != 0 && funcDump["leaf_func_attr"].count("coretype") != 0) {
        std::shared_ptr<LeafFuncAttribute> attr = std::make_shared<LeafFuncAttribute>();
        attr->coreType = static_cast<CoreType>(funcDump["leaf_func_attr"]["coretype"].get<int>());
        func->SetLeafFuncAttribute(attr);
    }

    if (funcDump.count("root_func_magic") != 0) {
        func->rootFunc_ = belongTo.GetFunctionByMagic(funcDump["root_func_magic"].get<int>()).get();
    }
    if (funcDump.count("programs") != 0) {
        uint64_t index = 0;
        for (auto &programMagic : funcDump["programs"]) {
            func->programs_.emplace(std::make_pair(index++, belongTo.GetFunctionByMagic(programMagic.get<int>()).get()));
        }
    }

    if (funcDump.count("topo") != 0) {
        func->topoInfo_.LoadJson(funcDump["topo"]);
    }

    if (funcDump.count("subfunc_param") != 0) {
        func->parameter_.FromJson(funcDump["subfunc_param"]);
    }

    if (funcDump.count("aic_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AIC, funcDump["aic_ready_subgraph_ids"].get<std::vector<int>>());
    }
    if (funcDump.count("aiv_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AIV, funcDump["aiv_ready_subgraph_ids"].get<std::vector<int>>());
    }

    if (funcDump.count("aicpu_ready_subgraph_ids") != 0) {
        func->SetReadySubGraphIds(CoreType::AICPU, funcDump["aicpu_ready_subgraph_ids"].get<std::vector<int>>());
    }

    if (funcDump.count("dynamic") != 0) {
        auto iterName = funcDump["dynamic"]["itername"];
        auto beginJson = funcDump["dynamic"]["begin"];
        SymbolicScalar begin = LoadSymbolicScalar(beginJson);
        auto endJson = funcDump["dynamic"]["end"];
        SymbolicScalar end = LoadSymbolicScalar(endJson);
        auto stepJson = funcDump["dynamic"]["step"];
        SymbolicScalar step = LoadSymbolicScalar(stepJson);
        LoopRange range(begin, end, step);
        auto originalBeginJson = funcDump["dynamic"]["originalBegin"];
        SymbolicScalar originalBegin = LoadSymbolicScalar(originalBeginJson);
        auto originalEndJson = funcDump["dynamic"]["originalEnd"];
        SymbolicScalar originalEnd = LoadSymbolicScalar(originalEndJson);
        LoopRange originalRange(originalBegin, originalEnd);
        auto attr = std::make_shared<DynloopFunctionAttribute>(iterName, range, originalRange);
        attr->unrollTimes = funcDump["dynamic"]["unrollTimes"];
        auto dynFuncDump = funcDump["dynamic"];
        if (dynFuncDump.count("paths") != 0) {
            std::vector<DynloopFunctionPath> pathList;
            auto pathsJson = funcDump["dynamic"]["paths"];
            for (auto &pathJson : pathsJson) {
                Function *root = func.get();
                std::vector<DynloopFunctionPathCondition> pathCondList;
                auto callOpMagic = pathJson[0];
                Operation *callop;
                for (auto &op : func->Operations().DuplicatedOpList()) {
                    if (op->GetOpMagic() == callOpMagic) {
                        callop = op;
                        break;
                    }
                }
                for (auto &pathCondJson : pathJson[1]) {
                    bool isSat = static_cast<bool>(pathCondJson[1]);
                    SymbolicScalar cond = LoadSymbolicScalar(pathCondJson[0]);
                    DynloopFunctionPathCondition pathCond;
                    pathCond.isSat_ = isSat;
                    pathCond.cond_ = cond;
                    pathCondList.push_back(pathCond);
                }
                DynloopFunctionPath path(root, pathCondList, callop);
                pathList.push_back(path);
            }
            attr->pathList = pathList;
        }
        func->SetDynloopAttribute(attr);
    }
    func->RefreshOpPosition();
    return func;
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

    opImmList = copyAttr->GetFromDynValidShape();
    MaybeNormalizeValue(RUNTIME_COA_GetValidShape, operandCoaList, operandCoaIndex, opImmList, coaIndexBase, valueToIndex);
    copyAttr->SetFromDynValidShape(opImmList);

    return operandCoaList;
}

static std::vector<SymbolicScalar> NormalizeTensor(LogicalTensorPtr operand, int coaIndexBase) {
    auto offset = OpImmediate::Specified(operand->GetOffset());
    auto dynOffset = OpImmediate::Specified(operand->GetDynOffset());
    auto shape = OpImmediate::Specified(operand->GetShape());
    auto rawshape = OpImmediate::Specified(operand->GetRawTensor()->GetRawShape());
    auto dynRawshape = OpImmediate::Specified(operand->GetRawTensor()->GetDynRawShape());
    auto dynValidShape = OpImmediate::Specified(operand->GetDynValidShape());

    int dim = shape.size();
    int operandCoaIndex = COA_INDEX_DIM_BASE;
    int coaIndex = coaIndexBase + COA_INDEX_DIM_BASE;
    std::vector<SymbolicScalar> operandCoaList(COA_INDEX_DIM_BASE + dim * COA_INDEX_TYPE_COUNT, 0);

    if (dynOffset.size()) {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, dynOffset, coaIndex, false);
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, offset, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, shape, coaIndex, false);
    operandCoaIndex += dim;
    coaIndex += dim;

    if (dynRawshape.size()) {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, dynRawshape, coaIndex, false);
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, rawshape, coaIndex, false);
    }
    
    operandCoaIndex += dim;
    coaIndex += dim;

    if (dynValidShape.size()) {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, dynValidShape, coaIndex, false);
    } else {
        OpImmediate::NormalizeValue(operandCoaList, operandCoaIndex, shape, coaIndex, false);
    }
    operandCoaIndex += dim;
    coaIndex += dim;

    return operandCoaList;
}

std::vector<std::vector<SymbolicScalar>> Function::NormalizeCoa(
    std::vector<int> &iOffset, std::vector<int> &oOffset) {
    std::unordered_map<int, Operation *> opmagicToOp;
    std::vector<std::pair<Operation*, int>> extraOutcasts;

    opmagicToOp.reserve(operations_.size());
    for (auto &op : operations_) {
        opmagicToOp[op->GetOpMagic()] = op.get();
        /* The valid-shape of following OP could not be deduced
           should be normalized also */
        if (op->GetOpcode() == Opcode::OP_VEC_DUP ||
            op->GetOpcode() == Opcode::OP_RESHAPE ||
            op->GetOpcode() == Opcode::OP_EXPAND) {
            extraOutcasts.emplace_back(op.get(), 0);
        }
    }

    int coaIndex = COA_INDEX_BASE;
    std::vector<std::vector<SymbolicScalar>> coaLists;
    bool valueToIndex = parent_->GetFunctionType() == FunctionType::DYNAMIC_LOOP_PATH;
    coaLists.reserve(incastPosition.size() + outcastPosition.size() + extraOutcasts.size());
    iOffset.reserve(incastPosition.size());
    SymbolicScalar getParamOffset = SymbolicScalar(AddRuntimeCoaPrefix("GET_PARAM_OFFSET"));
    for (auto [opmagic, k] : incastPosition) {
        auto op = opmagicToOp[opmagic];
        std::vector<SymbolicScalar> operandCoaList;
        if (IsCopyIn(op->GetOpcode()) && k == 0) {
            operandCoaList = NormalizeCopyIn(op, coaIndex, valueToIndex);
            op->SetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_coaIndex", coaIndex);
        } else {
            operandCoaList = NormalizeTensor(op->GetIOperands()[k], coaIndex);
        }
        op->SetIOpAttrOffset(k, coaIndex);
        iOffset.emplace_back(coaIndex);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }

    oOffset.reserve(outcastPosition.size() + extraOutcasts.size());
    for (auto [opmagic, k] : outcastPosition) {
        auto op = opmagicToOp[opmagic];
        std::vector<SymbolicScalar> operandCoaList;
        if (IsCopyOut(op->GetOpcode()) && k == 0) {
            operandCoaList = NormalizeCopyOut(op, coaIndex, valueToIndex);
            op->SetAttr<int>(OP_EMUOP_PREFIX + "GetTensorData_coaIndex", coaIndex);
        } else {
            operandCoaList = NormalizeTensor(op->GetOOperands()[k], coaIndex);
        }
        op->SetOOpAttrOffset(k, coaIndex);
        oOffset.emplace_back(coaIndex);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }

    for (auto [op, k]: extraOutcasts) {
        if (op->GetOOpAttrOffset(0) != -1)
            continue;
        auto operandCoaList = NormalizeTensor(op->GetOOperands()[k], coaIndex);
        op->SetOOpAttrOffset(k, coaIndex);
        oOffset.emplace_back(coaIndex);
        coaIndex += operandCoaList.size();
        coaLists.emplace_back(std::move(operandCoaList));
    }

    for (auto &op : operations_) {
        if (op->GetOpcode() == Opcode::OP_VEC_DUP) {
            if (op->HasAttr(OpAttributeKey::dynScalar)) {
                SymbolicScalar dynScalar = op->GetSymbolicScalarAttribute(OpAttributeKey::dynScalar);
                std::vector<SymbolicScalar> valueCoaList;
                MaybeNormalizeValue(valueCoaList, dynScalar, coaIndex, valueToIndex);
                op->SetAttribute(OpAttributeKey::dynScalar, dynScalar);
                coaLists.emplace_back(valueCoaList);
                coaIndex += 1;
            }
        }
    }

    return coaLists;
}

void Function::DumpTopoFile(const std::string &fileName) const
{
    Json totalTopoJson;
    for (const auto &topo : topoInfo_.GetTopology()) {
        Json sJson;
        sJson["taskId"] = topo.esgId;
        sJson["successors"] = Json::array();
        for (const auto &successor : topo.outGraph) {
            sJson["successors"].push_back(successor);
        }
        int id = operations_[topo.esgId]->GetProgramId();
        if (static_cast<size_t>(id) >= calleeMagicNameList_.size()) {
            continue;
        }
        sJson["funcName"] = calleeMagicNameList_[id];
        sJson["semanticLabel"] = operations_[topo.esgId]->GetSemanticLabelsStr();
        totalTopoJson.push_back(sJson);
    }
    std::ofstream ofs(fileName);
    ofs << totalTopoJson.dump(1) << std::endl;
    ofs.close();
}


std::string Function::DumpSSATitle() const {
    std::stringstream ss;
    ss << GetMagicName() << "[" << functionMagic_ << "]"
       << " " << GetFunctionHash()
       << " " << GetFunctionTypeNameDict().Find(GetFunctionType())
       << " " << GetGraphTypeNameDict().Find(GetGraphType());
    return ss.str();
}

std::string Function::DumpSSARawTensor(int indent) const {
    std::string prefix(indent, ' ');
    std::unordered_set<int> dumpedRawTensor;
    auto dumped = [&dumpedRawTensor](const std::shared_ptr<LogicalTensor> &tensor) -> bool {
        if (dumpedRawTensor.count(tensor->GetRawTensor()->GetRawMagic()) == 0) {
            dumpedRawTensor.insert(tensor->GetRawTensor()->GetRawMagic());
            return false;
        } else {
            return true;
        }
    };
    std::stringstream ss;
    int rawIndex = 0;
    for (size_t i = 0; i < inCasts_.size(); ++i) {
        if (!dumped(inCasts_[i])) {
            ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << rawIndex++ << "] "
               << inCasts_[i]->GetRawTensor()->DumpSSA() << "\n";
        }
    }
    for (size_t i = 0; i < outCasts_.size(); ++i) {
        if (!dumped(outCasts_[i])) {
            ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << rawIndex++ << "] "
               << outCasts_[i]->GetRawTensor()->DumpSSA() << "\n";
        }
    }
    for (size_t i = 0; i < operations_.size(); ++i) {
        for (auto &input : operations_[i]->GetIOperands()) {
            if (!dumped(input)) {
                ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << rawIndex++ << "] "
                   << input->GetRawTensor()->DumpSSA() << "\n";
            }
        }
        for (auto &output : operations_[i]->GetOOperands()) {
            if (!dumped(output)) {
                ss << prefix << "RAWTENSOR[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << rawIndex++ << "] "
                   << output->GetRawTensor()->DumpSSA() << "\n";
            }
        }
    }
    return ss.str();
}
std::string Function::DumpSSAIncast(int indent) const {
    std::string prefix(indent, ' ');
    std::stringstream ss;
    for (size_t i = 0; i < inCasts_.size(); ++i) {
        ss << prefix << "INCAST[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << i << "]  "
           << inCasts_[i]->DumpSSA(false, false, true);
        if (slotScope_ && i < slotScope_->ioslot.incastSlot.size()) {
            auto &incastSlotList = slotScope_->ioslot.incastSlot[i];
            ss << " fromSlot[";
            for (size_t k = 0; k < incastSlotList.size(); k++) {
                if (k != 0) {
                    ss << ", ";
                }
                ss << incastSlotList[k];
            }
            ss << "]";
        }
        ss << "\n";
    }
    return ss.str();
}
std::string Function::DumpSSAOutcast(int indent) const {
    std::string prefix(indent, ' ');
    std::stringstream ss;
    for (size_t i = 0; i < outCasts_.size(); ++i) {
        ss << prefix << "OUTCAST[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << i << "]  "
           << outCasts_[i]->DumpSSA(false, false, true);
        if (slotScope_ && i < slotScope_->ioslot.outcastSlot.size()) {
            auto &outcastSlotList = slotScope_->ioslot.outcastSlot[i];
            ss << " toSlot[";
            for (size_t k = 0; k < outcastSlotList.size(); k++) {
                if (k != 0) {
                    ss << ", ";
                }
                ss << outcastSlotList[k];
            }
            ss << "]";
        }
        ss << "\n";
    }
    return ss.str();
}

std::string Function::DumpSSAAttribute(int indent) const {
    std::string prefix(indent, ' ');
    std::stringstream ss;
    if (IsDynloop()) {
        auto attr = GetDynloopAttribute();
        ss << prefix << "LOOP SYMBOL " << attr->iterSymbolName << "\n";
        ss << prefix << "LOOP BEGIN  " << attr->Begin().Dump() << "\n";
        ss << prefix << "LOOP END    " << attr->End().Dump() << "\n";
        ss << prefix << "LOOP STEP   " << attr->Step().Dump() << "\n";
    }
    return ss.str();
}

constexpr int INDENT_TWO = 2;

std::string Function::DumpSSA() const {
    std::stringstream ss;
    ss << "\n-------------\n";
    ss << "Function " << DumpSSATitle() << " {\n";
    ss << DumpSSARawTensor(INDENT_TWO) << "\n";
    ss << DumpSSAIncast(INDENT_TWO) << "\n";
    ss << DumpSSAOutcast(INDENT_TWO) << "\n";
    ss << DumpSSAAttribute(INDENT_TWO) << "\n";
    for (size_t i = 0; i < operations_.size(); ++i) {
        ss << PREFIX << operations_[i]->DumpSSA(); // Operation dump
    }
    ss << "}\n";
    return ss.str();
}

std::string Function::DumpASM() const {
    std::stringstream ss;
    ss << "\n-------------\n";
    ss << "Function ";
    ss << funcMagicName_ << "[";
    ss << functionMagic_ << "] \n{\n";

    // Serialize input tensors (inCasts_)
    for (size_t i = 0; i < inCasts_.size(); ++i) {
        ss << "INCAST[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << i << "]  " << inCasts_[i]->DumpASM(false) << "\n";
    }
    ss << "\n";

    // Serialize output tensors (outCasts_)
    for (size_t i = 0; i < outCasts_.size(); ++i) {
        ss << "OUTCAST[" << std::setw(SPACE_NUM_THREE) << std::setfill(' ') << i << "]  " << outCasts_[i]->DumpASM(false) << "\n";
    }
    ss << "\n\n";

    // Serialize operations_
    for (size_t i = 0; i < operations_.size(); ++i) {
        ss << operations_[i]->DumpASM(); // Operation dump

        // Add producer information for input operands
        for (const auto &operand : operations_[i]->iOperand) {
            if (!operand->GetProducers().empty()) {
                ss << "\n                         ";
                ss << "    Input Tensor[" << operand->magic << "] Producers: {";
                for (const auto &producer : operand->GetProducers()) {
                    ss << "(" << producer->BelongTo()->GetFuncMagic() << "," << producer->GetOpMagic() << "), ";
                }
                ss.seekp(LAST_TWO, ss.cur); // Remove last comma and space
                ss << "}";
            }
        }

        // Add consumer information for output operands
        for (const auto &operand : operations_[i]->oOperand) {
            ss << "                         ";
            if (!operand->GetConsumers().empty()) {
                ss << "    Output Tensor[" << operand->magic << "] Consumers: {";
                for (const auto &consumer : operand->GetConsumers()) {
                    ss << "(" << consumer->BelongTo()->GetFuncMagic() << "," << consumer->GetOpMagic() << "), ";
                }
                ss.seekp(LAST_TWO, ss.cur);
                ss << "}\n";
            }
            if (operations_[i]->oOperand.size() == 0) {
                ss << "\n";
            }
        }
    }
    ss << "\n}\n";

    return ss.str();
}

std::string Function::Dump() const {
    if (config::GetPlatformConfig("USE_SSA", true)) {
        return DumpSSA();
    } else {
        return DumpASM();
    }
}

void Function::DumpFile(const std::string &filePath) const {
    std::ofstream fout(filePath);
    fout << Dump();
    fout.close();
}

void Function::UpdateOperandBeforeRemoveOp(Operation &op, const bool keepOutTensor) {
    // relink, replace input of following op with the input of current op
    if (!op.GetIOperands().empty() && !op.GetOOperands().empty()) {
        LogicalTensorPtr inputTensor = op.GetIOperands().at(0);
        LogicalTensorPtr outputTensor = op.GetOOperands().at(0);
        bool isOutCast = std::find(outCasts_.begin(), outCasts_.end(), outputTensor) != outCasts_.end();
        if (isOutCast || keepOutTensor) {
            outputTensor->RemoveProducer(op);
            for (auto &producer : inputTensor->GetProducers()) {
                outputTensor->AddProducer(*producer);
                producer->ReplaceOutputOperand(inputTensor, outputTensor);
            }
            inputTensor->GetProducers().clear();
        } else {
            inputTensor->RemoveConsumer(op);
            for (const auto &consumer : outputTensor->GetConsumers()) {
                inputTensor->AddConsumer(consumer);
                consumer->ReplaceInputOperand(outputTensor, inputTensor);
            }
            outputTensor->GetConsumers().clear();
        }
    }
}

/**
 * @brief handle input and output control edges
 * all input ctrl edges shall be moved to output ops of current op
 * all output ctrl edges shall be moved to input ops of current op
 * @param op
 */
void Function::HandleControlOps(Operation &op, std::vector<Operation *> &toRemoveOps) const {
    const auto &inputCtrlOpSet = op.GetInCtrlOperations();
    if (!inputCtrlOpSet.empty()) {
        auto outputOps = GetAllOutputOperations(op);
        for (auto peerCtrlOp : inputCtrlOpSet) {
            if (peerCtrlOp == nullptr) {
                continue;
            }
            if (peerCtrlOp->OnlyHasCtrlEdgeToOp(op)) {
                op.RemoveInCtrlOperation(*peerCtrlOp);
                toRemoveOps.push_back(peerCtrlOp);
            } else {
                for (auto &outputOp : outputOps) {
                    outputOp->AddInCtrlOperation(*peerCtrlOp);
                }
            }
        }
        op.ClearInCtrlOperations();
    }
    const auto &outputCtrlOpSet = op.GetOutCtrlOperations();
    if (!outputCtrlOpSet.empty()) {
        auto inputOps = GetAllInputOperations(op);
        for (auto &inputOp : inputOps) {
            for (auto outCtrlOp : outputCtrlOpSet) {
                inputOp->AddOutCtrlOperation(*outCtrlOp);
            }
        }
        op.ClearOutCtrlOperations();
    }
}

Operation *Function::GetOpByOpMagic(const int opMagic) const {
    for (auto op : operations_) {
        if (op->GetOpMagic() == opMagic) {
            return op.get();
        }
    }
    return nullptr;
}

bool Function::TensorReuse(const LogicalTensorPtr &dstTensor, const LogicalTensorPtr &srcTensor) {
    if (dstTensor == nullptr || srcTensor == nullptr) {
        return false;
    }
    if (dstTensor->Datatype() != srcTensor->Datatype() ||
        dstTensor->tensor->GetRawShapeSize() != srcTensor->tensor->GetRawShapeSize()) {
        ASLOGI("Data type or raw shape size of src and dst tensor is not same.");
        return false;
    }

    if (dstTensor->tensor->rawshape == srcTensor->tensor->rawshape) {
        dstTensor->tensor = srcTensor->tensor;
    } else {
        dstTensor->tensor->actualRawmagic = srcTensor->tensor->actualRawmagic == -1 ?
                                                 srcTensor->tensor->rawmagic :
                                                 srcTensor->tensor->actualRawmagic;
    }
    return true;
}

bool Function::IsFromInCast(const std::shared_ptr<LogicalTensor> &tensor) {
    for (auto &t : inCasts_) {
        if (t->GetRawMagic() == tensor->GetRawMagic()) {
            return true;
        }
    }
    return false;
}

bool Function::IsFromOutCast(const std::shared_ptr<LogicalTensor> &tensor) {
    for (auto &t : outCasts_) {
        if (t->GetRawMagic() == tensor->GetRawMagic()) {
            return true;
        }
    }
    return false;
}

bool Function::IsFromDummyOutCast(int rawMagic) {
    for (auto &t : outCasts_) {
        if (t->tensor->rawmagic == rawMagic) {
            return true;
        }
    }
    return false;
}

int Function::GetIncastIndex(std::shared_ptr<LogicalTensor> &tensor) const {
    for (size_t i = 0; i < inCasts_.size(); i++) {
        if (inCasts_[i] == tensor) {
            return (int)i;
        }
    }
    return INVALID_IOINDEX;
}

int Function::GetOutcastIndex(std::shared_ptr<LogicalTensor> &tensor) const {
    for (size_t i = 0; i < outCasts_.size(); i++) {
        if (outCasts_[i] == tensor) {
            return (int)i;
        }
    }
    return INVALID_IOINDEX;
}

void Function::ResetOperations() {
    operations_.clear();
    opPosition_.clear();
    sorted_ = false;
    tensorMap_.Reset();
    for (auto &i : inCasts_) {
        tensorMap_.Insert(i);
    }
    for (auto &t : inCasts_) {
        t->GetConsumers().clear();
    }
    for (auto &t : outCasts_) {
        t->GetProducers().clear();
    }
}

std::set<Operation *, LogicalTensor::CompareOp> Function::FindConsumers(const Operation &op) const {
    std::set<Operation *, LogicalTensor::CompareOp> consumers;
    for (const auto &output : op.oOperand) {
        for (auto &consumer : output->GetConsumers()) {
            if (consumer->BelongTo() == this) {
                consumers.emplace(consumer);
            }
        }
    }
    return consumers;
}

std::set<Operation *, LogicalTensor::CompareOp> Function::FindProducers(const Operation &op) const {
    std::set<Operation *, LogicalTensor::CompareOp> producers;
    for (const auto &input : op.iOperand) {
        for (auto &producer : input->GetProducers()) {
            if (producer->BelongTo() == this) {
                producers.emplace(producer);
            }
        }
    }
    return producers;
}

std::vector<OriArgInfo> Function::GetOpOriginArgsInfo() {
    std::map<int, OriArgInfo> args;
    int maxSubscript = 0;
    for (const auto &incast : inCasts_) {
        auto subscript = incast->GetRawTensor()->GetTensorInfo().subscript;
        if (subscript == -1) {
            continue;
        }
        maxSubscript = std::max(maxSubscript, subscript);
        OriArgInfo info{reinterpret_cast<uint64_t>(incast->GetRawTensor()->GetRawDataPtr()), incast->MemorySize(),
            incast->GetCachePolicy(CachePolicy::PREFETCH)};
        if (args.count(subscript) > 0) {
            ASSERT(args.at(subscript) == info);
        } else {
            args.emplace(subscript, info);
        }
    }
    for (const auto &outcast : outCasts_) {
        auto subscript = outcast->GetRawTensor()->GetTensorInfo().subscript;
        if (subscript == -1) {
            continue;
        }
        maxSubscript = std::max(maxSubscript, subscript);
        OriArgInfo info{reinterpret_cast<uint64_t>(outcast->GetRawTensor()->GetRawDataPtr()), outcast->MemorySize(),
            outcast->GetCachePolicy(CachePolicy::PREFETCH)};
        if (args.count(subscript) > 0) {
            ASSERT(args.at(subscript) == info);
        } else {
            args.emplace(subscript, info);
        }
    }

    std::vector<OriArgInfo> argsInfo(maxSubscript + 1);
    for (int i = 0; i <= maxSubscript; i++) {
        if (args.count(i) > 0) {
            argsInfo[i] = args.at(i);
        } else {
            argsInfo[i] = OriArgInfo{0, 0, false};
        }
    }
    return argsInfo;
}

void Function::OpValidCheck(Operation &op) const {
    std::unordered_set<const Operation *> opMap;
    std::unordered_set<std::shared_ptr<LogicalTensor>> incasts(GetIncast().begin(), GetIncast().end());
    if (SPECIAL_OPCODE_SET.count(op.GetOpcode()) != 0) {
        if (op.GetOpcode() == Opcode::OP_VIEW) {
            ASSERT(op.GetIOperands().size() == 1);
            ASSERT(op.GetOOperands().size() <= 1);
            auto opAttr = dynamic_cast<ViewOpAttribute *>(op.GetOpAttribute().get());
            ASSERT(opAttr != nullptr);
            ASSERT(op.GetIOperands()[0]->GetOffset().size() == opAttr->GetFromOffset().size());
            if (!op.GetOOperands().empty()) {
                ASSERT(op.GetOOperands()[0]->GetOffset().size() == opAttr->GetFromOffset().size());
            }
        }
        if (op.GetOpcode() == Opcode::OP_ASSEMBLE) {
            ASSERT(op.GetIOperands().size() == 1);
            ASSERT(op.GetOOperands().size() <= 1);
            auto opAttr = dynamic_cast<AssembleOpAttribute *>(op.GetOpAttribute().get());
            ASSERT(opAttr != nullptr);
            if (!op.GetIOperands().empty()) {
                ASSERT(op.GetIOperands()[0]->GetOffset().size() == opAttr->GetToOffset().size());
            }
            ASSERT(op.GetOOperands()[0]->GetOffset().size() == opAttr->GetToOffset().size());
        }
    } else {
        ASSERT(op.GetOpAttribute() == nullptr);
    }

    ASSERT(op.GetOpMagic() >= 0 && op.GetOpMagic() < opSeed_)
            << "function opSeed_ is: " << opSeed_ << ", opmagic is: " << op.GetOpMagic();
    if (!op.IsCall()) { // call 允许多输出，其余操作目前不允许
        ASSERT(op.GetOOperands().size() <= 1) << "size: " << op.GetOOperands().size();
    }
    for (auto &oOperand : op.GetOOperands()) {
        ASSERT(oOperand->GetShape().size() == oOperand->GetOffset().size());
        ASSERT(&oOperand->BelongFunction() == this);
        auto tmp = GetTensorMap().GetTensorByMagic(oOperand->magic);
        ASSERT(tmp == oOperand);
        ASSERT(oOperand->HasProducer(op)) << "opmagic: " << op.GetOpMagic() << "GetOOperands():" << oOperand->magic;
    }
    for (auto &iOperand : op.GetIOperands()) {
        ASSERT(iOperand->GetShape().size() == iOperand->GetOffset().size());
        ASSERT(&iOperand->BelongFunction() == this);
        if (!iOperand->GetProducers().empty() || incasts.count(iOperand) != 0) {
            auto tmp = GetTensorMap().GetTensorByMagic(iOperand->magic);
            ASSERT(tmp == iOperand);
        }

        ASSERT(iOperand->HasConsumer(op));
        for (const auto &producer : iOperand->GetProducers()) {
            ASSERT(producer->BelongTo() == this);
            ASSERT(producer->GetOpMagic() >= 0 && producer->GetOpMagic() < opSeed_)
                    << "function opSeed_ is: " << opSeed_ << ", producer in tensor(" << iOperand->magic << ","
                    << iOperand->tensor->rawmagic << ") is: " << producer;
            if (producer->IsDeleted()) {
                continue;
            }
            ASSERT(opMap.find(producer) != opMap.end());
        }
    }

    ASSERT(opMap.count(&op) == 0);
    opMap.emplace(&op);
}

void Function::ValidCheck() const {
    int opMagic = -1000000;
    for (auto &op : const_cast<Function &>(*this).Operations()) {
        opMagic = std::max(opMagic, op.GetOpMagic());
    }
    ASSERT(opMagic + 1 <= opSeed_);

    TensorMagicCheck();

    std::unordered_map<std::shared_ptr<LogicalTensor>, std::vector<Operation *>> used;
    for (auto &op : const_cast<Function &>(*this).Operations()) {
        if (op.IsDeleted()) {
            continue;
        }
        for (const auto &operand : op.GetOOperands()) {
            if (used.count(operand) > 0) {
                for (auto innerOp : used.at(operand)) {
                    ASSERT(innerOp->ComputeHash() != op.ComputeHash());
                }
            }
            used[operand].emplace_back(&op);
        }
    }

    for (auto &op : const_cast<Function &>(*this).Operations()) {
        if (op.IsDeleted()) {
            continue;
        }
        OpValidCheck(op);
    }
}

std::shared_ptr<OpAttribute> Function::CreateCallOpAttribute(const std::vector<std::vector<SymbolicScalar>> &argList) {
    FunctionHash hash;
    if (rootFunc_ != nullptr) {
        /* has rootFunc, then current function is cutted */
        hash = rootFunc_->ComputeHash();
    } else {
        hash = ComputeHash();
    }
    auto opAttribute = std::make_shared<CallOpAttribute>(hash, argList, GetMagicName());
    return opAttribute;
}

std::shared_ptr<LogicalTensor> Function::ConnectWithOverlap(std::shared_ptr<LogicalTensor> iOperand) {
    auto matches = GetTensorMap().Find(iOperand);
    if (matches.empty()) {
        return iOperand;
    }
    auto overlapStatus = CalcOverlap(iOperand, matches);
    ASSERT(!matches.empty());

    std::vector<std::vector<int>> offsetOfOverlaps;
    std::vector<std::shared_ptr<LogicalTensor>> needAddConsumer;
    sort(matches.begin(), matches.end(), [](const auto &a, const auto &b) -> bool { return a->offset < b->offset; });

    std::vector<int> minimumOffsets = matches.front()->offset;

    for (auto &m : matches) {
        offsetOfOverlaps.emplace_back(m->offset);
        for (size_t i = 0; i < minimumOffsets.size(); i++) {
            minimumOffsets[i] = std::min(minimumOffsets[i], m->offset[i]);
        }
    }
    for (auto &offsetOfOverlap : offsetOfOverlaps) {
        for (size_t i = 0; i < offsetOfOverlap.size(); i++) {
            offsetOfOverlap[i] -= minimumOffsets[i];
        }
    }

    switch (overlapStatus) {
        case OverlapStatus::PERFECTLY_MATCH_WITH_ALL: {
            auto assembleResult = std::make_shared<LogicalTensor>(*this, iOperand->Datatype(), iOperand->shape,
                iOperand->GetDynValidShape(), "Assemble_" + matches[0]->Symbol(), iOperand->nodetype,
                iOperand->tensorfmt);
            ASSERT(assembleResult->GetProducers().empty());
            for (size_t i = 0; i < matches.size(); i++) {
                auto &assembleOp = AddRawOperation(Opcode::OP_ASSEMBLE, {matches[i]}, {assembleResult});
                assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offsetOfOverlaps[i]));
            }
            return assembleResult;
        }
        case OverlapStatus::PERFECTLY_MATCH: {
            // IF there is an existing tensor that fully matches GetIOperands(), change GetIOperands() to
            // the existing tensor
            return matches.front();
        }
        case OverlapStatus::BE_COVERED: {
            auto viewResult = std::make_shared<LogicalTensor>(*this, matches.front()->tensor->datatype, iOperand->shape,
                "View_" + matches.front()->tensor->symbol, matches.front()->nodetype, iOperand->tensorfmt);
            auto &viewOp = AddRawOperation(Opcode::OP_VIEW, {matches.front()}, {viewResult});
            viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(
                iOperand->GetOffset(), iOperand->GetDynOffset(), iOperand->GetDynValidShape()));
            if (!iOperand->GetDynValidShape().empty()) {
                viewResult->UpdateDynValidShape(iOperand->GetDynValidShape());
            }
            return viewResult;
        }
        case OverlapStatus::BE_COVERED_BY_ALL: {
            std::vector<int> minimumOffset;
            std::vector<int> maximumShape;
            CalcShapeAndOffsetOfGroup(matches, minimumOffset, maximumShape);

            auto assembleResult = std::make_shared<LogicalTensor>(
                *this, matches[0]->Datatype(), maximumShape, "Assemble_" + matches[0]->Symbol(), iOperand->nodetype, iOperand->tensorfmt);
            ASSERT(assembleResult->GetProducers().empty());
            for (size_t i = 0; i < matches.size(); i++) {
                auto &assembleOp = AddRawOperation(Opcode::OP_ASSEMBLE, {matches[i]}, {assembleResult});
                assembleOp.SetOpAttribute(std::make_shared<AssembleOpAttribute>(offsetOfOverlaps[i]));
            }

            auto viewResult = std::make_shared<LogicalTensor>(*this, assembleResult->Datatype(), iOperand->shape,
                "View_" + assembleResult->Symbol(), assembleResult->nodetype, iOperand->tensorfmt);
            auto &viewOp = AddRawOperation(Opcode::OP_VIEW, {assembleResult}, {viewResult});
            std::vector<int> newOffset = TensorOffset::Sub(iOperand->GetOffset(), minimumOffset);
            std::vector<SymbolicScalar> newDynOffset = TensorOffset::Sub(iOperand->GetDynOffset(), minimumOffset);
            // fill valid shape
            viewOp.SetOpAttribute(std::make_shared<ViewOpAttribute>(newOffset, newDynOffset));
            return viewResult;
        }
        default: ASSERT(false) << "unexpected behavior";
    }

    ASSERT(false);
    return nullptr;
}

DefineProg::DefineProg(const std::string &name) : isRecording_(true) {
    Program::GetInstance().SetName(name);
}
DefineProg::~DefineProg() {
    if (isRecording_) {
        ALOG_INFO("prog.end: name=", Program::GetInstance().Name());
    }
}