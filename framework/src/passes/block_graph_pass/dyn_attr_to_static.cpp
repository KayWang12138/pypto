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
 * \file dyn_attr_to_static.cpp
 * \brief
 */

#include "passes/block_graph_pass/dyn_attr_to_static.h"

namespace npu {
namespace tile_fwk {

Status SToIWrapper(const std::string str, int& result) {
    try {
        result = std::stoi(str);
        return SUCCESS;
    } catch (const std::exception &e) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to convert %s to int, error is %s.", str.c_str(), e.what());
    }
    return FAILED;
}

void DynAttrToStatic::RefSpecifiedValue(std::vector<SymbolicScalar> &oriList,
    std::vector<std::reference_wrapper<SymbolicScalar>> &newList) const
{
    for (auto &value : oriList) {
        newList.push_back(std::reference_wrapper<SymbolicScalar>(value));
    }
}

void DynAttrToStatic::FilterSpecifiedValue(std::vector<OpImmediate> &oriList,
    std::vector<std::reference_wrapper<SymbolicScalar>> &newList) const
{
    for (auto &value : oriList) {
        if (value.IsSpecified()) {
            newList.push_back(std::reference_wrapper<SymbolicScalar>(value.GetSpecifiedValue()));
        }
    }
}

std::vector<std::reference_wrapper<SymbolicScalar>> DynAttrToStatic::GetOpDynamicAttributeList(Operation &op) {
    std::vector<std::reference_wrapper<SymbolicScalar>> dynamicAttributeList;
    auto opcode = op.GetOpcode();
    if(opcode == Opcode::OP_VIEW) {
        auto viewAttr = std::static_pointer_cast<ViewOpAttribute>(op.GetOpAttribute());
        if (viewAttr != nullptr) {
            RefSpecifiedValue(viewAttr->GetFromDynOffset(), dynamicAttributeList);
            RefSpecifiedValue(viewAttr->GetToDynValidShape(), dynamicAttributeList);
        }
        return dynamicAttributeList;
    }

    if(opcode == Opcode::OP_ASSEMBLE) {
        auto assembleAttr = std::static_pointer_cast<AssembleOpAttribute>(op.GetOpAttribute());
        if (assembleAttr != nullptr) {
            RefSpecifiedValue(assembleAttr->GetToDynOffset(), dynamicAttributeList);
            RefSpecifiedValue(assembleAttr->GetFromDynValidShape(), dynamicAttributeList);
        }
        return dynamicAttributeList;
    } 

    const std::set<Opcode> specifiedOps = {Opcode::OP_VEC_DUP, Opcode::OP_EXPAND, Opcode::OP_RESHAPE};
    if (specifiedOps.count(opcode)) {
        auto &attrDict = op.GetAllAttr();
        auto it = attrDict.find(OpAttributeKey::dynScalar);
        if (it != attrDict.end()) {
            auto &value = *npu::tile_fwk::AnyCast<SymbolicScalar>(&it->second);
            dynamicAttributeList.push_back(std::reference_wrapper<SymbolicScalar>(value));
        }
        return dynamicAttributeList;
    }

    if (OpcodeManager::Inst().IsCopyInOrOut(opcode)) {
        auto copyAttr = std::static_pointer_cast<CopyOpAttribute>(op.GetOpAttribute());
        if (copyAttr != nullptr) {
            FilterSpecifiedValue(copyAttr->GetToOffset(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetFromOffset(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetToDynValidShape(), dynamicAttributeList);
            FilterSpecifiedValue(copyAttr->GetFromDynValidShape(), dynamicAttributeList);
        }
    }
    return dynamicAttributeList;
}

Status DynAttrToStatic::GetCallee(const Operation *callop, Function *&callFunc) {
    auto callopAttr = std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute());
    callFunc = Program::GetInstance().GetFunctionByMagicName(callopAttr->GetCalleeMagicName());
    if (callFunc == nullptr) {
        APASS_LOG_ERROR_F(Elements::Operation, "Get callee function %s failed.", callopAttr->GetCalleeMagicName().c_str());
        return FAILED;
    }
    return SUCCESS;
}

Status DynAttrToStatic::BuildLeafToCaller(Function *func) {
    if (func->IsFunctionTypeAndGraphType(
        {FunctionType::DYNAMIC, FunctionType::DYNAMIC_LOOP, FunctionType::DYNAMIC_LOOP_PATH}, GraphType::TENSOR_GRAPH)) {
        for (auto callop : func->GetCallopList()) {
            Function *nextFunc = nullptr;
            if (GetCallee(callop, nextFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, %s[%d] GetCallee failed.",
                    func->GetRawName().c_str(), callop->GetOpcodeStr().c_str(), callop->GetOpMagic());
                return FAILED;
            }
            if (BuildLeafToCaller(nextFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, nextFunc at %s failed",
                    func->GetRawName().c_str(), nextFunc->GetRawName().c_str());
                return FAILED;
            }
        }
        return SUCCESS;
    } else if (func->GetGraphType() == GraphType::TILE_GRAPH) {
        Function *rootFunc = func->GetRootFunction();
        return BuildLeafToCaller(rootFunc);
    } else if (func->GetGraphType() == GraphType::EXECUTE_GRAPH) {
        for (auto callop : func->GetCallopList()) {
            Function *leafFunc = nullptr;
            if (GetCallee(callop, leafFunc) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s, %s[%d] GetCallee failed.",
                    func->GetRawName().c_str(), callop->GetOpcodeStr().c_str(), callop->GetOpMagic());
                return FAILED;
            }
            leaf2Caller[leafFunc].push_back(callop);
        }
        return SUCCESS;
    }
    APASS_LOG_ERROR_F(Elements::Operation, "BuildLeafToCaller at %s entered unexpected function type %d.",
        func->GetRawName().c_str(), static_cast<int>(func->GetFunctionType()));
    return FAILED;
}

Status DynAttrToStatic::BuildNewCoa(
        std::reference_wrapper<SymbolicScalar>& dynScalar,
        std::vector<std::vector<SymbolicScalar>>& callopArglistOneDim)
{
    // 1. 拆解dynScalar到对应的COA表达式
    std::string dynParamExpr = SymbolicExpressionTable::BuildExpression(dynScalar);
    if (dynParamExpr.find(COA_PREFIX) != 1) { // dynParamExpr格式是"(RUNTIME_GET_COA_XXX"
        APASS_LOG_INFO_F(Elements::Operation, "BuildNewCoa skips non-COA dynamic expression.");
        return SUCCESS;
    }
    CoaInfo coaExpr;
    if (coaExpr.ParseCoaString(dynParamExpr) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "BuildNewCoa found unexpected COA expression at dynParamExpr.");
        return FAILED;
    }
    int coaIndex = coaExpr.CalculateCoaIndex();
    
    // 2. 遍历不同caller下的取值，确认是否是常数
    IsConstMetric scalarValue;
    for (auto argList : callopArglistOneDim) {
        auto callopAttr = argList[coaIndex];
        if (!callopAttr.IsImmediate()) {
            scalarValue.MarkNotConst();
        } else {
            scalarValue.UpdateValue(callopAttr.Concrete());
        }
    }

    // 3. 刷新新的COA宏
    APASS_LOG_INFO_F(Elements::Operation, "BuildNewCoa update dynScalar with isConst=%d, value=%d.", scalarValue.isConst, scalarValue.attrValue);
    dynScalar.get() = coaExpr.BuildMaybeConstCoa(scalarValue.isConst, scalarValue.attrValue);
    return SUCCESS;
}

Status DynAttrToStatic::TryRemoveDynAttr(Function* leafFunc, std::vector<Operation*> callList) {
    // 1. 为leafFunc拿到它所有caller的一维的callopArglistOneDim
    std::vector<std::vector<SymbolicScalar>> callopArglistOneDim;
    for (size_t i = 0; i < callList.size(); i++) {
        auto callop = std::static_pointer_cast<CallOpAttribute>(callList[i]->GetOpAttribute());
        callopArglistOneDim.push_back(callop->GetLinearArgList());
    }

    // 2. 依次为leafFunc的所有op拿到所有动态attr，为每个动态attr刷新coa宏
    auto operationViewer = leafFunc->Operations(false);
    for (size_t j = 0; j < operationViewer.size(); j++) {
        auto &op = operationViewer[j];
        std::vector<std::reference_wrapper<SymbolicScalar>> dynScalarList = GetOpDynamicAttributeList(op);
        for (auto dynScalar : dynScalarList) {
            if (BuildNewCoa(dynScalar, callopArglistOneDim) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "TryRemoveDynAttr failed to execute BuildNewCoa for op [%d][%s].",
                    op.GetOpMagic(), op.GetOpcodeStr().c_str());
                return FAILED;
            }
        }
    }

    // 3. 为dynParam的赋值刷新coa宏
    unsigned isSupportUnaligned = config::GetCodeGenOption<bool>(SUPPORT_DYNAMIC_UNALIGNED);
    if (!isSupportUnaligned) {
        return SUCCESS;
    }
    for (const auto &dynParam : leafFunc->GetDynParamTable()) {
        if (dynParam.second.dim.IsValid()) {
            std::reference_wrapper<SymbolicScalar> dynExpr = const_cast<SymbolicScalar&>(dynParam.second.dim);
            if (BuildNewCoa(dynExpr, callopArglistOneDim) != SUCCESS) {
                APASS_LOG_ERROR_F(Elements::Operation, "TryRemoveDynAttr failed to execute BuildNewCoa for dynExpr %s.",
                    SymbolicExpressionTable::BuildExpression(dynExpr).c_str());
                return FAILED;
            }
        }
    }

    return SUCCESS;
}


Status DynAttrToStatic::RunOnFunction(Function &function) {
    APASS_LOG_INFO_F(Elements::Operation, "==============> Start DynAttrToStatic.");
    // 1. 遍历所有rootFunc，找到每个leaf的所有caller，生成leaf2Caller map
    if (BuildLeafToCaller(&function) != SUCCESS) {
        APASS_LOG_ERROR_F(Elements::Operation, "Failed to call BuildLeafToCaller.");
        return FAILED;
    }

    // 2. 遍历leaf2Caller，尝试为每个leaf消除动态attributes
    for (const auto& pair : leaf2Caller) {
        if (TryRemoveDynAttr(pair.first, pair.second) != SUCCESS) {
            APASS_LOG_ERROR_F(Elements::Operation, "Failed to call TryRemoveDynAttr for leafFunc %s.", pair.first->GetRawName().c_str());
            return FAILED;
        }
    }
    
    APASS_LOG_INFO_F(Elements::Operation, "==============> End DynAttrToStatic.");
    return SUCCESS;
}
} // namespace tile_fwk
} // namespace npu