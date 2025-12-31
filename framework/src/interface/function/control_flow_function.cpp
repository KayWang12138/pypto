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
 * \file control_flow_function.cpp
 * \brief
 */

#include "interface/function/control_flow_function.h"

#include "interface/program/program.h"
#include "interface/operation/operation.h"
#include "tilefwk/symbolic_scalar.h"

namespace npu::tile_fwk {
ControlFlowFunction::ControlFlowFunction(const Program &belongTo, const std::string &funcMagicName,
    const std::string &funcRawName, Function *parentFunc)
    : Function(belongTo, funcMagicName, funcRawName, parentFunc),
      dyndevAttr_(nullptr) {
}

bool ControlFlowFunction::IsDynloop() const {
    if(IsFunctionType(FunctionType::DYNAMIC)) return false;
    return GetDynloopAttribute() != nullptr;
}

DyndevFunctionAttribute::ValueDependDesc ControlFlowFunction::LookupValueDepend() {
    struct ValueDependSearcher {
        static void Search(DyndevFunctionAttribute::ValueDependDesc &desc, const SymbolicScalar &attr) {
            std::vector<RawSymbolicScalarPtr> callList = LookupExpressionByOpcode(attr.Raw(), SymbolicOpcode::T_MOP_CALL);
            for (auto &call : callList) {
                auto caller = call->GetExpressionOperandList()[0];
                if (!caller->IsSymbol()) {
                    continue;
                }
                std::string name = caller->GetSymbolName();
                if (CallIsGetInputData(name)) {
                    desc.getInputDataCount++;
                } else if (CallIsGetTensorData(name)) {
                    desc.getTensorDataCount++;
                }
            }
        }
    };

    DyndevFunctionAttribute::ValueDependDesc desc;
    if (GetFunctionType() == FunctionType::DYNAMIC_LOOP) {
        auto loopAttr = GetDynloopAttribute();
        ValueDependSearcher::Search(desc, loopAttr->Begin());
        ValueDependSearcher::Search(desc, loopAttr->End());
        ValueDependSearcher::Search(desc, loopAttr->Step());
        for (auto &path : loopAttr->GetPathList()) {
            for (auto &cond : path.GetPathCondList()) {
                ValueDependSearcher::Search(desc, cond.GetCond());
            }
        }
    } else {
        for (auto &op : Operations(false)) {
            std::vector<std::reference_wrapper<SymbolicScalar>> attrList = op.GetDynamicAttributeList();
            for (auto &attr : attrList) {
                ValueDependSearcher::Search(desc, attr.get());
            }
        }
    }
    return desc;
}

} // namespace npu::tile_fwk

