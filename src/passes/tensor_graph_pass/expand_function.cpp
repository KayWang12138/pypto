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
 * \file expand_function.cpp
 * \brief
 */

#include <map>
#include "interface/operation/opcode.h"
#include "interface/function/function.h"
#include "interface/tensor/raw_tensor.h"
#include "interface/tensor/logical_tensor.h"
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/program/program.h"
#include "interface/operation/operation_impl.h"
#include "interface/configs/config_manager.h"
#include "passes/tensor_graph_pass/expand_function.h"
#include "passes/pass_check/expand_function_checker.h"
#include "passes/statistics/tensor_and_tile_graph_statistic.h"

using namespace npu::tile_fwk;

namespace npu::tile_fwk {
namespace {
bool CheckAssembleNeedCopy(Function &function, const std::shared_ptr<Operation> &op) {
    if (op->GetOpcode() != Opcode::OP_ASSEMBLE) {
        return false;
    }
    if (op->oOperand.empty() || op->iOperand.empty()) {
        return false;
    }
    auto assembleOpAttribute = dynamic_cast<AssembleOpAttribute *>(op->GetOpAttribute().get());
    if (assembleOpAttribute == nullptr) {
        return false;
    }
    if (assembleOpAttribute->GetToDynOffset().empty()) {
        return false;
    }
    auto producers = function.FindProducers(*op);
    if (producers.size() != 1) {
        return false;
    }
    if (((*producers.begin()) == nullptr) || ((*producers.begin())->GetOpcode() != Opcode::OP_RESHAPE)) {
        return false;
    }
    for (size_t i = 1; i < op->oOperand[0]->shape.size(); i++) {
        if (op->oOperand[0]->shape[i] != op->iOperand[0]->shape[i]) {
            op->SetAttr("NeedCopy", true);
            ALOG_INFO_F("assemble %d need to check expansion.",  op->GetOpMagic());
            return true;
        }
    }
    return false;
}

Status UpdateIOOperand(const std::vector<OperationPtr> &tensorOperations) {
    for (auto &op : tensorOperations) {
        // clear consumers and producers
        for (auto &iOperand : op->GetIOperands()) {
            if (iOperand == nullptr) {return FAILED;}
            iOperand->GetConsumers().clear();
            iOperand->GetProducers().clear();
        }
        for (auto &oOperand : op->GetOOperands()) {
            if (oOperand == nullptr) {return FAILED;}
            oOperand->GetConsumers().clear();
            oOperand->GetProducers().clear();
        }
    }
    return SUCCESS;
}
}

Status ExpandFunction::PreCheck(Function &function) {
    ExpandFunctionChecker checker;
    return checker.DoPreCheck(function);
}

Status ExpandFunction::PostCheck(Function &function) {
    ExpandFunctionChecker checker;
    return checker.DoPostCheck(function);
}

Status ExpandFunction::RunOnFunction(Function &function) {
    ALOG_INFO_F("===> Start ExpandFunctionPass for function [%s].", function.GetRawName().c_str());
    if (Expandfunction(function) != SUCCESS) {return FAILED;}
    ALOG_INFO_F("===> End ExpandFunctionPass for function [%s].", function.GetRawName().c_str());
    return SUCCESS;
}

Status ExpandFunction::Expandfunction(Function &function) const {
    if (!function.IsGraphType(GraphType::TENSOR_GRAPH)) {
        ALOG_INFO_F("Function is not static tensor graph, skip expanding. function name: %s ", function.GetRawName().c_str());
        return SUCCESS;
    }
    function.expandFunctionAccelerate = true;
    function.SetGraphType(GraphType::TILE_GRAPH);

    std::vector<OperationPtr> tensorOperations;
    auto operationViewer = function.Operations();
    for (size_t i = 0; i < operationViewer.size(); i++) {
        tensorOperations.emplace_back(operationViewer.operations_[i]);
    }

    function.ResetOperations();
    if (UpdateIOOperand(tensorOperations) != SUCCESS) {return FAILED;}

    for (auto &op : tensorOperations) {
        if (op == nullptr) {return FAILED;}
        if (op->GetOpcode() == Opcode::OP_NOP || op->GetOpcode() == Opcode::OP_PRINT) {
            continue;
        }
        bool needCopy = CheckAssembleNeedCopy(function, op);
        if (op->GetOpcode() == Opcode::OP_VIEW || (op->GetOpcode() == Opcode::OP_ASSEMBLE && !needCopy) || op->GetOpcode() == Opcode::OP_PAD) {
            auto &newOp = function.AddOperation(op->GetOpcode(), op->GetIOperands(), op->GetOOperands());
            newOp.SetOpAttribute(op->GetOpAttribute());
            newOp.CopyAttrFrom(*op, OP_EMUOP_PREFIX);
        } else {
            ConfigManager::Instance().SetSemanticLabel(op->GetSemanticLabels()[0]);
            size_t opListPreSize = function.Operations(false).size();
            ExpandOperationInto(function, op->GetTileShape(), op->GetOpcode(), op->GetIOperands(), op->GetOOperands(), *op);
            auto opListPost = function.Operations(false);
            if (op->GetOpcode() == Opcode::OP_ADDS) {
                for (size_t i = opListPreSize; i < opListPost.size(); i++) {
                    auto &newOp = opListPost[i];
                    newOp.CopyAttrFrom(*op, OP_EMUOP_PREFIX);
                }
            }
        }
    }
    function.expandFunctionAccelerate = false;
    return SUCCESS;
}

void ExpandFunction::DoHealthCheckBefore(Function &function, const std::string &folderPath) {
    ALOG_INFO_F("Before ExpandFunction, Health Report: TensorGraph START");
    std::string fileName = GetDumpFilePrefix(function, true);
    HealthCheckTensorGraph(function, folderPath, fileName);
    ALOG_INFO_F("Before ExpandFunction, Health Report: TensorGraph END");
}
} // namespace npu::tile_fwk
