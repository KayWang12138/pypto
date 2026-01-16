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
 * \file operation.cpp
 * \brief
 */

#include "interface/interpreter/function.h"
#include "interface/utils/log.h"
#include "interface/interpreter/operation.h"

namespace npu::tile_fwk {

static int64_t GetAsParameterCoaIndex(const RawSymbolicScalarPtr &value) {
    if (value->IsExpressionCall("RUNTIME_COA_GET_PARAM_OFFSET")) {
        auto &operands = value->GetExpressionOperandList();
        auto base = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX]->GetImmediateValue();
        auto dimIdx = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX]->GetImmediateValue();
        return base + COA_INDEX_DIM_BASE + dimIdx;
    } else if (value->IsExpressionCall("RUNTIME_COA_GET_PARAM_VALID_SHAPE")) {
        auto &operands = value->GetExpressionOperandList();
        auto dim = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_SIZE_INDEX]->GetImmediateValue();
        auto base = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_COA_INDEX]->GetImmediateValue();
        auto dimIdx = operands[RUNTIME_GET_PARAM_OFFSET_OPERAND_INDEX_DIM_INDEX]->GetImmediateValue();
        return base + COA_INDEX_DIM_BASE + dim * 3 + dimIdx;
    }
    return -1;
}

std::vector<int64_t> OperationInterpreter::EvaluateOpImmediate(
    FunctionFrame *frame, const std::vector<OpImmediate> &opImmList) {
    std::vector<int64_t> result;
    for (auto &opImm : opImmList) {
        int64_t res = 0;
        if (opImm.IsSpecified()) {
            auto opImmValue = opImm.GetSpecifiedValue();
            auto coaIndex = GetAsParameterCoaIndex(opImmValue.Raw());
            if (coaIndex != -1) {
                auto attr = frame->callopAttr->GetLinearArgList()[coaIndex];
                res = EvaluateSymbolicScalar(attr);
            } else {
                res = EvaluateSymbolicScalar(opImm.GetSpecifiedValue());
            }
        } else {
            int index = opImm.GetParameterIndex();
            auto attr = frame->callopAttr->GetLinearArgList()[index];
            res = EvaluateSymbolicScalar(attr);
        }
        result.push_back(res);
    }
    return result;
}

void OperationInterpreter::ExecuteOperation(ExecuteOperationContext *ctx) {
    auto iOperands = OperationInterpreter::GetValidDataView(*ctx->ioperandDataViewList);
    auto oOperands = OperationInterpreter::GetValidDataView(*ctx->ooperandInplaceDataViewList);
    ExecuteOperationContext ctxValid = {ctx->frame, this, ctx->op, &iOperands, {}, &oOperands};
    try {
        OperationInterpreter::CallOperationInterpreterFunc(&ctxValid);
    } catch (std::exception &e) {
        // 打印当前 op 输出相关的动态信息，便于排查执行错误
        auto *op = ctx->op;
        auto *func = ctx->frame->func;
        if (op != nullptr) {
            auto &oTensors = op->GetOOperands();
            for (size_t i = 0; i < oTensors.size(); ++i) {
                auto tensor = oTensors[i];
                if (tensor == nullptr) {
                    continue;
                }
                // 静态 shape
                const auto &shape = tensor->shape;
                std::stringstream shapeSs;
                shapeSs << "[";
                for (size_t k = 0; k < shape.size(); ++k) {
                    if (k != 0) {
                        shapeSs << ", ";
                    }
                    shapeSs << shape[k];
                }
                shapeSs << "]";

                // 动态 valid shape / offset（符号表达形式）
                const auto &dynValidShape = tensor->GetDynValidShape();
                const auto &dynOffset = tensor->GetDynOffset();
                std::stringstream dynValidShapeSs;
                dynValidShapeSs << "[";
                for (size_t k = 0; k < dynValidShape.size(); ++k) {
                    if (k != 0) {
                        dynValidShapeSs << ", ";
                    }
                    dynValidShapeSs << dynValidShape[k].Dump();
                }
                dynValidShapeSs << "]";

                std::stringstream dynOffsetSs;
                dynOffsetSs << "[";
                for (size_t k = 0; k < dynOffset.size(); ++k) {
                    if (k != 0) {
                        dynOffsetSs << ", ";
                    }
                    dynOffsetSs << dynOffset[k].Dump();
                }
                dynOffsetSs << "]";

                ALOG_ERROR_F(
                    "ExecuteOperation error: op %s (magic=%d) output[%zu] tensorMagic=%d, "
                    "shape=%s, dynValidShape=%s, dynOffset=%s",
                    op->GetOpcodeStr().c_str(),
                    op->GetOpMagic(),
                    i,
                    tensor->magic,
                    shapeSs.str().c_str(),
                    dynValidShapeSs.str().c_str(),
                    dynOffsetSs.str().c_str());
            }
        }
        auto func = ctx->frame->func;
        func->DumpFile(config::LogTensorGraphFolder() + "/" + func->GetRawName() + ".tifwkgr");
        throw std::runtime_error(ctx->Dump() + e.what());
    }
}

std::string ExecuteOperationContext::Dump() const {
    std::stringstream ss;
    ss << "func: " << frame->func->GetRawName() << "\n";

    if (auto loc = op->GetLocation(); loc) {
        ss << "filename: " << loc->GetFileName() << "\n";
        ss << "lineno: " << loc->GetLineno() << "\n";
    }

    auto printType = [&](auto &viewList) {
        for (size_t i = 0; i < viewList.size(); i++) {
            if (i != 0)
                ss << ", ";
            ss << viewList[i]->DumpType();
        }
    };

    ss << op->Dump();
    printType(*ooperandInplaceDataViewList);
    ss << " = " << op->GetOpcodeStr() << " ";
    printType(*ioperandDataViewList);
    ss << "\n";
    return ss.str();
}
}