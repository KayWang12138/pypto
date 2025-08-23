/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "interface/interpreter/function.h"
#include "interface/utils/log.h"
#include "interface/interpreter/operation.h"

namespace npu::tile_fwk {

void ExecuteOpView(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto iop = ctx->ioperandDataViewList->at(0);
    ASSERT(iop != nullptr) << ctx->op->Dump();

    auto view = std::static_pointer_cast<ViewOpAttribute>(ctx->op->GetOpAttribute());
    std::vector<int> offset = ctx->opInter->EvaluateOffset(view->GetFromOffset(), view->GetFromDynOffset());
    std::vector<int> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, OpImmediate::Specified(view->GetToDynValidShape()));
    auto iopValid = std::make_shared<LogicalTensorData>(iop->GetData(), shape, offset);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    Calculator::CalcCopy(oop.get(), iopValid.get(), ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_VIEW, Opcode::OP_VIEW, ExecuteOpView);

void ExecuteOpAssemble(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);

    auto assemble = std::static_pointer_cast<AssembleOpAttribute>(ctx->op->GetOpAttribute());
    std::vector<int> offset = ctx->opInter->EvaluateOffset(assemble->GetToOffset(), assemble->GetToDynOffset());
    auto ret = oop->View(iop->GetShape(), offset);
    Calculator::CalcCopy(ret.get(), iop.get(), ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_ASSEMBLE, Opcode::OP_ASSEMBLE, ExecuteOpAssemble);

void ExecuteOpNone(ExecuteOperationContext *ctx) {
    (void)ctx;
}
REGISTER_CLACOP_FUNC(OP_PHASE1, Opcode::OP_PHASE1, ExecuteOpNone);
REGISTER_CLACOP_FUNC(OP_PHASE2, Opcode::OP_PHASE2, ExecuteOpNone);
REGISTER_CLACOP_FUNC(OP_SYNC_SRC, Opcode::OP_SYNC_SRC, ExecuteOpNone);
REGISTER_CLACOP_FUNC(OP_SYNC_DST, Opcode::OP_SYNC_DST, ExecuteOpNone);
REGISTER_CLACOP_FUNC(OP_BAR_V, Opcode::OP_BAR_V, ExecuteOpNone);
REGISTER_CLACOP_FUNC(OP_BAR_M, Opcode::OP_BAR_M, ExecuteOpNone);

void ExecuteOpCopyOut(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);

    auto copyout = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
    auto [from, toOffsetAttr] = copyout->GetCopyOutAttr();
    std::vector<int> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyout->GetShape());
    std::vector<int> rawShape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyout->GetRawShape());
    std::vector<int> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, toOffsetAttr);

    std::vector<int> iopShape = iop->GetShape();
    if (oop->GetIsSpilled()) {
        std::fill(toOffset.begin(), toOffset.end(), 0);
    }

    bool inputCombineAxisDone = ctx->op->GetBoolAttribute("input_combine_axis_done");
    if (inputCombineAxisDone && iopShape.size() == SIZE_TWO) {
        std::vector<int> iopTransShape = {iopShape[1], iopShape[0]};
        std::vector<int> axises = {0, 1};
        auto iopTrans = LogicalTensorData::CreateEmpty(iop->GetDataType(), iopTransShape, std::vector<int>(0));
        Calculator::CalcTransposeAdjDim(iopTrans.get(), iop.get(), axises[0], ctx->opInter->GetPoolPtr());
        iop = iopTrans;
    }

    // HACK: copyin's ooperand's shape might be different from copy shape
    auto iopValid = iop;
    auto oopValid = oop->View(iopValid->GetShape(), toOffset);
    if (from == MemoryType::MEM_L0C) {
        if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "atomic_add")) {
            // HACK: l0c copy out might add
            Calculator::CalcAdd(oopValid.get(), iopValid.get(), oopValid.get(), ctx->opInter->GetPoolPtr());
        } else {
            // HACK: l0c copy out might result in different data type
            Calculator::CalcCast(
                oopValid.get(), iopValid.get(), oopValid->GetDataType(), CastMode::CAST_NONE, ctx->opInter->GetPoolPtr());
        }
    } else {
        Calculator::CalcCopy(oopValid.get(), iopValid.get(), ctx->opInter->GetPoolPtr());
    }
}
REGISTER_CLACOP_FUNC(OP_COPY_OUT, Opcode::OP_COPY_OUT, ExecuteOpCopyOut);

void ExecuteOpCopyIn(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);

    auto copyin = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());

    bool outputCombineAxisDone = ctx->op->GetBoolAttribute("input_combine_axis_done");
    std::vector<int> oopShape = oop->GetShape();
    std::vector<int> axises = {0, 1};
    LogicalTensorDataPtr oopTrans;
    if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
        std::vector<int> transShape = {oopShape[1], oopShape[0]};
        oopTrans = LogicalTensorData::CreateEmpty(oop->GetDataType(), transShape, std::vector<int>(0));
    }

    // HACK: copyin's default attribute should be full tensor
    auto iopValid = iop;
    auto oopValid = oop;
    if (copyin != nullptr) {
        std::vector<int> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetShape());
        std::vector<int> rawShape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetRawShape());
        std::vector<int> fromOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetFromOffset());
        std::vector<int> dynvalidshape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyin->GetToDynValidShape());

        // HACK: copyin's ooperand's shape might be different from copy shape
        iopValid = iop->View(dynvalidshape, fromOffset);
        if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
            oopTrans = oopTrans->View(dynvalidshape, std::vector<int>(fromOffset.size(), 0));
        } else {
            oopValid = oop->View(dynvalidshape, std::vector<int>(fromOffset.size(), 0));
        }
    }

    if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
        Calculator::CalcCopy(oopTrans.get(), iopValid.get(), ctx->opInter->GetPoolPtr());
        Calculator::CalcTransposeAdjDim(oopValid.get(), oopTrans.get(), axises[0], ctx->opInter->GetPoolPtr());
    } else {
        Calculator::CalcCopy(oopValid.get(), iopValid.get(), ctx->opInter->GetPoolPtr());
    }
}
REGISTER_CLACOP_FUNC(OP_COPY_IN, Opcode::OP_COPY_IN, ExecuteOpCopyIn);

void ExecuteOpCopy(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    Calculator::CalcCopy(oop.get(), iop.get(), ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_REGISTER_COPY, Opcode::OP_REGISTER_COPY, ExecuteOpCopy);

std::string FormatString(const std::string &s, OperationInterpreter *opInter) {
    std::stringstream ss;
    size_t pos = 0;
    while (pos < s.size()) {
        if (s[pos] == '{') {
            size_t end = s.find('}', pos + 1);
            if (end == std::string::npos) {
                ss << s.substr(pos);
                break;
            } else {
                auto symbol = s.substr(pos + 1, end - pos - 1);
                ss << opInter->EvaluateSymbolicScalar(symbol);
                pos = end + 1;
            }
        } else {
            ss << s[pos];
            pos++;
        }
    }
    return ss.str();
}

void ExecutePrint(ExecuteOperationContext *ctx) {
    auto &iop = ctx->ioperandDataViewList->at(0);
    auto cond = ctx->op->GetSymbolicScalarAttribute(OP_ATTR_PREFIX + "cond");
    if (!ctx->opInter->EvaluateSymbolicScalar(cond)) {
        return;
    }

    if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "fname")) {
        auto fname = ctx->op->GetStringAttribute(OP_ATTR_PREFIX + "fname");
        auto fpath = config::LogTopFolder() + "/tensor/" + FormatString(fname, ctx->opInter);
        auto shape = iop->GetValidShape();
        if (shape.empty()) {
            shape = iop->GetShape();
        }
        auto oop = LogicalTensorData::CreateEmpty(iop->GetDataType(), shape, shape);
        Calculator::CalcCopy(oop.get(), iop.get(), ctx->opInter->GetPoolPtr());
        oop->GetData()->ToFile(fpath);
    }

    if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "msg")) {
        auto msg = ctx->op->GetStringAttribute(OP_ATTR_PREFIX + "msg");
        std::cout << FormatString(msg, ctx->opInter) << "\n" << iop->Dump() << std::endl;
    }
}
REGISTER_CLACOP_FUNC(OP_PRINT, Opcode::OP_PRINT, ExecutePrint);
}