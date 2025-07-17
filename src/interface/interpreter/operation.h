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
 * \file operation_interpreter.h
 * \brief
 */
/*for flow verify tool */

#pragma once

#include "interface/utils/thread_pool.h"
#include "interface/operation/attribute.h"
#include "interface/tensor/symbolic_scalar_evaluate.h"
#include "calculator.h"

namespace npu::tile_fwk {

constexpr int DATATYPE_EIGHT = 8;

struct FunctionFrame;
struct ExecuteOperationContext {
    FunctionFrame *frame;

    Operation *op;
    const std::vector<std::shared_ptr<LogicalTensorData>> *ioperandDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> *ooperandDataViewList;
    std::vector<std::shared_ptr<LogicalTensorData>> *ooperandInplaceDataViewList;
};

class OperationInterpreter {
public:
    OperationInterpreter(int threadCount) : evaluateSymbol(std::make_shared<EvaluateSymbol>()), pool(threadCount) {}

    std::shared_ptr<EvaluateSymbol> evaluateSymbol;

    ScalarImmediateType EvaluateSymbolicScalar(const SymbolicScalar &ss) {
        return evaluateSymbol->EvaluateSymbolicScalar(ss);
    }
    std::vector<int> EvaluateOffset(const std::vector<int> &offset, const std::vector<SymbolicScalar> &dynOffset) {
        return evaluateSymbol->EvaluateOffset(offset, dynOffset);
    }
    std::vector<int> EvaluateOpImmediate(FunctionFrame *frame, const std::vector<OpImmediate> &opImmList);

    std::vector<int> EvaluateValidShape(const std::vector<SymbolicScalar> &dynValidShape) {
        return evaluateSymbol->EvaluateValidShape(dynValidShape);
    }

    void ExecuteOperation(ExecuteOperationContext *ctx);

    util::ThreadPool &GetPool() { return pool; }

    int GetThreadCount() const { return pool.GetThreadCount(); }

private:
    std::vector<std::shared_ptr<LogicalTensorData>> GetValidDataView(
        const std::vector<std::shared_ptr<LogicalTensorData>> &dataViewList) const {
        std::vector<std::shared_ptr<LogicalTensorData>> result = dataViewList;
        for (size_t index = 0; index < dataViewList.size(); index++) {
            auto operand = dataViewList[index];
            if ((!operand->GetValidShape().empty()) && (operand->GetValidShape() != operand->GetShape())) {
                result[index] =
                    operand->View(operand->GetValidShape(), std::vector<int>(operand->GetValidShape().size(), 0));
            }
        }
        return result;
    }

    void ExecuteOpView(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto iop = ctx->ioperandDataViewList->at(0);
        ASSERT(iop != nullptr) << ctx->op->Dump();

        auto view = std::static_pointer_cast<ViewOpAttribute>(ctx->op->GetOpAttribute());
        std::vector<int> offset = EvaluateOffset(view->GetFromOffset(), view->GetFromDynOffset());
        std::vector<int> shape = ctx->op->GetOOperands().front()->GetShape();
        for (size_t k = 0; k < shape.size(); k++) {
            // Aligned
            ASSERT(offset[k] + shape[k] <= iop->GetData()->GetShape()[k]);
        }

        auto iopValid = std::make_shared<LogicalTensorData>(iop->GetData(), shape, offset);
        auto oop = ctx->ooperandInplaceDataViewList->at(0);
        Calculator::CalcCopy(oop.get(), iopValid.get(), &pool);
    }

    void ExecuteOpCopyIn(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &oop = ctx->ooperandInplaceDataViewList->at(0);
        auto &iop = ctx->ioperandDataViewList->at(0);

        auto copyin = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());

        bool outputCombineAxisDone = ctx->op->GetBoolAttribute("input_combine_axis_done");
        std::vector<int> oopShape = oop->GetShape();
        std::vector<int> axises = {0, 1};
        std::shared_ptr<LogicalTensorData> oopTrans;
        if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
            std::vector<int> transShape = {oopShape[1], oopShape[0]};
            oopTrans = LogicalTensorData::CreateEmpty(oop->GetDataType(), transShape, std::vector<int>(0));
        }

        // HACK: copyin's default attribute should be full tensor
        auto iopValid = iop;
        auto oopValid = oop;
        if (copyin != nullptr) {
            std::vector<int> shape = EvaluateOpImmediate(ctx->frame, copyin->GetShape());
            std::vector<int> rawShape = EvaluateOpImmediate(ctx->frame, copyin->GetRawShape());
            std::vector<int> fromOffset = EvaluateOpImmediate(ctx->frame, copyin->GetFromOffset());
            std::vector<int> dynvalidshape = EvaluateOpImmediate(ctx->frame, copyin->GetToDynValidShape());

            // HACK: copyin's ooperand's shape might be different from copy shape
            iopValid = iop->View(shape, fromOffset);
            if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
                oopTrans = oopTrans->View(shape, std::vector<int>(fromOffset.size(), 0));
            } else {
                oopValid = oop->View(shape, std::vector<int>(fromOffset.size(), 0));
            }
        }

        if (outputCombineAxisDone && oopShape.size() == SIZE_TWO) {
            Calculator::CalcCopy(oopTrans.get(), iopValid.get(), &pool);
            Calculator::CalcTransposeAdjDim(oopValid.get(), oopTrans.get(), axises[0], &pool);
        } else {
            Calculator::CalcCopy(oopValid.get(), iopValid.get(), &pool);
        }
    }

    void ExecuteOpIndexOutcast(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ioperandDataViewList->size() == SIZE_THREE);
        auto oop = ctx->ooperandInplaceDataViewList->at(0);
        auto src = ctx->ioperandDataViewList->at(0);
        auto index = ctx->ioperandDataViewList->at(1);
        auto dst = ctx->ioperandDataViewList->at(2);
        int axis = ctx->op->GetIntAttribute("axis");

        auto oopCopy = oop;

        if (std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute())) {
            auto copyoutAttr = std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
            std::vector<int> shape = dst->GetShape();
            std::vector<int> toOffset = EvaluateOpImmediate(ctx->frame, copyoutAttr->GetToOffset());
            oopCopy = oop->View(shape, toOffset);
        }
        Calculator::CalcIndexCopy(oopCopy.get(), src.get(), index.get(), dst.get(), axis, &pool);
    }

    void ExecuteOpCopy(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &oop = ctx->ooperandInplaceDataViewList->at(0);
        auto &iop = ctx->ioperandDataViewList->at(0);
        Calculator::CalcCopy(oop.get(), iop.get(), &pool);
    }

    void ExecuteOpReshape(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &oop = ctx->ooperandInplaceDataViewList->at(0);
        auto &iop = ctx->ioperandDataViewList->at(0);
        Calculator::CalcReshape(oop.get(), iop.get(), &pool);
    }

    void ExecuteOpExpand(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto oop = ctx->ooperandInplaceDataViewList->at(0);
        auto iop = ctx->ioperandDataViewList->at(0);

        auto oopShape = oop->GetShape();
        auto iopShape = iop->GetShape();
        int axis = FindDifferentAxis(oopShape, iopShape);
        if (axis != -1) {
            if (iopShape[axis] != 1) {
                iopShape[axis] = 1;
                iop = iop->View(iopShape, std::vector<int>(iopShape.size(), 0));
            }
            Calculator::CalcExpand(oop.get(), iop.get(), axis, &pool);
        } else {
            Calculator::CalcCopy(oop.get(), iop.get(), &pool);
        }
    }

    void ExecuteOpTransposeDataMove(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto oop = ctx->ooperandInplaceDataViewList->at(0);
        auto iop = ctx->ioperandDataViewList->at(0);

        std::vector<int> axises = ctx->op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
        auto oopCopy = oop;
        if (std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute())) {
            auto copyoutAttr = std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
            std::vector<int> shape = EvaluateOpImmediate(ctx->frame, copyoutAttr->GetShape());
            std::vector<int> toOffset = EvaluateOpImmediate(ctx->frame, copyoutAttr->GetToOffset());
            oopCopy = oop->View(shape, toOffset);
        }
        std::sort(axises.begin(), axises.end());
        ASSERT(axises[0] + 1 == axises[1]);
        Calculator::CalcTransposeAdjDim(oopCopy.get(), iop.get(), axises[0], &pool);
    }

    void ExecuteOpTranspose(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto oop = ctx->ooperandInplaceDataViewList->at(0);
        auto iop = ctx->ioperandDataViewList->at(0);
        auto axises = ctx->op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
        std::sort(axises.begin(), axises.end());
        ASSERT(axises[0] + 1 == axises[1]);
        Calculator::CalcTransposeAdjDim(oop.get(), iop.get(), axises[0], &pool);
    }

    void ExecuteOpAssemble(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &oop = ctx->ooperandInplaceDataViewList->at(0);
        auto &iop = ctx->ioperandDataViewList->at(0);

        auto assemble = std::static_pointer_cast<AssembleOpAttribute>(ctx->op->GetOpAttribute());
        std::vector<int> offset = EvaluateOffset(assemble->GetToOffset(), assemble->GetToDynOffset());
        auto ret = oop->View(iop->GetShape(), offset);
        Calculator::CalcCopy(ret.get(), iop.get(), &pool);
    }

    void ExecuteOpAlloc(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() <= 1);
        ASSERT(ctx->ioperandDataViewList->size() == 0);
    }

    void ExecuteDuplicate(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &ret = ctx->ooperandInplaceDataViewList->at(0);
        auto &oper = ctx->ioperandDataViewList->at(0);
        Opcode opCode = ctx->op->GetOpcode();
        if (opCode == Opcode::OP_L1_TO_L0_BT) {
            std::vector<int> axises = {0, 1};
            Calculator::CalcTransposeAdjDim(ret.get(), oper.get(), axises[0], &pool);
        } else {
            Calculator::CalcCopy(ret.get(), oper.get(), &pool);
        }
    }

    void ExecuteOpCopyOut(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &oop = ctx->ooperandInplaceDataViewList->at(0);
        auto iop = ctx->ioperandDataViewList->at(0);

        auto copyout = std::static_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
        auto [from, toOffsetAttr] = copyout->GetCopyOutAttr();
        std::vector<int> shape = EvaluateOpImmediate(ctx->frame, copyout->GetShape());
        std::vector<int> rawShape = EvaluateOpImmediate(ctx->frame, copyout->GetRawShape());
        std::vector<int> toOffset = EvaluateOpImmediate(ctx->frame, toOffsetAttr);

        std::vector<int> iopShape = iop->GetShape();
        if (oop->GetIsSpilled()) {
            std::fill(toOffset.begin(), toOffset.end(), 0);
        }

        bool inputCombineAxisDone = ctx->op->GetBoolAttribute("input_combine_axis_done");
        if (inputCombineAxisDone && iopShape.size() == SIZE_TWO) {
            std::vector<int> iopTransShape = {iopShape[1], iopShape[0]};
            std::vector<int> axises = {0, 1};
            auto iopTrans = LogicalTensorData::CreateEmpty(iop->GetDataType(), iopTransShape, std::vector<int>(0));
            Calculator::CalcTransposeAdjDim(iopTrans.get(), iop.get(), axises[0], &pool);
            iop = iopTrans;
        }

        // HACK: copyin's ooperand's shape might be different from copy shape
        auto iopValid = iop->View(shape, std::vector<int>(toOffset.size(), 0));
        auto oopValid = oop->View(shape, toOffset);
        if (from == MemoryType::MEM_L0C) {
            if (ctx->op->HasAttribute(OP_ATTR_PREFIX + "atomic_add")) {
                // HACK: l0c copy out might add
                Calculator::CalcAdd(oopValid.get(), iopValid.get(), oopValid.get(), &pool);
            } else {
                // HACK: l0c copy out might result in different data type
                Calculator::CalcCast(
                    oopValid.get(), iopValid.get(), oopValid->GetDataType(), CastMode::CAST_NONE, &pool);
            }
        } else {
            Calculator::CalcCopy(oopValid.get(), iopValid.get(), &pool);
        }
    }

    void ExecuteOpVecDup(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 0);
        auto &ret = ctx->ooperandInplaceDataViewList->at(0);
        auto scalarVal = ctx->op->GetAttribute(OpAttributeKey::scalar);
        auto value = scalarVal.HasValue() ? npu::tile_fwk::AnyCast<float>(scalarVal) : 0.0f;
        ASSERT(ret->GetDataType() == DT_FP32);
        Element element(DT_FP32, value);
        Calculator::CalcVecDup(ret.get(), &element, &pool);
    }

    void ExecuteOpCast(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &ret = ctx->ooperandInplaceDataViewList->at(0);
        auto &iop = ctx->ioperandDataViewList->at(0);
        CastMode mode = static_cast<CastMode>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "mode"));
        Calculator::CalcCast(ret.get(), iop.get(), ret->GetDataType(), mode, &pool);
    }

    template <Opcode opcode>
    void ExecuteOpUnary(ExecuteOperationContext *ctx) {
        static_assert(
            opcode == Opcode::OP_EXP || opcode == Opcode::OP_SQRT || opcode == Opcode::OP_ABS, "Invalid opcode");
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &ret = ctx->ooperandInplaceDataViewList->at(0);
        auto &iop = ctx->ioperandDataViewList->at(0);
        switch (opcode) {
            case Opcode::OP_EXP: Calculator::CalcExp(ret.get(), iop.get(), &pool); break;
            case Opcode::OP_SQRT: Calculator::CalcSqrt(ret.get(), iop.get(), &pool); break;
            case Opcode::OP_ABS: Calculator::CalcAbs(ret.get(), iop.get(), &pool); break;
            default: ASSERT(false);
        }
    }

    int FindDifferentAxis(const std::vector<int> &lhs, const std::vector<int> &rhs) {
        ASSERT(lhs.size() == rhs.size());
        for (size_t k = 0; k < lhs.size(); k++) {
            if (lhs[k] != rhs[k]) {
                return static_cast<int>(k);
            }
        }
        return -1;
    }

    template <Opcode opcode>
    void ExecuteOpBinary(ExecuteOperationContext *ctx) {
        static_assert(opcode == Opcode::OP_ADD || opcode == Opcode::OP_SUB || opcode == Opcode::OP_MUL ||
                          opcode == Opcode::OP_DIV || opcode == Opcode::OP_S_MAX || opcode == Opcode::OP_S_MIN ||
                          opcode == Opcode::OP_PAIRMAX || opcode == Opcode::OP_PAIRSUM ||
                          opcode == Opcode::OP_ADD_BRC || opcode == Opcode::OP_SUB_BRC ||
                          opcode == Opcode::OP_MUL_BRC || opcode == Opcode::OP_DIV_BRC,
            "Invalid opcode");
        if (opcode == Opcode::OP_ADD_BRC || opcode == Opcode::OP_SUB_BRC || opcode == Opcode::OP_MUL_BRC ||
            opcode == Opcode::OP_DIV_BRC) {
            ASSERT(ctx->ooperandInplaceDataViewList->size() == SIZE_TWO);
        } else {
            ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        }
        ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO);
        auto ret = ctx->ooperandInplaceDataViewList->at(0);
        auto lhs = ctx->ioperandDataViewList->at(0);
        auto rhs = ctx->ioperandDataViewList->at(1);
        bool inputCombineAxisDone = false;

        if (opcode == Opcode::OP_ADD_BRC || opcode == Opcode::OP_SUB_BRC || opcode == Opcode::OP_MUL_BRC ||
            opcode == Opcode::OP_DIV_BRC) {
            inputCombineAxisDone = ctx->op->GetBoolAttribute("input_combine_axis_done");
        }
        if (inputCombineAxisDone) {
            auto lhsShape = lhs->GetShape();
            auto rhsShape = rhs->GetShape();
            std::vector<int> axises = {0, 1};
            if (lhsShape[0] == 1 && lhsShape.size() == SIZE_TWO) {
                std::vector<int> lhsTransShape = {lhs->GetShape()[1], lhs->GetShape()[0]};
                auto lhsTrans =
                    LogicalTensorData::CreateEmpty(lhs->GetDataType(), lhsTransShape, std::vector<int>(0));
                Calculator::CalcTransposeAdjDim(lhsTrans.get(), lhs.get(), axises[0], &pool);
                lhs = lhsTrans;
            }
            if (rhsShape[0] == 1 && rhsShape.size() == SIZE_TWO) {
                std::vector<int> rhsTransShape = {rhs->GetShape()[1], rhs->GetShape()[0]};
                auto rhsTrans =
                    LogicalTensorData::CreateEmpty(rhs->GetDataType(), rhsTransShape, std::vector<int>(0));
                Calculator::CalcTransposeAdjDim(rhsTrans.get(), rhs.get(), axises[0], &pool);
                rhs = rhsTrans;
            }
        }

        if (lhs->GetShape() != rhs->GetShape()) {
            // HACK: element wise bop auto expand axis
            auto retShape = ret->GetShape();
            auto operExpand =
                LogicalTensorData::CreateEmpty(ret->GetDataType(), ret->GetShape(), std::vector<int>(0));
            auto lhsShape = lhs->GetShape();
            auto rhsShape = rhs->GetShape();
            if (retShape != lhsShape) {
                int axis = FindDifferentAxis(retShape, lhsShape);
                ASSERT(axis != -1);
                Calculator::CalcExpand(operExpand.get(), lhs.get(), axis, &pool);
                lhs = operExpand;
            } else {
                int axis = FindDifferentAxis(retShape, rhsShape);
                ASSERT(axis != -1);
                Calculator::CalcExpand(operExpand.get(), rhs.get(), axis, &pool);
                rhs = operExpand;
            }
        }
        switch (opcode) {
            case Opcode::OP_ADD: Calculator::CalcAdd(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_ADD_BRC: Calculator::CalcAddBrc(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_PAIRSUM: Calculator::CalcPairSum(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_SUB: Calculator::CalcSub(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_SUB_BRC: Calculator::CalcSubBrc(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_MUL: Calculator::CalcMul(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_MUL_BRC: Calculator::CalcMulBrc(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_DIV: Calculator::CalcDiv(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_DIV_BRC: Calculator::CalcDivBrc(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_S_MAX: Calculator::CalcMax(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_PAIRMAX: Calculator::CalcPairMax(ret.get(), lhs.get(), rhs.get(), &pool); break;
            case Opcode::OP_S_MIN: Calculator::CalcMin(ret.get(), lhs.get(), rhs.get(), &pool); break;
            default: ASSERT(false);
        }
    }

    template <Opcode opcode>
    void ExecuteOpBinaryScalar(ExecuteOperationContext *ctx) {
        static_assert(opcode == Opcode::OP_ADDS || opcode == Opcode::OP_SUBS || opcode == Opcode::OP_MULS ||
                          opcode == Opcode::OP_DIVS || opcode == Opcode::OP_S_MAXS || opcode == Opcode::OP_S_MINS,
            "Invalid opcode");
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto &ret = ctx->ooperandInplaceDataViewList->at(0);
        auto &lhs = ctx->ioperandDataViewList->at(0);
        auto scalarVal = ctx->op->GetAttribute(OpAttributeKey::scalar);
        auto value = scalarVal.HasValue() ? npu::tile_fwk::AnyCast<float>(scalarVal) : 0.0f;
        bool reverse = ctx->op->GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand");
        ASSERT(ret->GetDataType() == DT_FP32);
        Element element(DT_FP32, value);
        switch (opcode) {
            case Opcode::OP_ADDS: Calculator::CalcAddS(ret.get(), lhs.get(), &element, reverse, &pool); break;
            case Opcode::OP_SUBS: Calculator::CalcSubS(ret.get(), lhs.get(), &element, reverse, &pool); break;
            case Opcode::OP_MULS: Calculator::CalcMulS(ret.get(), lhs.get(), &element, reverse, &pool); break;
            case Opcode::OP_DIVS: Calculator::CalcDivS(ret.get(), lhs.get(), &element, reverse, &pool); break;
            case Opcode::OP_S_MAXS: Calculator::CalcMaxS(ret.get(), lhs.get(), &element, reverse, &pool); break;
            case Opcode::OP_S_MINS: Calculator::CalcMinS(ret.get(), lhs.get(), &element, reverse, &pool); break;
            default: ASSERT(false);
        }
    }

    template <Opcode opcode>
    void ExecuteOpReduce(ExecuteOperationContext *ctx) {
        static_assert(opcode == Opcode::OP_ROWSUM_SINGLE || opcode == Opcode::OP_ROWMAX_SINGLE ||
                          opcode == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE ||
                          opcode == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE || opcode == Opcode::OP_ROWSUMLINE,
            "Invalid opcode");
        ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
        ASSERT(ctx->ioperandDataViewList->size() == 1);
        auto oop = ctx->ooperandInplaceDataViewList->at(0);
        auto iop = ctx->ioperandDataViewList->at(0);
        int axis = ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "AXIS");
        bool outputCombineAxisDone = false;
        if (opcode == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE || opcode == Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE) {
            outputCombineAxisDone = ctx->op->GetBoolAttribute("output_combine_axis_done");
        }
        std::shared_ptr<LogicalTensorData> oopTrans;
        switch (opcode) {
            case Opcode::OP_ROWSUM_SINGLE:
            case Opcode::OP_ROWSUMLINE:
            case Opcode::OP_ROWMAX_SINGLE: {
                if (oop->GetShape()[axis] != 1) {
                    std::vector<int> oopShape = oop->GetShape();
                    oopShape[axis] = 1;
                    oop = oop->View(oopShape, std::vector<int>(oopShape.size(), 0));
                }
            } break;
            case Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE:
            case Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE: {
                if (outputCombineAxisDone) {
                    std::vector<int> transShape = {oop->GetShape()[1], oop->GetShape()[0]};
                    transShape[axis] = 1;
                    oopTrans = LogicalTensorData::CreateEmpty(oop->GetDataType(), transShape, std::vector<int>(0));
                }
            } break;
            default: ASSERT(false);
        }

        switch (opcode) {
            case Opcode::OP_ROWSUM_SINGLE:
            case Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE: {
                if (outputCombineAxisDone) {
                    std::vector<int> axises = {0, 1};
                    Calculator::CalcRowSumSingle(oopTrans.get(), iop.get(), axis, &pool);
                    Calculator::CalcTransposeAdjDim(oop.get(), oopTrans.get(), axises[0], &pool);
                } else {
                    Calculator::CalcRowSumSingle(oop.get(), iop.get(), axis, &pool);
                }
            } break;
            case Opcode::OP_ROWMAX_SINGLE:
            case Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE: {
                if (outputCombineAxisDone) {
                    std::vector<int> axises = {0, 1};
                    Calculator::CalcRowMaxSingle(oopTrans.get(), iop.get(), axis, &pool);
                    Calculator::CalcTransposeAdjDim(oop.get(), oopTrans.get(), axises[0], &pool);
                } else {
                    Calculator::CalcRowMaxSingle(oop.get(), iop.get(), axis, &pool);
                }
            } break;
            case Opcode::OP_ROWSUMLINE: Calculator::CalcRowSumLine(oop.get(), iop.get(), axis, &pool); break;
            default: ASSERT(false);
        }
    }

    void ExecuteOpAMulB(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        ASSERT(ctx->ioperandDataViewList->size() == SIZE_TWO || ctx->ioperandDataViewList->size() == SIZE_THREE);
        auto ret = ctx->ooperandInplaceDataViewList->at(0);
        auto lhs = ctx->ioperandDataViewList->at(0);
        auto rhs = ctx->ioperandDataViewList->at(1);
        if (ret->GetDataType() != lhs->GetDataType()) {
            // HACK: matmul type might be different between ioperand and ooperand
            auto lhsCast = LogicalTensorData::CreateEmpty(ret->GetDataType(), lhs->GetShape(), lhs->GetValidShape());
            auto rhsCast = LogicalTensorData::CreateEmpty(ret->GetDataType(), rhs->GetShape(), rhs->GetValidShape());
            Calculator::CalcCast(lhsCast.get(), lhs.get(), ret->GetDataType(), CastMode::CAST_NONE, &pool);
            Calculator::CalcCast(rhsCast.get(), rhs.get(), ret->GetDataType(), CastMode::CAST_NONE, &pool);
            lhs = lhsCast;
            rhs = rhsCast;
        }

        int k1 = ctx->op->GetTileShape().K(1);
        int k2 = ctx->op->GetTileShape().K(2);
        int kStep = lhs->GetShape()[1];
        if (k1 != 0 && k2 != 0) {
            kStep = std::gcd(k1, k2);
        }
        switch (ctx->op->GetOpcode()) {
            case Opcode::OP_A_MUL_B: Calculator::CalcMatMul(ret.get(), lhs.get(), rhs.get(), kStep, &pool); break;
            case Opcode::OP_A_MULACC_B:
                Calculator::CalcMatMulAcc(
                    ret.get(), lhs.get(), rhs.get(), kStep, ctx->ioperandDataViewList->at(SIZE_TWO).get(), &pool);
                break;
            case Opcode::OP_A_MUL_BT: Calculator::CalcMatMulTrans(ret.get(), lhs.get(), rhs.get(), kStep, &pool); break;
            case Opcode::OP_A_MULACC_BT:
                Calculator::CalcMatMulTransAcc(
                    ret.get(), lhs.get(), rhs.get(), kStep, ctx->ioperandDataViewList->at(SIZE_TWO).get(), &pool);
                break;
            default: ASSERT(false); break;
        }
    }

    void ExecuteOpReduceAcc(ExecuteOperationContext *ctx) {
        ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
        auto &ret = ctx->ooperandInplaceDataViewList->at(0);

        std::vector<LogicalTensorData *> dataViewList(ctx->ioperandDataViewList->size());
        for (size_t k = 0; k < ctx->ioperandDataViewList->size(); k++) {
            dataViewList[k] = ctx->ioperandDataViewList->at(k).get();
        }

        Calculator::CalcReduceAcc(ret.get(), &dataViewList, &pool);
    }

    void ExecuteOpNone(ExecuteOperationContext *ctx) { (void)ctx; }

private:
    util::ThreadPool pool{64};
};

#undef CASE_DATA_TYPE_DIS
#undef CASE_DATA_TYPE
} // namespace npu::tile_fwk
