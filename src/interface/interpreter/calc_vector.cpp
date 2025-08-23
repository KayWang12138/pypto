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

static int FindDifferentAxis(const std::vector<int> &lhs, const std::vector<int> &rhs) {
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
        opcode == Opcode::OP_MUL_BRC || opcode == Opcode::OP_DIV_BRC || opcode == Opcode::OP_PAIRMIN,
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
            Calculator::CalcTransposeAdjDim(lhsTrans.get(), lhs.get(), axises[0], ctx->opInter->GetPoolPtr());
            lhs = lhsTrans;
        }
        if (rhsShape[0] == 1 && rhsShape.size() == SIZE_TWO) {
            std::vector<int> rhsTransShape = {rhs->GetShape()[1], rhs->GetShape()[0]};
            auto rhsTrans =
                LogicalTensorData::CreateEmpty(rhs->GetDataType(), rhsTransShape, std::vector<int>(0));
            Calculator::CalcTransposeAdjDim(rhsTrans.get(), rhs.get(), axises[0], ctx->opInter->GetPoolPtr());
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
            Calculator::CalcExpand(operExpand.get(), lhs.get(), axis, ctx->opInter->GetPoolPtr());
            lhs = operExpand;
        } else {
            int axis = FindDifferentAxis(retShape, rhsShape);
            ASSERT(axis != -1);
            Calculator::CalcExpand(operExpand.get(), rhs.get(), axis, ctx->opInter->GetPoolPtr());
            rhs = operExpand;
        }
    }
    switch (opcode) {
        case Opcode::OP_ADD: Calculator::CalcAdd(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_ADD_BRC: Calculator::CalcAddBrc(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_PAIRSUM: Calculator::CalcPairSum(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_SUB: Calculator::CalcSub(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_SUB_BRC: Calculator::CalcSubBrc(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_MUL: Calculator::CalcMul(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_MUL_BRC: Calculator::CalcMulBrc(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_DIV: Calculator::CalcDiv(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_DIV_BRC: Calculator::CalcDivBrc(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_S_MAX: Calculator::CalcMax(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_PAIRMAX: Calculator::CalcPairMax(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_PAIRMIN: Calculator::CalcPairMin(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_S_MIN: Calculator::CalcMin(ret.get(), lhs.get(), rhs.get(), ctx->opInter->GetPoolPtr()); break;
        default: ASSERT(false);
    }
}
REGISTER_CLACOP_FUNC(OP_ADD, Opcode::OP_ADD, ExecuteOpBinary<Opcode::OP_ADD>);
REGISTER_CLACOP_FUNC(OP_ADD_BRC, Opcode::OP_ADD_BRC, ExecuteOpBinary<Opcode::OP_ADD_BRC>);
REGISTER_CLACOP_FUNC(OP_SUB, Opcode::OP_SUB, ExecuteOpBinary<Opcode::OP_SUB>);
REGISTER_CLACOP_FUNC(OP_SUB_BRC, Opcode::OP_SUB_BRC, ExecuteOpBinary<Opcode::OP_SUB_BRC>);
REGISTER_CLACOP_FUNC(OP_MUL, Opcode::OP_MUL, ExecuteOpBinary<Opcode::OP_MUL>);
REGISTER_CLACOP_FUNC(OP_MUL_BRC, Opcode::OP_MUL_BRC, ExecuteOpBinary<Opcode::OP_MUL_BRC>);
REGISTER_CLACOP_FUNC(OP_DIV, Opcode::OP_DIV, ExecuteOpBinary<Opcode::OP_DIV>);
REGISTER_CLACOP_FUNC(OP_DIV_BRC, Opcode::OP_DIV_BRC, ExecuteOpBinary<Opcode::OP_DIV_BRC>);
REGISTER_CLACOP_FUNC(OP_S_ADD, Opcode::OP_S_ADD, ExecuteOpBinary<Opcode::OP_ADD>);
REGISTER_CLACOP_FUNC(OP_S_SUB, Opcode::OP_S_SUB, ExecuteOpBinary<Opcode::OP_SUB>);
REGISTER_CLACOP_FUNC(OP_S_MUL, Opcode::OP_S_MUL, ExecuteOpBinary<Opcode::OP_MUL>);
REGISTER_CLACOP_FUNC(OP_S_DIV, Opcode::OP_S_DIV, ExecuteOpBinary<Opcode::OP_DIV>);
REGISTER_CLACOP_FUNC(OP_S_MAX, Opcode::OP_S_MAX, ExecuteOpBinary<Opcode::OP_S_MAX>);
REGISTER_CLACOP_FUNC(OP_PAIRMAX, Opcode::OP_PAIRMAX, ExecuteOpBinary<Opcode::OP_PAIRMAX>);
REGISTER_CLACOP_FUNC(OP_PAIRMIN, Opcode::OP_PAIRMIN, ExecuteOpBinary<Opcode::OP_PAIRMIN>);
REGISTER_CLACOP_FUNC(OP_PAIRSUM, Opcode::OP_PAIRSUM, ExecuteOpBinary<Opcode::OP_PAIRSUM>);
REGISTER_CLACOP_FUNC(OP_S_MIN, Opcode::OP_S_MIN, ExecuteOpBinary<Opcode::OP_S_MIN>);
REGISTER_CLACOP_FUNC(OP_MAXIMUM, Opcode::OP_MAXIMUM, ExecuteOpBinary<Opcode::OP_S_MAX>);

void ExecuteOpVecDup(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 0);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto scalarVal = ctx->op->GetAttribute(OpAttributeKey::scalar);
    auto element = scalarVal.HasValue() ? npu::tile_fwk::AnyCast<Element>(scalarVal) : Element(DT_FP32, 0.0f);
    ASSERT(ret->GetDataType() == DT_FP32);
    Calculator::CalcVecDup(ret.get(), &element, ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_VEC_DUP, Opcode::OP_VEC_DUP, ExecuteOpVecDup);

template <Opcode opcode>
void ExecuteOpReduce(ExecuteOperationContext *ctx) {
    static_assert(opcode == Opcode::OP_ROWSUM_SINGLE || opcode == Opcode::OP_ROWMAX_SINGLE ||
        opcode == Opcode::OP_ROWMIN_SINGLE || opcode == Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE ||
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
    LogicalTensorDataPtr oopTrans;
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
        case Opcode::OP_ROWMIN_SINGLE: {
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
                Calculator::CalcRowSumSingle(oopTrans.get(), iop.get(), axis, ctx->opInter->GetPoolPtr());
                Calculator::CalcTransposeAdjDim(oop.get(), oopTrans.get(), axises[0], ctx->opInter->GetPoolPtr());
            } else {
                Calculator::CalcRowSumSingle(oop.get(), iop.get(), axis, ctx->opInter->GetPoolPtr());
            }
        } break;
        case Opcode::OP_ROWMAX_SINGLE:
        case Opcode::OP_ROWMIN_SINGLE:
        case Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE: {
            if (outputCombineAxisDone) {
                std::vector<int> axises = {0, 1};
                Calculator::CalcRowMaxSingle(oopTrans.get(), iop.get(), axis, ctx->opInter->GetPoolPtr());
                Calculator::CalcTransposeAdjDim(oop.get(), oopTrans.get(), axises[0], ctx->opInter->GetPoolPtr());
            } else {
                Calculator::CalcRowMaxSingle(oop.get(), iop.get(), axis, ctx->opInter->GetPoolPtr());
            }
        } break;
        case Opcode::OP_ROWSUMLINE: Calculator::CalcRowSumLine(oop.get(), iop.get(), axis, ctx->opInter->GetPoolPtr()); break;
        default: ASSERT(false);
    }
}
REGISTER_CLACOP_FUNC(OP_ROWSUM_SINGLE, Opcode::OP_ROWSUM_SINGLE, ExecuteOpReduce<Opcode::OP_ROWSUM_SINGLE>);
REGISTER_CLACOP_FUNC(OP_ROWSUMLINE, Opcode::OP_ROWSUMLINE, ExecuteOpReduce<Opcode::OP_ROWSUMLINE>);
REGISTER_CLACOP_FUNC(OP_ROWMAX_SINGLE, Opcode::OP_ROWMAX_SINGLE, ExecuteOpReduce<Opcode::OP_ROWMAX_SINGLE>);
REGISTER_CLACOP_FUNC(OP_ROWMIN_SINGLE, Opcode::OP_ROWMIN_SINGLE, ExecuteOpReduce<Opcode::OP_ROWMIN_SINGLE>);
REGISTER_CLACOP_FUNC(OP_ROWSUM_COMBINE_AXIS_SINGLE, Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE, ExecuteOpReduce<Opcode::OP_ROWSUM_COMBINE_AXIS_SINGLE>);
REGISTER_CLACOP_FUNC(OP_ROWMAX_COMBINE_AXIS_SINGLE, Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE, ExecuteOpReduce<Opcode::OP_ROWMAX_COMBINE_AXIS_SINGLE>);

void ExecuteOpCast(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    CastMode mode = static_cast<CastMode>(ctx->op->GetIntAttribute(OP_ATTR_PREFIX + "mode"));
    Calculator::CalcCast(ret.get(), iop.get(), ret->GetDataType(), mode, ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_CAST, Opcode::OP_CAST, ExecuteOpCast);

template <Opcode opcode>
void ExecuteOpUnary(ExecuteOperationContext *ctx) {
    static_assert(
        opcode == Opcode::OP_EXP || opcode == Opcode::OP_SQRT || opcode == Opcode::OP_ABS, "Invalid opcode");
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    switch (opcode) {
        case Opcode::OP_EXP: Calculator::CalcExp(ret.get(), iop.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_SQRT: Calculator::CalcSqrt(ret.get(), iop.get(), ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_ABS: Calculator::CalcAbs(ret.get(), iop.get(), ctx->opInter->GetPoolPtr()); break;
        default: ASSERT(false);
    }
}
REGISTER_CLACOP_FUNC(OP_EXP, Opcode::OP_EXP, ExecuteOpUnary<Opcode::OP_EXP>);
REGISTER_CLACOP_FUNC(OP_SQRT, Opcode::OP_SQRT, ExecuteOpUnary<Opcode::OP_SQRT>);
REGISTER_CLACOP_FUNC(OP_ABS, Opcode::OP_ABS, ExecuteOpUnary<Opcode::OP_ABS>);

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
        Calculator::CalcExpand(oop.get(), iop.get(), axis, ctx->opInter->GetPoolPtr());
    } else {
        Calculator::CalcCopy(oop.get(), iop.get(), ctx->opInter->GetPoolPtr());
    }
}
REGISTER_CLACOP_FUNC(OP_EXPAND, Opcode::OP_EXPAND, ExecuteOpExpand);

void ExecuteOpTransposeDataMove(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);

    std::vector<int> axises = ctx->op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
    auto oopCopy = oop;
    if (std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute())) {
        auto copyoutAttr = std::dynamic_pointer_cast<CopyOpAttribute>(ctx->op->GetOpAttribute());
        std::vector<int> shape = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyoutAttr->GetShape());
        std::vector<int> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyoutAttr->GetToOffset());
        oopCopy = oop->View(shape, toOffset);
    }
    std::sort(axises.begin(), axises.end());
    ASSERT(axises[0] + 1 == axises[1]);
    Calculator::CalcTransposeAdjDim(oopCopy.get(), iop.get(), axises[0], ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_TRANSPOSE_MOVEOUT, Opcode::OP_TRANSPOSE_MOVEOUT, ExecuteOpTransposeDataMove);

void ExecuteOpTranspose(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() <= SIZE_TWO);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto oop = ctx->ooperandInplaceDataViewList->at(0);
    auto iop = ctx->ioperandDataViewList->at(0);
    auto axises = ctx->op->GetVectorIntAttribute(OP_ATTR_PREFIX + "shape");
    std::sort(axises.begin(), axises.end());
    ASSERT(axises[0] + 1 == axises[1]);
    Calculator::CalcTransposeAdjDim(oop.get(), iop.get(), axises[0], ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_TRANSPOSE_VNCHWCONV, Opcode::OP_TRANSPOSE_VNCHWCONV, ExecuteOpTranspose);

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
        std::vector<int> toOffset = ctx->opInter->EvaluateOpImmediate(ctx->frame, copyoutAttr->GetToOffset());
        oopCopy = oop->View(shape, toOffset);
    }
    Calculator::CalcIndexCopy(oopCopy.get(), src.get(), index.get(), dst.get(), axis, ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_INDEX_OUTCAST, Opcode::OP_INDEX_OUTCAST, ExecuteOpIndexOutcast);

void ExecuteOpReduceAcc(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    auto &ret = ctx->ooperandInplaceDataViewList->at(0);

    std::vector<LogicalTensorData *> dataViewList(ctx->ioperandDataViewList->size());
    for (size_t k = 0; k < ctx->ioperandDataViewList->size(); k++) {
        dataViewList[k] = ctx->ioperandDataViewList->at(k).get();
    }

    Calculator::CalcReduceAcc(ret.get(), &dataViewList, ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_REDUCE_ACC, Opcode::OP_REDUCE_ACC, ExecuteOpReduceAcc);

void ExecuteOpReshape(ExecuteOperationContext *ctx) {
    ASSERT(ctx->ooperandInplaceDataViewList->size() == 1);
    ASSERT(ctx->ioperandDataViewList->size() == 1);
    auto &oop = ctx->ooperandInplaceDataViewList->at(0);
    auto &iop = ctx->ioperandDataViewList->at(0);
    Calculator::CalcReshape(oop.get(), iop.get(), ctx->opInter->GetPoolPtr());
}
REGISTER_CLACOP_FUNC(OP_RESHAPE, Opcode::OP_RESHAPE, ExecuteOpReshape);

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
    auto element = scalarVal.HasValue() ? npu::tile_fwk::AnyCast<Element>(scalarVal) : Element(DT_FP32, 0.0f);
    bool reverse = ctx->op->GetBoolAttribute(OP_ATTR_PREFIX + "reverseOperand");
    ASSERT(ret->GetDataType() == DT_FP32);
    switch (opcode) {
        case Opcode::OP_ADDS: Calculator::CalcAddS(ret.get(), lhs.get(), &element, reverse, ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_SUBS: Calculator::CalcSubS(ret.get(), lhs.get(), &element, reverse, ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_MULS: Calculator::CalcMulS(ret.get(), lhs.get(), &element, reverse, ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_DIVS: Calculator::CalcDivS(ret.get(), lhs.get(), &element, reverse, ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_S_MAXS: Calculator::CalcMaxS(ret.get(), lhs.get(), &element, reverse, ctx->opInter->GetPoolPtr()); break;
        case Opcode::OP_S_MINS: Calculator::CalcMinS(ret.get(), lhs.get(), &element, reverse, ctx->opInter->GetPoolPtr()); break;
        default: ASSERT(false);
    }
}
REGISTER_CLACOP_FUNC(OP_ADDS, Opcode::OP_ADDS, ExecuteOpBinaryScalar<Opcode::OP_ADDS>);
REGISTER_CLACOP_FUNC(OP_SUBS, Opcode::OP_SUBS, ExecuteOpBinaryScalar<Opcode::OP_SUBS>);
REGISTER_CLACOP_FUNC(OP_MULS, Opcode::OP_MULS, ExecuteOpBinaryScalar<Opcode::OP_MULS>);
REGISTER_CLACOP_FUNC(OP_DIVS, Opcode::OP_DIVS, ExecuteOpBinaryScalar<Opcode::OP_DIVS>);
REGISTER_CLACOP_FUNC(OP_S_ADDS, Opcode::OP_S_ADDS, ExecuteOpBinaryScalar<Opcode::OP_ADDS>);
REGISTER_CLACOP_FUNC(OP_S_SUBS, Opcode::OP_S_SUBS, ExecuteOpBinaryScalar<Opcode::OP_SUBS>);
REGISTER_CLACOP_FUNC(OP_S_MULS, Opcode::OP_S_MULS, ExecuteOpBinaryScalar<Opcode::OP_MULS>);
REGISTER_CLACOP_FUNC(OP_S_DIVS, Opcode::OP_S_DIVS, ExecuteOpBinaryScalar<Opcode::OP_DIVS>);
REGISTER_CLACOP_FUNC(OP_S_MAXS, Opcode::OP_S_MAXS, ExecuteOpBinaryScalar<Opcode::OP_S_MAXS>);
REGISTER_CLACOP_FUNC(OP_S_MINS, Opcode::OP_S_MINS, ExecuteOpBinaryScalar<Opcode::OP_S_MINS>);
}