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
 * \file vjp_comparison.cpp
 * \brief VJP rules for comparison and conditional operations.
 */

#include "../vjp_registry.h"
#include "tilefwk/tilefwk_op.h"
#include "interface/inner/element.h"

namespace npu::tile_fwk {

namespace {

LogicalTensorPtr CreateTensor(VJPContext& ctx, const Shape& shape, DataType dtype) {
    if (ctx.function == nullptr) {
        return nullptr;
    }
    return std::make_shared<LogicalTensor>(*ctx.function, dtype, shape);
}

LogicalTensorPtr CreateZerosTensor(VJPContext& ctx, const Shape& shape, DataType dtype) {
    if (ctx.function == nullptr) {
        return nullptr;
    }
    auto output = std::make_shared<LogicalTensor>(*ctx.function, dtype, shape);

    Operation& dupOp = ctx.function->AddRawOperation(Opcode::OP_VEC_DUP, {}, {output});

    Element scalarValue;
    switch (dtype) {
        case DataType::DT_FP32:
            scalarValue = Element(DataType::DT_FP32, 0.0f);
            break;
        case DataType::DT_FP16:
            scalarValue = Element(DataType::DT_FP16, 0.0);
            break;
        case DataType::DT_BF16:
            scalarValue = Element(DataType::DT_BF16, 0.0);
            break;
        default:
            scalarValue = Element(DataType::DT_FP32, 0.0f);
            break;
    }
    dupOp.SetAttr(OpAttributeKey::scalar, scalarValue);

    std::vector<int64_t> shapeVec(shape.begin(), shape.end());
    dupOp.SetAttr(OP_ATTR_PREFIX + "shape", shapeVec);

    return output;
}

LogicalTensorPtr CreateWhereTTOp(VJPContext& ctx, LogicalTensorPtr condition,
                                  LogicalTensorPtr x, LogicalTensorPtr y) {
    if (condition == nullptr || x == nullptr || y == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    auto output = CreateTensor(ctx, x->GetShape(), x->Datatype());
    if (output == nullptr) {
        return nullptr;
    }

    // Only pass single output to comply with GraphCheck assertion
    // tempTensor is optional and will be created during tiling if needed
    Operation& whereOp = ctx.function->AddRawOperation(Opcode::OP_WHERE_TT,
                                                        {condition, x, y}, {output});
    whereOp.SetAttr(OP_ATTR_PREFIX + "whereBitMode", static_cast<int64_t>(0));

    return output;
}

LogicalTensorPtr CreateCmpOp(VJPContext& ctx, LogicalTensorPtr a, LogicalTensorPtr b, OpType cmpType) {
    if (a == nullptr || b == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    auto output = CreateTensor(ctx, a->GetShape(), DataType::DT_BOOL);
    if (output == nullptr) {
        return nullptr;
    }

    Operation& cmpOp = ctx.function->AddRawOperation(Opcode::OP_CMP, {a, b}, {output});
    cmpOp.SetAttr(OP_ATTR_PREFIX + "cmp_operation", static_cast<int64_t>(cmpType));
    cmpOp.SetAttr(OP_ATTR_PREFIX + "cmp_mode", static_cast<int64_t>(OutType::BOOL));

    return output;
}

/**
 * VJP for Where: output = condition ? x : y
 *
 * d_x = condition ? dOut : 0
 * d_y = condition ? 0 : dOut
 */
VJPResult vjp_where_tt(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto condition = ctx.GetSaved("input");
    if (condition == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto x = ctx.GetSaved("other");
    auto y = ctx.GetSaved("input2");

    Shape outputShape(dOut->GetShape().begin(), dOut->GetShape().end());
    DataType dtype = dOut->Datatype();

    auto zeros = CreateZerosTensor(ctx, outputShape, dtype);
    if (zeros == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto dX = CreateWhereTTOp(ctx, condition, dOut, zeros);
    auto dY = CreateWhereTTOp(ctx, condition, zeros, dOut);

    return {{"input", nullptr}, {"other", dX}, {"input2", dY}};
}

/**
 * VJP for Maximum: output = max(x, y)
 *
 * d_x = (x >= y) ? dOut : 0
 * d_y = (x < y) ? dOut : 0  = (x >= y) ? 0 : dOut
 */
VJPResult vjp_maximum(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto x = ctx.GetSaved("input");
    auto y = ctx.GetSaved("other");
    if (x == nullptr || y == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    Shape outputShape(dOut->GetShape().begin(), dOut->GetShape().end());
    DataType dtype = dOut->Datatype();

    auto mask = CreateCmpOp(ctx, x, y, OpType::GE);
    if (mask == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto zeros = CreateZerosTensor(ctx, outputShape, dtype);
    if (zeros == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto dX = CreateWhereTTOp(ctx, mask, dOut, zeros);
    auto dY = CreateWhereTTOp(ctx, mask, zeros, dOut);

    return {{"input", dX}, {"other", dY}};
}

/**
 * VJP for Minimum: output = min(x, y)
 *
 * d_x = (x <= y) ? dOut : 0
 * d_y = (x > y) ? dOut : 0  = (x <= y) ? 0 : dOut
 */
VJPResult vjp_minimum(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto x = ctx.GetSaved("input");
    auto y = ctx.GetSaved("other");
    if (x == nullptr || y == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    Shape outputShape(dOut->GetShape().begin(), dOut->GetShape().end());
    DataType dtype = dOut->Datatype();

    auto mask = CreateCmpOp(ctx, x, y, OpType::LE);
    if (mask == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto zeros = CreateZerosTensor(ctx, outputShape, dtype);
    if (zeros == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto dX = CreateWhereTTOp(ctx, mask, dOut, zeros);
    auto dY = CreateWhereTTOp(ctx, mask, zeros, dOut);

    return {{"input", dX}, {"other", dY}};
}

} // anonymous namespace

REG_VJP(OP_WHERE_TT, vjp_where_tt);
REG_VJP(OP_MAXIMUM, vjp_maximum);
REG_VJP(OP_MINIMUM, vjp_minimum);

} // namespace npu::tile_fwk
