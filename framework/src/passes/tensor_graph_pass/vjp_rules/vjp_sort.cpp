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
 * \file vjp_sort.cpp
 * \brief VJP rules for sorting operations (topk).
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

LogicalTensorPtr CreateScatterOp(VJPContext& ctx, LogicalTensorPtr self, LogicalTensorPtr index,
                                  LogicalTensorPtr src, int64_t axis) {
    if (self == nullptr || index == nullptr || src == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    auto output = CreateTensor(ctx, self->GetShape(), self->Datatype());
    if (output == nullptr) {
        return nullptr;
    }

    Operation& scatterOp = ctx.function->AddRawOperation(Opcode::OP_SCATTER,
                                                          {self, index, src}, {output});
    scatterOp.SetAttr(OP_ATTR_PREFIX + "axis", axis);

    return output;
}

/**
 * VJP for TopK: (values, indices) = topk(input, k, axis)
 *
 * d_input = scatter(zeros, indices, d_values)
 *
 * The gradient of the values output is scattered back to the original
 * positions in the input tensor using the indices output.
 *
 * Note: TopK has two outputs (values, indices). We only compute gradient
 * for values (output), not for indices (output1) since indices are not
 * differentiable.
 */
VJPResult vjp_topk(VJPContext& ctx) {
    auto dValues = ctx.GetOutputGrad("output");
    if (dValues == nullptr) {
        return {{"input", nullptr}};
    }

    auto input = ctx.GetSaved("input");
    auto indices = ctx.GetSaved("output1");
    if (input == nullptr || indices == nullptr) {
        return {{"input", nullptr}};
    }

    int64_t axis = ctx.GetAttr<int64_t>(OP_ATTR_PREFIX + "axis", -1);
    if (axis < 0) {
        axis += static_cast<int64_t>(input->GetShape().size());
    }

    Shape inputShape = input->GetShape();
    DataType dtype = input->Datatype();

    auto zeros = CreateZerosTensor(ctx, inputShape, dtype);
    if (zeros == nullptr) {
        return {{"input", nullptr}};
    }

    auto dInput = CreateScatterOp(ctx, zeros, indices, dValues, axis);

    return {{"input", dInput}};
}

} // anonymous namespace

REG_VJP(OP_TOPK, vjp_topk);

} // namespace npu::tile_fwk
