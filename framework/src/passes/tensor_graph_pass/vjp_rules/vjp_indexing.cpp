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
 * \file vjp_indexing.cpp
 * \brief VJP rules for indexing operations (gather, scatter, index_add).
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

LogicalTensorPtr CreateGatherOp(VJPContext& ctx, LogicalTensorPtr params, LogicalTensorPtr indices,
                                 int64_t axis, const Shape& outputShape) {
    if (params == nullptr || indices == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    auto output = CreateTensor(ctx, outputShape, params->Datatype());
    if (output == nullptr) {
        return nullptr;
    }

    Operation& gatherOp = ctx.function->AddRawOperation(Opcode::OP_GATHER, {params, indices}, {output});
    gatherOp.SetAttr(OP_ATTR_PREFIX + "axis", axis);

    return output;
}

LogicalTensorPtr CreateIndexAddOp(VJPContext& ctx, LogicalTensorPtr self, LogicalTensorPtr src,
                                   LogicalTensorPtr index, int64_t axis) {
    if (self == nullptr || src == nullptr || index == nullptr || ctx.function == nullptr) {
        return nullptr;
    }

    auto output = CreateTensor(ctx, self->GetShape(), self->Datatype());
    if (output == nullptr) {
        return nullptr;
    }

    Operation& indexAddOp = ctx.function->AddRawOperation(Opcode::OP_INDEX_ADD,
                                                           {self, src, index}, {output});
    indexAddOp.SetAttr(OP_ATTR_PREFIX + "axis", axis);

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
 * VJP for Gather: output[i] = input[index[i]]
 *
 * d_input = index_add(zeros, index, d_out)
 *
 * Gradient only flows to positions specified by index.
 */
VJPResult vjp_gather(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto params = ctx.GetSaved("input");
    auto indices = ctx.GetSaved("other");
    if (params == nullptr || indices == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    int64_t axis = ctx.GetAttr<int64_t>(OP_ATTR_PREFIX + "axis", 0);

    Shape paramsShape = params->GetShape();
    DataType dtype = params->Datatype();

    auto zeros = CreateZerosTensor(ctx, paramsShape, dtype);
    if (zeros == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto dParams = CreateIndexAddOp(ctx, zeros, dOut, indices, axis);

    return {{"input", dParams}, {"other", nullptr}};
}

/**
 * VJP for GatherElement: output[i][j] = input[index[i][j]][j]
 *
 * d_input = scatter(zeros, index, d_out)
 */
VJPResult vjp_gather_element(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto params = ctx.GetSaved("input");
    auto indices = ctx.GetSaved("other");
    if (params == nullptr || indices == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    int64_t axis = ctx.GetAttr<int64_t>(OP_ATTR_PREFIX + "axis", 0);

    Shape paramsShape = params->GetShape();
    DataType dtype = params->Datatype();

    auto zeros = CreateZerosTensor(ctx, paramsShape, dtype);
    if (zeros == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}};
    }

    auto dParams = CreateScatterOp(ctx, zeros, indices, dOut, axis);

    return {{"input", dParams}, {"other", nullptr}};
}

/**
 * VJP for Scatter: output[index[i]] = src[i]
 *
 * d_src = gather(d_out, index)
 * d_self = d_out with scattered positions zeroed (complex, return d_out for now)
 */
VJPResult vjp_scatter(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}, {"input2", nullptr}};
    }

    auto indices = ctx.GetSaved("other");
    auto src = ctx.GetSaved("input2");
    if (indices == nullptr || src == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}, {"input2", nullptr}};
    }

    int64_t axis = ctx.GetAttr<int64_t>(OP_ATTR_PREFIX + "axis", 0);

    Shape srcShape = src->GetShape();

    auto dSrc = CreateGatherOp(ctx, dOut, indices, axis, srcShape);

    return {{"input", dOut}, {"other", nullptr}, {"input2", dSrc}};
}

/**
 * VJP for ScatterElement: output[index[i][j]][j] = src[i][j]
 *
 * d_src = gather_element(d_out, index)
 */
VJPResult vjp_scatter_element(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}, {"input2", nullptr}};
    }

    auto self = ctx.GetSaved("input");
    auto indices = ctx.GetSaved("other");
    auto src = ctx.GetSaved("input2");
    if (self == nullptr || indices == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}, {"input2", nullptr}};
    }

    int64_t axis = ctx.GetAttr<int64_t>(OP_ATTR_PREFIX + "axis", 0);

    Shape srcShape = src != nullptr ? src->GetShape() : indices->GetShape();

    auto dSrc = CreateGatherOp(ctx, dOut, indices, axis, srcShape);

    return {{"input", dOut}, {"other", nullptr}, {"input2", dSrc}};
}

/**
 * VJP for IndexAdd: output = self + scatter(zeros, index, src)
 *
 * d_self = d_out
 * d_src = gather(d_out, index)
 */
VJPResult vjp_index_add(VJPContext& ctx) {
    auto dOut = ctx.GetOutputGrad("output");
    if (dOut == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}, {"input2", nullptr}};
    }

    auto indices = ctx.GetSaved("input2");
    auto src = ctx.GetSaved("other");
    if (indices == nullptr || src == nullptr) {
        return {{"input", nullptr}, {"other", nullptr}, {"input2", nullptr}};
    }

    int64_t axis = ctx.GetAttr<int64_t>(OP_ATTR_PREFIX + "axis", 0);

    Shape srcShape = src->GetShape();

    auto dSrc = CreateGatherOp(ctx, dOut, indices, axis, srcShape);

    return {{"input", dOut}, {"other", dSrc}, {"input2", nullptr}};
}

} // anonymous namespace

REG_VJP(OP_GATHER, vjp_gather);
REG_VJP(OP_GATHER_ELEMENT, vjp_gather_element);
REG_VJP(OP_SCATTER, vjp_scatter);
REG_VJP(OP_SCATTER_ELEMENT, vjp_scatter_element);
REG_VJP(OP_INDEX_ADD, vjp_index_add);

} // namespace npu::tile_fwk
