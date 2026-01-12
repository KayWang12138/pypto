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
 * \file operation_common.cpp
 * \brief
 */

#include "operation_common.h"

namespace npu::tile_fwk {

inline const std::vector<size_t> &GetShapeLenLimit(const std::string &op) {
    // if the limit of op is not [1, 4], should add here
    static std::unordered_map<std::string, const std::vector<size_t>> op_shape_len_limit = {
        {    "ADD", {1, 4}},
        {   "CAST", {1, 4}},
        {"DEFAULT", {1, 4}}
    };
    if (op_shape_len_limit.find(op) == op_shape_len_limit.end()) {
        return op_shape_len_limit.at("DEFAULT");
    }
    return op_shape_len_limit.at(op);
}

void CheckTensorShape(const LogicalTensorPtr &tensor, const std::string &op) {
    auto shape = tensor->shape;
    // valid input dims must in [1, 4]
    auto shape_len_limit = GetShapeLenLimit(op);
    if (shape.size() < shape_len_limit[0] || shape.size() > shape_len_limit[1]) {
        ASSERT(false && "The dims of tensor out of range.");
    }
    size_t shapeSize = 1;
    for (const auto &value : shape) {
        if (value > INT32_MAX) {
            ASSERT(false && "The dim value of tensor must less than or equal to INT32_MAX(2,147,483,647)");
        }
        shapeSize *= static_cast<size_t>(value);
        if (shapeSize > INT32_MAX) {
            ASSERT(false && "The shape size of tensor must less than or equal to INT32_MAX(2,147,483,647)");
        }
    }
}

using DTypeCheckSet = std::tuple<std::vector<std::vector<DataType>>, bool, std::vector<std::vector<DataType>>, bool>;
inline const DTypeCheckSet &GetSupportDTypeSet(const std::string &op) {
    static const common_support_dtype = {DT_FP32, DT_FP16, DT_BF16};
    static const scalar_dtype = {DT_DOUBLE, DT_FP32, DT_INT32, DT_BOOL};
    // inputs dtypes..., input dtype consistent, output dtypes..., output dtype consistent with first input
    static const std::unordered_map<std::string, DTypeCheckSet> op_2_dtype = {
        {"ADD", ({common_support_dtype, common_support_dtype}, true, {common_support_dtype}, true)},
        {"SUB", ({common_support_dtype, common_support_dtype}, true, {common_support_dtype}, true)},
        {"MUL", ({common_support_dtype, common_support_dtype}, true, {common_support_dtype}, true)},
        {"DIV", ({common_support_dtype, common_support_dtype}, true, {common_support_dtype}, true)},
        {"ADDS", ({common_support_dtype, scalar_dtype}, false, {common_support_dtype}, true)},
        {"SUBS", ({common_support_dtype, scalar_dtype}, false, {common_support_dtype}, true)},
        {"MULS", ({common_support_dtype, scalar_dtype}, false, {common_support_dtype}, true)},
        {"DIVS", ({common_support_dtype, scalar_dtype}, false, {common_support_dtype}, true)},
        {"VEC_DUP", ({common_support_dtype, scalar_dtype}, false, {common_support_dtype}, true)},
        {"CAST",
         ({{DT_FP32, DT_INT32, DT_FP16, DT_BF16}}, false, {{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT8}}, false)},
        {"SQRT", ({common_support_dtype}, false, {common_support_dtype}, true)},
        {"RSQRT", ({common_support_dtype}, false, {common_support_dtype}, true)},
        {"EXP", ({common_support_dtype}, false, {common_support_dtype}, true)},
        {"ROWMAX_SINGLE", ({common_support_dtype}, false, {common_support_dtype}, true)},
        {"ROWMIN_SINGLE", ({common_support_dtype}, false, {common_support_dtype}, true)},
        {"ROWSUM_SINGLE", ({{DT_FP32}}, false, {{DT_FP32}}, true)},
        {"TOPK", ({{DT_FP32}}, false, {{DT_FP32}}, true)},
        {"SCATTER", ({common_support_dtype, common_support_dtype}, true, {common_support_dtype}, true)},
        {"SCATTER_ELEMENT", ({common_support_dtype, common_support_dtype}, true, {common_support_dtype}, true)},
        {"SCATTER_UPDATE", ({common_support_dtype, common_support_dtype}, true, {common_support_dtype}, true)},
        {"TRANSPOSE_MOVEIN", ({common_support_dtype}, false, {common_support_dtype}, false)},
        {"TRANSPOSE_MOVEOUT", ({common_support_dtype}, false, {common_support_dtype}, false)},
        {"TRANSPOSE_VNCHWCONV", ({common_support_dtype}, false, {common_support_dtype}, false)},
        {"WHERE_TT",
         ({common_support_dtype, common_support_dtype, common_support_dtype}, false, {common_support_dtype}, false)},
        {"WHERE_TS",
         ({common_support_dtype, common_support_dtype, common_support_dtype}, false, {common_support_dtype}, false)},
        {"WHERE_ST",
         ({common_support_dtype, common_support_dtype, common_support_dtype}, false, {common_support_dtype}, false)},
        {"WHERE_SS",
         ({common_support_dtype, common_support_dtype, common_support_dtype}, false, {common_support_dtype}, false)},
        {"LN", ({common_support_dtype}, false, {common_support_dtype}, true)},
        {"LOGICALAND", ({{DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8},
                            {DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8}},
         false, {{DT_BOOL}}, false)},
        {"LOGICALNOT", ({{DT_FP32, DT_FP16, DT_BF16, DT_BOOL, DT_INT8, DT_UINT8}}, false, {{DT_BOOL}}, false)},
        {"RANGE",
         ({{DT_UINT64, DT_UINT32}, {DT_FP32, DT_INT32, DT_FP16, DT_BF16}, {DT_FP32, DT_INT32, DT_FP16, DT_BF16}},
         false, {{DT_FP32, DT_INT32, DT_FP16, DT_BF16}}, false)},
        {"CMP", ({common_support_dtype, common_support_dtype}, true, {{DT_BOOL}}, false)},
        {"NEG", ({{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, false,
         {{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, true)},
        {"ABS", ({common_support_dtype}, false, {common_support_dtype}, true)},
        {"GATHER", ({{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}, {DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}},
         true, {{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, true)},
        {"GATHERELEMENT",
         ({{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}, {DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, true,
         {{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, true)},
        {"MAXIMUM", ({{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}, {DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}},
         true, {{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, true)},
        {"MINIMUM", ({{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}, {DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}},
         true, {{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, true)},
        {"MAXS", ({{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}, {DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}},
         true, {{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, true)},
        {"MINS", ({{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}, {DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}},
         true, {{DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16}}, true)},
        {"CONCAT", ({{DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT8}, {DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT8}},
         true, {{DT_FP32, DT_FP16, DT_BF16, DT_INT16, DT_INT8}}, true)},
        {"EXPAND", {DT_FP32, DT_INT32, DT_UINT32, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_UINT8, DT_INT8}},
        {"EXPAND",
         ({{DT_FP32, DT_INT32, DT_UINT32, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_UINT8, DT_INT8},
                 {DT_UINT64, DT_UINT32}},
         false, {{DT_FP32, DT_INT32, DT_UINT32, DT_FP16, DT_BF16, DT_INT16, DT_UINT16, DT_UINT8, DT_INT8}},
         true)},
        // {"INDEX_ADD", {DT_FP32, DT_INT32, DT_FP16, DT_BF16, DT_INT16, DT_INT8}},
        // {"ONEHOT", {DT_INT64, DT_INT32, DT_INT16, DT_INT8}},
        // {"INDEX_PUT", {DT_FP32, DT_INT32, DT_FP16, DT_INT16}},
    };
    if (op_2_dtype.find(op) == op_2_dtype.end()) {
        ASSERT(false) << "Operation " + op + " has not set dtype checker yet.";
    }
    return op_2_dtype.at(op);
}

inline const std::vector<DataType> &GetInputDtype(const std::string &op, size_t index = 0) {
    const auto &inputs_dtype = std::get<0>(GetSupportDTypeSet(op));
    auto size = inputs_dtype.size();
    ASSERT(size > index) << "Expect [0, " + std::to_string(size) + "), but got " + std::to_string(index);
    return inputs_dtype[index];
}

inline const std::vector<DataType> &GetOutputDtype(const std::string &op, size_t index = 0) {
    const auto &outputs_dtype = std::get<2>(GetSupportDTypeSet(op));
    auto size = outputs_dtype.size();
    ASSERT(size > index) << "Expect [0, " + std::to_string(size) + "), but got " + std::to_string(index);
    return outputs_dtype[index];
}

inline bool NeedInputDTypeSame(const std::string &op) {
    return std::get<1>(GetSupportDTypeSet(op));
}

inline bool NeedFirstOutputDTypeSameFirstInput(const std::string &op) {
    return std::get<3>(GetSupportDTypeSet(op));
}

void CheckTensorDType(const LogicalTensorPtr &tensor, const std::vector<DataType> &dtypes) {
    const auto dtype = tensor->GetRawTensor()->GetDataType();
    if (std::find(dtypes.begin(), dtypes.end(), dtype) == dtypes.end()) {
        ASSERT(false) << "Operation " + op + " not support " + DataType2String(dtype) + " yet.";
    }
}

void CheckOperationInputsDType(const std::vector<LogicalTensorPtr> &inputs, const std::string &op) {
    for (size_t index = 0; index < inputs.size(); ++index) {
        CheckTensorDType(inputs[index], GetInputDtype(op, index));
        if (NeedInputDTypeSame(op) &&
            inputs[index]->GetRawTensor()->GetDataType() != inputs[0]->GetRawTensor()->GetDataType()) {
            ASSERT(false) << "Operation " + op + " inputs dtype check fail.";
        }
    }
}

void CheckOperationOutputsDType(const std::vector<LogicalTensorPtr> &outputs, const std::string &op) {
    for (size_t index = 0; index < outputs.size(); ++index) {
        CheckTensorDType(outputs[index], GetOutputDtype(op, index));
    }
}

std::vector<int> GetBroadCastShape(LogicalTensorPtr &operand1, LogicalTensorPtr &operand2) {
    std::vector<int64_t> opShape1(operand1->shape);
    std::vector<int64_t> opShape2(operand2->shape);
    auto maxShapeSize = std::max(opShape1.size(), opShape2.size());
    if (opShape1.size() != maxShapeSize) {
        opShape1.insert(opShape1.begin(), maxShapeSize - opShape1.size(), 1);
    }
    if (opShape2.size() != maxShapeSize) {
        opShape2.insert(opShape2.begin(), maxShapeSize - opShape2.size(), 1);
    }
    std::vector<int> broadCastShape(maxShapeSize, 0);
    for (size_t i = 0; i < maxShapeSize; i++) {
        broadCastShape[i] = std::max(opShape1[i], opShape2[i]);
    }
    return broadCastShape;
}

std::vector<int> GetBroadcastAxes(const Shape &shape1, const Shape &shape2) {
    Shape shape1_(shape1), shape2_(shape2);
    std::vector<int> result = {};
    auto maxShapeSize = std::max(shape1_.size(), shape2_.size());
    if (shape1_.size() != maxShapeSize) {
        shape1_.insert(shape1_.begin(), maxShapeSize - shape1_.size(), 1);
    }
    if (shape2_.size() != maxShapeSize) {
        shape2_.insert(shape2_.begin(), maxShapeSize - shape2_.size(), 1);
    }
    for (size_t i = 0; i < shape1_.size(); i++) {
        if (shape1_[i] != shape2_[i] && (shape1_[i] == 1 || shape2_[i] == 1)) {
            result.push_back(i);
        }
    }
    return result;
}
} // namespace npu::tile_fwk