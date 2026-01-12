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

inline const std::vector<size_t> &GetShapeLenLimit(const Opcode op) {
    // if the limit of op is not [1, 4], should add here
    static std::unordered_map<const Opcode, const std::vector<size_t>> op_shape_len_limit = {
        {    Opcode::OP_ADD, {1, 4}},
        {   Opcode::OP_CAST, {1, 4}},
        {Opcode::OP_UNKNOWN, {1, 4}}
    };
    if (op_shape_len_limit.find(op) == op_shape_len_limit.end()) {
        return op_shape_len_limit.at(Opcode::OP_UNKNOWN);
    }
    return op_shape_len_limit.at(op);
}

void CheckTensorShape(const LogicalTensorPtr &tensor, const Opcode op) {
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

using InputsDTypeSet = std::vector<std::vector<DataType>>;
using DTypeCheckSet = std::tuple<InputsDTypeSet, std::vector<std::vector<int>>>;
inline const DTypeCheckSet &GetSupportDTypeSet(const Opcode op) {
    static const dtype_bool = {DT_BOOL};
    static const dtype_8 = {DT_INT8, DT_UINT8};
    static const dtype_16 = {DT_INT16, DT_UINT16};
    static const dtype_32 = {DT_INT16, DT_UINT16};
    static const dtype_int_32_16 = {DT_INT32, DT_INT16};
    static const dtype_base = {DT_FP32, DT_FP16, DT_BF16};
    static const dtype_binary = dtype_base + dtype_int_32_16;
    static const dtype_32_16 = dtype_base + dtype_32 + dtype_16;
    static const dtype_uint8_bool = {DT_UINT8, DT_BOOL};
    static const dtype_int_64_32_16 = {DT_INT64, DT_INT32, DT_INT16};
    static const dtype_int = {DT_INT64, DT_INT32, DT_INT16, DT_INT8};
    static const dtype_fp32 = {DT_FP32};
    static const dtype_int64_32 = {DT_INT64, DT_INT32};
    static const dtype_logical = {DT_FP32, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_BOOL};
    static const dtype_base_int32 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32};
    static const dtype_base_int32_16 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const dtype_base_int32_16_8 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8};
    static const dtype_base_int_uint = dtype_base + dtype_32 + dtype_16 + dtype_8;
    static const dtype_base_int_uint_bool = dtype_base + dtype_32 + dtype_16 + dtype_8 + dtype_bool;
    // inputs dtypes..., input dtype consistent, output dtypes..., output dtype consistent with first input
    static const std::unordered_map<std::string, DTypeCheckSet> op_2_dtype = {
        // unary
        {                Opcode::OP_ABS,                                                   ({dtype_base},       {})},
        {                Opcode::OP_EXP,                                                   ({dtype_base},       {})},
        {                 Opcode::OP_LN,                                                   ({dtype_base},       {})},
        {               Opcode::OP_SQRT,                                                   ({dtype_base},       {})},
        {              Opcode::OP_RSQRT,                                                   ({dtype_base},       {})},
        // one input
        {               Opcode::OP_CAST,                                             ({dtype_base_int32},       {})},
        {            Opcode::OP_BITSORT,                                                   ({dtype_fp32},       {})},
        {            Opcode::OP_EXTRACT,                                                   ({dtype_fp32},       {})},
        {            Opcode::OP_MRGSORT,                                                   ({dtype_fp32},       {})},
        {                Opcode::OP_NEG,                                          ({dtype_base_int32_16},       {})},
        {      Opcode::OP_ROWMAX_SINGLE,                                                   ({dtype_base},       {})},
        {      Opcode::OP_ROWMIN_SINGLE,                                                   ({dtype_base},       {})},
        {      Opcode::OP_ROWSUM_SINGLE,                                                   ({dtype_base},       {})},
        {         Opcode::OP_ROWMAXLINE,                                                 ({dtype_binary},       {})},
        {         Opcode::OP_ROWMINLINE,                                                 ({dtype_binary},       {})},
        {         Opcode::OP_ROWSUMLINE,                                                 ({dtype_binary},       {})},
        {               Opcode::OP_TOPK,                                                   ({dtype_fp32},       {})},
        {   Opcode::OP_TRANSPOSE_MOVEIN,                                                  ({dtype_32_16},       {})},
        {  Opcode::OP_TRANSPOSE_MOVEOUT,                                                  ({dtype_32_16},       {})},
        {Opcode::OP_TRANSPOSE_VNCHWCONV,                                                  ({dtype_32_16},       {})},
        // binary
        {                Opcode::OP_ADD,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {                Opcode::OP_SUB,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {                Opcode::OP_MUL,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {                Opcode::OP_DIV,                                       ({dtype_base, dtype_base}, {{0, 1}})},
        {               Opcode::OP_ADDS,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_SUBS,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_MULS,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_DIVS,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_MAXS,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_MINS,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {            Opcode::OP_MAXIMUM,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        {            Opcode::OP_MINIMUM,                                   ({dtype_binary, dtype_binary}, {{0, 1}})},
        // two input
        {                Opcode::OP_CMP,                                       ({dtype_base, dtype_base},   {0, 1})},
        {               Opcode::OP_CMPS,                                       ({dtype_base, dtype_base},   {0, 1})},
        {             Opcode::OP_GATHER,                    ({dtype_base_int32_16_8, dtype_int_64_32_16},       {})},
        {     Opcode::OP_GATHER_ELEMENT,                          ({dtype_base_int32_16, dtype_int64_32},       {})},
        // {Opcode::OP_VEC_DUP, ({dtype_base, scalar_dtype}, false, {dtype_base}, true)},
        {            Opcode::OP_SCATTER,                       ({dtype_base, dtype_int64_32, dtype_base},   {0, 2})},
        {    Opcode::OP_SCATTER_ELEMENT,                                   ({dtype_base, dtype_int64_32},       {})},
        {      Opcode::OP_INDEX_OUTCAST, ({dtype_base_int32_16, dtype_int_64_32_16, dtype_base_int32_16},   {0, 2})},
        {           Opcode::OP_WHERE_TT,                     ({dtype_uint8_bool, dtype_base, dtype_base},   {1, 2})},
        {           Opcode::OP_WHERE_TS,                     ({dtype_uint8_bool, dtype_base, dtype_base},   {1, 2})},
        {           Opcode::OP_WHERE_ST,                     ({dtype_uint8_bool, dtype_base, dtype_base},   {1, 2})},
        {           Opcode::OP_WHERE_SS,                     ({dtype_uint8_bool, dtype_base, dtype_base},   {1, 2})},
        {         Opcode::OP_LOGICALAND,                                 ({dtype_logical, dtype_logical},   {0, 1})},
        {         Opcode::OP_LOGICALNOT,                                                ({dtype_logical},       {})},
        {              Opcode::OP_RANGE,                     ({dtype_binary, dtype_binary, dtype_binary},       {})},
        {      Opcode::OP_REGISTER_COPY,                 ({dtype_base_int32_16_8, dtype_base_int32_16_8},   {0, 1})},
        {             Opcode::OP_EXPAND,                                     ({dtype_base_int_uint_bool},       {})},
        {          Opcode::OP_INDEX_ADD,     ({dtype_base_int32_16, dtype_base_int32_16, dtype_int64_32},   {0, 1})},
        {             Opcode::OP_ONEHOT,                           ({dtype_int_64_32_16, dtype_int64_32},       {})},
        {          Opcode::OP_INDEX_PUT,                     ({dtype_base_int_uint, dtype_base_int_uint}, {{0, 1}})},
        {            Opcode::OP_CUM_SUM,                                   ({dtype_base, dtype_int64_32},       {})},
    };
    if (op_2_dtype.find(op) == op_2_dtype.end()) {
        ASSERT(false) << "Operation " + op + " has not set dtype checker yet.";
    }
    return op_2_dtype.at(op);
}

inline const std::vector<DataType> &GetInputDtype(const Opcode op, size_t index = 0) {
    const auto &inputs_dtype = std::get<0>(GetSupportDTypeSet(op));
    auto size = inputs_dtype.size();
    ASSERT(size > index) << "Expect [0, " + std::to_string(size) + "), but got " + std::to_string(index);
    return inputs_dtype[index];
}

void CheckTensorDType(const LogicalTensorPtr &tensor, const std::vector<DataType> &dtypes) {
    const auto dtype = tensor->GetRawTensor()->GetDataType();
    if (std::find(dtypes.begin(), dtypes.end(), dtype) == dtypes.end()) {
        ASSERT(false) << "Operation " + op + " not support " + DataType2String(dtype) + " yet.";
    }
}

void CheckOperationDType(const std::vector<LogicalTensorPtr> &inputs, const Opcode op) {
    for (size_t index = 0; index < inputs.size(); ++index) {
        CheckTensorDType(inputs[index], GetInputDtype(op, index));
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