/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
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
            ASSERT(false && "The dim value of tensor must less than or equal to "
                            "INT32_MAX(2,147,483,647)");
        }
        shapeSize *= static_cast<size_t>(value);
        if (shapeSize > INT32_MAX) {
            ASSERT(false && "The shape size of tensor must less than or equal to "
                            "INT32_MAX(2,147,483,647)");
        }
    }
}

using InputsDTypeSet = std::vector<std::vector<DataType>>;
using Constraint = std::tuple<const InputsDTypeSet, const std::vector<std::vector<int>>>;
inline const Constraint &GetSupportDTypeSet(const Opcode op) {
    static const auto dtype_fp32 = {DT_FP32};
    static const auto dtype_u8_b = {DT_UINT8, DT_BOOL};
    static const auto dtype_i64_32 = {DT_INT64, DT_INT32};
    static const auto dtype_base = {DT_FP32, DT_FP16, DT_BF16};
    static const auto dtype_binary = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const auto dtype_32_16 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_UINT32, DT_UINT16};
    static const auto dtype_i64_32_16 = {DT_INT64, DT_INT32, DT_INT16};
    static const auto dtype_i = {DT_INT64, DT_INT32, DT_INT16, DT_INT8};
    static const auto dtype_logical = {DT_FP32, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_BOOL};
    static const auto dtype_base_i32 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32};
    static const auto dtype_b_i3_1 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const auto dtype_b_i3_1_8 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8};
    static const auto dtype_base_i_u = {
        DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, DT_UINT32, DT_UINT16, DT_UINT8};
    static const auto dtype_base_i_u_b = {
        DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, DT_UINT32, DT_UINT16, DT_UINT8, DT_BOOL};
    // inputs dtypes..., input dtype consistent, output dtypes..., output dtype
    // consistent with first input
    static const std::unordered_map<const Opcode, const Constraint> op_2_dtype = {
        // unary
        {                Opcode::OP_ABS,                                  Constraint({dtype_base},       {})},
        {                Opcode::OP_EXP,                                  Constraint({dtype_base},       {})},
        {                 Opcode::OP_LN,                                  Constraint({dtype_base},       {})},
        {               Opcode::OP_SQRT,                                  Constraint({dtype_base},       {})},
        {              Opcode::OP_RSQRT,                                  Constraint({dtype_base},       {})},
        // one input
        {               Opcode::OP_CAST,                              Constraint({dtype_base_i32},       {})},
        {            Opcode::OP_BITSORT,                                  Constraint({dtype_fp32},       {})},
        {            Opcode::OP_EXTRACT,                                  Constraint({dtype_fp32},       {})},
        {            Opcode::OP_MRGSORT,                                  Constraint({dtype_fp32},       {})},
        {                Opcode::OP_NEG,                                Constraint({dtype_b_i3_1},       {})},
        {      Opcode::OP_ROWMAX_SINGLE,                                  Constraint({dtype_base},       {})},
        {      Opcode::OP_ROWMIN_SINGLE,                                  Constraint({dtype_base},       {})},
        {      Opcode::OP_ROWSUM_SINGLE,                                  Constraint({dtype_base},       {})},
        {         Opcode::OP_ROWMAXLINE,                                Constraint({dtype_binary},       {})},
        {         Opcode::OP_ROWMINLINE,                                Constraint({dtype_binary},       {})},
        {         Opcode::OP_ROWSUMLINE,                                Constraint({dtype_binary},       {})},
        {               Opcode::OP_TOPK,                                  Constraint({dtype_fp32},       {})},
        {   Opcode::OP_TRANSPOSE_MOVEIN,                                 Constraint({dtype_32_16},       {})},
        {  Opcode::OP_TRANSPOSE_MOVEOUT,                                 Constraint({dtype_32_16},       {})},
        {Opcode::OP_TRANSPOSE_VNCHWCONV,                                 Constraint({dtype_32_16},       {})},
        // binary
        {                Opcode::OP_ADD,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {                Opcode::OP_SUB,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {                Opcode::OP_MUL,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {                Opcode::OP_DIV,                      Constraint({dtype_base, dtype_base}, {{0, 1}})},
        {               Opcode::OP_ADDS,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_SUBS,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_MULS,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_DIVS,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_MAXS,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {               Opcode::OP_MINS,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {            Opcode::OP_MAXIMUM,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        {            Opcode::OP_MINIMUM,                  Constraint({dtype_binary, dtype_binary}, {{0, 1}})},
        // two input
        {                Opcode::OP_CMP,                      Constraint({dtype_base, dtype_base}, {{0, 1}})},
        {               Opcode::OP_CMPS,                      Constraint({dtype_base, dtype_base}, {{0, 1}})},
        {             Opcode::OP_GATHER,             Constraint({dtype_b_i3_1_8, dtype_i64_32_16},       {})},
        {     Opcode::OP_GATHER_ELEMENT,                  Constraint({dtype_b_i3_1, dtype_i64_32},       {})},
        // {Opcode::OP_VEC_DUP, Constraint({dtype_base, scalar_dtype}, false,
        // {dtype_base}, true)},
        {            Opcode::OP_SCATTER,        Constraint({dtype_base, dtype_i64_32, dtype_base}, {{0, 2}})},
        {    Opcode::OP_SCATTER_ELEMENT,                    Constraint({dtype_base, dtype_i64_32},       {})},
        {      Opcode::OP_INDEX_OUTCAST, Constraint({dtype_b_i3_1, dtype_i64_32_16, dtype_b_i3_1}, {{0, 2}})},
        {           Opcode::OP_WHERE_TT,          Constraint({dtype_u8_b, dtype_base, dtype_base}, {{1, 2}})},
        {           Opcode::OP_WHERE_TS,          Constraint({dtype_u8_b, dtype_base, dtype_base}, {{1, 2}})},
        {           Opcode::OP_WHERE_ST,          Constraint({dtype_u8_b, dtype_base, dtype_base}, {{1, 2}})},
        {           Opcode::OP_WHERE_SS,          Constraint({dtype_u8_b, dtype_base, dtype_base}, {{1, 2}})},
        {         Opcode::OP_LOGICALAND,                Constraint({dtype_logical, dtype_logical}, {{0, 1}})},
        {         Opcode::OP_LOGICALNOT,                               Constraint({dtype_logical},       {})},
        {              Opcode::OP_RANGE,    Constraint({dtype_binary, dtype_binary, dtype_binary},       {})},
        {      Opcode::OP_REGISTER_COPY,              Constraint({dtype_b_i3_1_8, dtype_b_i3_1_8}, {{0, 1}})},
        {             Opcode::OP_EXPAND,                            Constraint({dtype_base_i_u_b},       {})},
        {          Opcode::OP_INDEX_ADD,    Constraint({dtype_b_i3_1, dtype_b_i3_1, dtype_i64_32}, {{0, 1}})},
        {             Opcode::OP_ONEHOT,               Constraint({dtype_i64_32_16, dtype_i64_32},       {})},
        {          Opcode::OP_INDEX_PUT,              Constraint({dtype_base_i_u, dtype_base_i_u}, {{0, 1}})},
        {            Opcode::OP_CUM_SUM,                    Constraint({dtype_base, dtype_i64_32},       {})},
    };
    if (op_2_dtype.find(op) == op_2_dtype.end()) {
        ASSERT(false) << "Operation " << OpcodeManager::Inst().GetOpcodeStr(op) << " has not set dtype checker yet.";
    }
    return op_2_dtype.at(op);
}

inline const std::vector<DataType> &GetInputSupportDtype(const Opcode op, size_t index = 0) {
    const auto &inputs_dtype = std::get<0>(GetSupportDTypeSet(op));
    auto size = inputs_dtype.size();
    ASSERT(size > index) << "Expect [0, " << size << "), but got " << index << ".";
    return inputs_dtype[index];
}

void CheckTensorDType(const LogicalTensorPtr &tensor, const std::vector<DataType> &dtypes) {
    const auto dtype = tensor->GetRawTensor()->GetDataType();
    if (std::find(dtypes.begin(), dtypes.end(), dtype) == dtypes.end()) {
        ASSERT(false) << DataType2String(dtype) << " is not supported.";
    }
}

void CheckInputDTypConstraint(const std::vector<LogicalTensorPtr> &inputs, const std::vector<int> &cons) {
    if (cons.empty()) {
        return;
    }
    const auto dtype = inputs[cons.front()]->GetRawTensor()->GetDataType();
    std::for_each(cons.begin() + 1, cons.end(), [&inputs, dtype](auto &con) {
        const auto idtype = inputs[con]->GetRawTensor()->GetDataType();
        if (dtype != idtype) {
            ASSERT(false) << "The dtype of input[" << con << "] should be " << DataType2String(dtype) << " but got "
                          << DataType2String(idtype);
        }
    });
}

void CheckOperationDType(const std::vector<LogicalTensorPtr> &inputs, const Opcode op) {
    for (size_t index = 0; index < inputs.size(); ++index) {
        CheckTensorDType(inputs[index], GetInputSupportDtype(op, index));
    }
    auto consReq = std::get<1>(GetSupportDTypeSet(op));
    std::for_each(
        consReq.begin(), consReq.end(), [&inputs](const auto &cons) { CheckInputDTypConstraint(inputs, cons); });
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

void CheckAxisRange(const Tensor &tensor, int &axis) {
    int shapeSize = tensor.GetShape().size();
    if (axis < 0) {
        axis += shapeSize;
    }
    ASSERT(axis >= 0 && axis < shapeSize) << "Axis is not in the reasonable range!";
}
} // namespace npu::tile_fwk
