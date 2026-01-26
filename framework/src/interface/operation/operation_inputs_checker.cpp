/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file operation_inputs_checker.cpp
 * \brief
 */

#include "operation_inputs_checker.h"

namespace npu::tile_fwk {
// if the limit of op is not [1, 4], should add here
const std::unordered_map<const Opcode, const ShapeConstraint> OpInputsChecker::shape_constraints_ = {
    {    Opcode::OP_ADD, {1, 4}},
    {   Opcode::OP_CAST, {1, 4}},
    {Opcode::OP_UNKNOWN, {1, 4}}
};

// {opcode, input dtype constraint(the dtypes of input supported, the index of the inputs whose dtype must be same)}.
const std::unordered_map<const Opcode, const TypeConstraint> OpInputsChecker::dtype_constraints_ = {
    // unary
    {                Opcode::OP_ABS,                                                                 TypeConstraint({dtype_base},{})                                                                                                                                 },
    {                Opcode::OP_EXP,                                                                 TypeConstraint({dtype_base},       {})},
    {                 Opcode::OP_LN,                                                                 TypeConstraint({dtype_base},       {})},
    {               Opcode::OP_SQRT,                                                                 TypeConstraint({dtype_base},       {})},
    {              Opcode::OP_RSQRT,                                                                 TypeConstraint({dtype_base},       {})},
    // one input
    {               Opcode::OP_CAST,                                                             TypeConstraint({dtype_base_i32},       {})},
    {            Opcode::OP_BITSORT,                                                                 TypeConstraint({dtype_fp32},       {})},
    {            Opcode::OP_EXTRACT,                                                                 TypeConstraint({dtype_fp32},       {})},
    {            Opcode::OP_MRGSORT,                                                                 TypeConstraint({dtype_fp32},       {})},
    {                Opcode::OP_NEG,                                                               TypeConstraint({dtype_b_i3_1},       {})},
    {      Opcode::OP_ROWMAX_SINGLE,                                                                 TypeConstraint({dtype_base},       {})},
    {      Opcode::OP_ROWMIN_SINGLE,                                                                 TypeConstraint({dtype_base},       {})},
    {      Opcode::OP_ROWSUM_SINGLE,                                                                 TypeConstraint({dtype_base},       {})},
    {         Opcode::OP_ROWMAXLINE,                                                               TypeConstraint({dtype_binary},       {})},
    {         Opcode::OP_ROWMINLINE,                                                               TypeConstraint({dtype_binary},       {})},
    {         Opcode::OP_ROWSUMLINE,                                                               TypeConstraint({dtype_binary},       {})},
    {               Opcode::OP_TOPK,                                                                 TypeConstraint({dtype_fp32},       {})},
    {   Opcode::OP_TRANSPOSE_MOVEIN,                                                                TypeConstraint({dtype_32_16},       {})},
    {  Opcode::OP_TRANSPOSE_MOVEOUT,                                                                TypeConstraint({dtype_32_16},       {})},
    {Opcode::OP_TRANSPOSE_VNCHWCONV,                                                                TypeConstraint({dtype_32_16},       {})},
    // binary
    {                Opcode::OP_ADD,                                                 TypeConstraint({dtype_binary, dtype_binary}, {{0, 1}})},
    {                Opcode::OP_SUB,                                                 TypeConstraint({dtype_binary, dtype_binary}, {{0, 1}})},
    {                Opcode::OP_MUL,                                                 TypeConstraint({dtype_binary, dtype_binary}, {{0, 1}})},
    {                Opcode::OP_DIV,                                                     TypeConstraint({dtype_base, dtype_base}, {{0, 1}})},
    {               Opcode::OP_ADDS,                                                               TypeConstraint({dtype_binary},       {})},
    {               Opcode::OP_SUBS,                                                               TypeConstraint({dtype_binary},       {})},
    {               Opcode::OP_MULS,                                                               TypeConstraint({dtype_binary},       {})},
    {               Opcode::OP_DIVS,                                                               TypeConstraint({dtype_binary},       {})},
    {               Opcode::OP_MAXS,                                                               TypeConstraint({dtype_binary},       {})},
    {               Opcode::OP_MINS,                                                               TypeConstraint({dtype_binary},       {})},
    {            Opcode::OP_MAXIMUM,                                                 TypeConstraint({dtype_binary, dtype_binary}, {{0, 1}})},
    {            Opcode::OP_MINIMUM,                                                 TypeConstraint({dtype_binary, dtype_binary}, {{0, 1}})},
    // two input
    {                Opcode::OP_CMP,                                                     TypeConstraint({dtype_base, dtype_base}, {{0, 1}})},
    {               Opcode::OP_CMPS,                                                                 TypeConstraint({dtype_base},       {})},
    {             Opcode::OP_GATHER,                                            TypeConstraint({dtype_b_i3_1_8, dtype_i64_32_16},       {})},
    {     Opcode::OP_GATHER_ELEMENT,                                                 TypeConstraint({dtype_b_i3_1, dtype_i64_32},       {})},
    {            Opcode::OP_VEC_DUP,                                                               TypeConstraint({dtype_binary},       {})},
    {            Opcode::OP_SCATTER,                                       TypeConstraint({dtype_base, dtype_i64_32, dtype_base}, {{0, 2}})},
    {    Opcode::OP_SCATTER_ELEMENT,                                                   TypeConstraint({dtype_base, dtype_i64_32},       {})},
    {      Opcode::OP_INDEX_OUTCAST,                                TypeConstraint({dtype_b_i3_1, dtype_i64_32_16, dtype_b_i3_1}, {{0, 2}})},
    {           Opcode::OP_WHERE_TT,                                         TypeConstraint({dtype_u8_b, dtype_base, dtype_base}, {{1, 2}})},
    {           Opcode::OP_WHERE_TS,                                                     TypeConstraint({dtype_u8_b, dtype_base},       {})},
    {           Opcode::OP_WHERE_ST,                                                     TypeConstraint({dtype_u8_b, dtype_base},       {})},
    {           Opcode::OP_WHERE_SS,                                                                 TypeConstraint({dtype_u8_b},       {})},
    {         Opcode::OP_LOGICALAND,                                               TypeConstraint({dtype_logical, dtype_logical},       {})},
    {         Opcode::OP_LOGICALNOT,                                                              TypeConstraint({dtype_logical},       {})},
    {              Opcode::OP_RANGE,                                   TypeConstraint({dtype_binary, dtype_binary, dtype_binary},       {})},
    {      Opcode::OP_REGISTER_COPY,                                             TypeConstraint({dtype_b_i3_1_8, dtype_b_i3_1_8}, {{0, 1}})},
    {             Opcode::OP_EXPAND,                                                           TypeConstraint({dtype_base_i_u_b},       {})},
    {          Opcode::OP_INDEX_ADD,                                   TypeConstraint({dtype_b_i3_1, dtype_b_i3_1, dtype_i64_32}, {{0, 1}})},
    {             Opcode::OP_ONEHOT,                                                            TypeConstraint({dtype_i64_32_16},       {})},
    {          Opcode::OP_INDEX_PUT, TypeConstraint({dtype_base_i_u, dtype_base_i_u, dtype_i_u, dtype_i_u, dtype_i_u, dtype_i_u},
     {{0, 1}, {2, 3, 4, 5}})                                                                                                               },
    {            Opcode::OP_CUM_SUM,                                                               TypeConstraint({dtype_binary},       {})},
};

void OpInputsChecker::AddInput(const LogicalTensorPtr &tensor) {
    inputs_shape_.push_back(tensor->shape);
    inputs_type_.push_back(tensor->GetRawTensor()->GetDataType());
}

void OpInputsChecker::AddInput(const Element &scalar) {
    inputs_type_.push_back(scalar->GetDataType());
}

void OpInputsChecker::Check() const {
    CheckShape();
    CheckType();
}

void OpInputsChecker::Check(const std::vector<LogicalTensorPtr> &inputs) const {
    inputs_shape_.clear();
    inputs_type_.clear();
    std::for_each(inputs.begin(), inputs.end(), [](const auto &input) { AddInput(input); });
    Check();
}

const ShapeConstraint &OpInputsChecker::GetShapeTypeConstraint() const {
    if (shape_constraints_.find(opcode_) == shape_constraints_.end()) {
        return shape_constraints_.at(Opcode::OP_UNKNOWN);
    }
    return shape_constraints_.at(opcode_);
}

void OpInputsChecker::CheckShape() const {
    // valid input dims must in [1, 4]
    auto shape_con = GetShapeTypeConstraint();
    std::for_each(inputs_shape_.begin(), inputs_shape_.end(), [&shape_con](const auto &shape) {
        if (shape.size() < shape_con[0] || shape.size() > shape_con[1]) {
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
    });
}

const TypeConstraint &OpInputsChecker::GetTypeTypeConstraint() const {
    if (dtype_constraints_.find(opcode_) == dtype_constraints_.end()) {
        ASSERT(false) << "Operation " << OpcodeManager::Inst().GetOpcodeStr(opcode_)
                      << " has not set dtype checker yet.";
    }
    return dtype_constraints_.at(opcode_);
}

void OpInputsChecker::CheckInputTypeConsistency(const TypeSet &types, const std::vector<int> &indexs) const {
    if (indexs.empty() || types.empty()) {
        return;
    }
    const auto type = types[indexs.front()];
    std::for_each(indexs.begin() + 1, indexs.end(), [&types, type](auto &index) {
        if (type != types[index]) {
            ASSERT(false) << "The dtype of input[" << con << "] should be " << DataType2String(type) << " but got "
                          << DataType2String(types[index]);
        }
    });
}

void OpInputsChecker::CheckType() const {
    const auto &dtype_con = std::get<0>(GetTypeTypeConstraint());
    std::for_each(inputs_type_.begin(), inputs_type_.end(), [&dtype_con](const auto &type) {
        if (std::find(dtype_con.begin(), dtype_con.end(), type) == dtypes.end()) {
            ASSERT(false) << DataType2String(dtype) << " is not supported.";
        }
    });
    const auto index_cons = std::get<1>(GetTypeTypeConstraint());
    std::for_each(index_cons.begin(), index_cons.end(),
        [&inputs_type_](const auto &indexs) { CheckInputTypeConsistency(inputs_type_, indexs); });
}
} // namespace npu::tile_fwk
