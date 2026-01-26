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

static const InputTypeSet dtype_fp32 = {{DT_FP32}};
static const InputTypeSet dtype_unary = {
    {DT_FP32, DT_FP16, DT_BF16}
};
static const InputTypeSet dtype_cast = {DT_FP32, DT_FP16, DT_BF16, DT_INT32};
static const InputTypeSet dtype_bin_s = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16}
};
static const InputTypeSet dtype_bin_base = {
    {DT_FP32, DT_FP16, DT_BF16},
    {DT_FP32, DT_FP16, DT_BF16}
};
static const InputTypeSet dtype_binary = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16},
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16}
};
static const InputTypeSet dtype_transpose = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_UINT32, DT_UINT16};
static const InputTypeSet dtype_index_add = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16},
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16},
    {DT_INT64, DT_INT32}
};
static const InputTypeSet dtype_index_outcase = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16},
    {DT_INT64, DT_INT32, DT_INT16},
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16}
};
static const InputTypeSet dtype_index_put = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, DT_UINT32, DT_UINT16, DT_UINT8},
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, DT_UINT32, DT_UINT16, DT_UINT8},
    {DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT64, DT_UINT32, DT_UINT16, DT_UINT8},
    {DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT64, DT_UINT32, DT_UINT16, DT_UINT8},
    {DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT64, DT_UINT32, DT_UINT16, DT_UINT8},
    {DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT64, DT_UINT32, DT_UINT16, DT_UINT8}
};
static const InputTypeSet dtype_gather = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8},
    {DT_INT64, DT_INT32, DT_INT16}
};
static const InputTypeSet dtype_gather_element = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16},
    {DT_INT64, DT_INT32}
};
static const InputTypeSet dtype_scatter = {
    {DT_FP32, DT_FP16, DT_BF16},
    {DT_INT64, DT_INT32},
    {DT_FP32, DT_FP16, DT_BF16}
};
static const InputTypeSet dtype_scatter_element = {
    {DT_FP32, DT_FP16, DT_BF16},
    {DT_INT64, DT_INT32}
};
static const InputTypeSet dtype_where_ss = {
    {DT_UINT8, DT_BOOL}
};
static const InputTypeSet dtype_where_st = {
    {DT_UINT8, DT_BOOL},
    {DT_FP32, DT_FP16, DT_BF16},
    {DT_FP32, DT_FP16, DT_BF16}
};
static const InputTypeSet dtype_where_tt = {
    {DT_UINT8, DT_BOOL},
    {DT_FP32, DT_FP16, DT_BF16},
    {DT_FP32, DT_FP16, DT_BF16}
};
static const InputTypeSet dtype_logical_not = {
    {DT_FP32, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_BOOL}
};
static const InputTypeSet dtype_logical_and = {
    {DT_FP32, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_BOOL},
    {DT_FP32, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_BOOL}
};
static const InputTypeSet dtype_concat = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8},
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8}
};
static const InputTypeSet dtype_expand = {
    {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, DT_UINT32, DT_UINT16, DT_UINT8, DT_BOOL}
};
static const InputTypeSet dtype_onehot = {dtype_i64_32_16};

static const InputIndices indices_empty = {};
static const InputIndices indices_binary = {
    {0, 1}
};
static const InputIndices indices_skip = {
    {0, 2}
};
const InputIndices indices_where = {
    {1, 2}
};
static const InputIndices indices_concat = {
    {0, 1},
    {2, 3, 4, 5}
};

// {opcode, input dtype constraint(the dtypes of input supported, the index of the inputs whose dtype must be same)}.
const std::unordered_map<const Opcode, const TypeConstraint> OpInputsChecker::dtype_constraints_ = {
    // unary
    {                Opcode::OP_ABS,           TypeConstraint(dtype_unary,  indices_empty)},
    {                Opcode::OP_EXP,           TypeConstraint(dtype_unary,  indices_empty)},
    {                 Opcode::OP_LN,           TypeConstraint(dtype_unary,  indices_empty)},
    {               Opcode::OP_SQRT,           TypeConstraint(dtype_unary,  indices_empty)},
    {              Opcode::OP_RSQRT,           TypeConstraint(dtype_unary,  indices_empty)},
    // one input
    {               Opcode::OP_CAST,            TypeConstraint(dtype_cast,  indices_empty)},
    {            Opcode::OP_BITSORT,            TypeConstraint(dtype_fp32,  indices_empty)},
    {            Opcode::OP_EXTRACT,            TypeConstraint(dtype_fp32,  indices_empty)},
    {            Opcode::OP_MRGSORT,            TypeConstraint(dtype_fp32,  indices_empty)},
    {                Opcode::OP_NEG,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {      Opcode::OP_ROWMAX_SINGLE,           TypeConstraint(dtype_unary,  indices_empty)},
    {      Opcode::OP_ROWMIN_SINGLE,           TypeConstraint(dtype_unary,  indices_empty)},
    {      Opcode::OP_ROWSUM_SINGLE,           TypeConstraint(dtype_unary,  indices_empty)},
    {         Opcode::OP_ROWMAXLINE,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {         Opcode::OP_ROWMINLINE,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {         Opcode::OP_ROWSUMLINE,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {               Opcode::OP_TOPK,            TypeConstraint(dtype_fp32,  indices_empty)},
    {   Opcode::OP_TRANSPOSE_MOVEIN,       TypeConstraint(dtype_transpose,  indices_empty)},
    {  Opcode::OP_TRANSPOSE_MOVEOUT,       TypeConstraint(dtype_transpose,  indices_empty)},
    {Opcode::OP_TRANSPOSE_VNCHWCONV,       TypeConstraint(dtype_transpose,  indices_empty)},
    // binary
    {                Opcode::OP_ADD,          TypeConstraint(dtype_binary, indices_binary)},
    {                Opcode::OP_SUB,          TypeConstraint(dtype_binary, indices_binary)},
    {                Opcode::OP_MUL,          TypeConstraint(dtype_binary, indices_binary)},
    {                Opcode::OP_DIV,        TypeConstraint(dtype_bin_base, indices_binary)},
    {               Opcode::OP_ADDS,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {               Opcode::OP_SUBS,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {               Opcode::OP_MULS,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {               Opcode::OP_DIVS,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {               Opcode::OP_MAXS,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {               Opcode::OP_MINS,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {            Opcode::OP_MAXIMUM,          TypeConstraint(dtype_binary, indices_binary)},
    {            Opcode::OP_MINIMUM,          TypeConstraint(dtype_binary, indices_binary)},
    // two input
    {                Opcode::OP_CMP,        TypeConstraint(dtype_bin_base, indices_binary)},
    {               Opcode::OP_CMPS,           TypeConstraint(dtype_unary,  indices_empty)},
    {             Opcode::OP_GATHER,          TypeConstraint(dtype_gather,  indices_empty)},
    {     Opcode::OP_GATHER_ELEMENT,  TypeConstraint(dtype_gather_element,  indices_empty)},
    {            Opcode::OP_VEC_DUP,           TypeConstraint(dtype_bin_s,  indices_empty)},
    {            Opcode::OP_SCATTER,         TypeConstraint(dtype_scatter,   indices_skip)},
    {    Opcode::OP_SCATTER_ELEMENT, TypeConstraint(dtype_scatter_element,  indices_empty)},
    {      Opcode::OP_INDEX_OUTCAST,   TypeConstraint(dtype_index_outcase,   indices_skip)},
    {           Opcode::OP_WHERE_TT,        TypeConstraint(dtype_where_tt,  indices_where)},
    {           Opcode::OP_WHERE_TS,        TypeConstraint(dtype_where_st,  indices_empty)},
    {           Opcode::OP_WHERE_ST,        TypeConstraint(dtype_where_st,  indices_empty)},
    {           Opcode::OP_WHERE_SS,        TypeConstraint(dtype_where_ss,  indices_empty)},
    {         Opcode::OP_LOGICALAND,     TypeConstraint(dtype_logical_and,  indices_empty)},
    {         Opcode::OP_LOGICALNOT,     TypeConstraint(dtype_logical_not,  indices_empty)},
    {      Opcode::OP_REGISTER_COPY,          TypeConstraint(dtype_concat, indices_binary)},
    {             Opcode::OP_EXPAND,          TypeConstraint(dtype_expand,  indices_empty)},
    {          Opcode::OP_INDEX_ADD,       TypeConstraint(dtype_index_add, indices_binary)},
    {             Opcode::OP_ONEHOT,          TypeConstraint(dtype_onehot,  indices_empty)},
    {          Opcode::OP_INDEX_PUT,       TypeConstraint(dtype_index_put, indices_concat)},
    {            Opcode::OP_CUM_SUM,           TypeConstraint(dtype_bin_s,  indices_empty)},
};

void OpInputsChecker::AddInput(const LogicalTensorPtr &tensor) {
    inputs_shape_.push_back(tensor->shape);
    inputs_type_.push_back(tensor->GetRawTensor()->GetDataType());
}

void OpInputsChecker::AddInput(const Element &scalar) {
    inputs_type_.push_back(scalar.GetDataType());
}

void OpInputsChecker::Check() const {
    CheckShape();
    CheckType();
}

void OpInputsChecker::Check(const std::vector<LogicalTensorPtr> &inputs) {
    inputs_shape_.clear();
    inputs_type_.clear();
    std::for_each(inputs.begin(), inputs.end(), [this](const auto &input) { AddInput(input); });
    Check();
}

const ShapeConstraint &OpInputsChecker::GetShapeConstraint() const {
    if (shape_constraints_.find(opcode_) == shape_constraints_.end()) {
        return shape_constraints_.at(Opcode::OP_UNKNOWN);
    }
    return shape_constraints_.at(opcode_);
}

void OpInputsChecker::CheckShape() const {
    // valid input dims must in [1, 4]
    auto shape_con = GetShapeConstraint();
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

const TypeConstraint &OpInputsChecker::GetTypeConstraint() const {
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
            ASSERT(false) << "The dtype of input[" << index << "] should be " << DataType2String(type) << " but got "
                          << DataType2String(types[index]);
        }
    });
}

void OpInputsChecker::CheckType() const {
    const auto &dtype_con = GetTypeConstraint().types_;
    size_t index = 0;
    std::for_each(inputs_type_.begin(), inputs_type_.end(), [&dtype_con, &index](const auto &type) {
        if (std::find(dtype_con[index].begin(), dtype_con[index].end(), type) == dtype_con[index].end()) {
            ASSERT(false) << DataType2String(type) << " is not supported.";
        }
    });
    const auto index_cons = GetTypeConstraint().indices_;
    std::for_each(index_cons.begin(), index_cons.end(),
        [this](const auto &indexs) { CheckInputTypeConsistency(inputs_type_, indexs); });
}
} // namespace npu::tile_fwk
