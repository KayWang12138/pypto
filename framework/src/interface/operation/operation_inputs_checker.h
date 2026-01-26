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
 * \file operation_inputs_checker.h
 * \brief
 */

#ifndef INTERFACE_MAIN_OPERATION_INPUTS_CHECKER_H
#define INTERFACE_MAIN_OPERATION_INPUTS_CHECKER_H

#include "interface/operation/opcode.h"
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {
using ShapeConstraint = std::vector<size_t>;
using TypeSet = std::vector<DataType>;
using InputTypeSet = std::vector<TypeSet>;
using TypeConstraint = std::tuple<const InputTypeSet, const std::vector<std::vector<int>>>;

class OpInputsChecker {
public:
    static OpInputsChecker GetInstance(const Opcode opcode) {
        OpInputsChecker instance(opcode);
        return instance;
    }

    void AddInput(const LogicalTensorPtr &tensor);

    void AddInput(const Element &scalar);

    void Check() const;

    void Check(const std::vector<LogicalTensorPtr> &inputs) const;

private:
    explicit OpInputsChecker(const Opcode opcode) : opcode_(opcode) {}
    OpInputsChecker() = delete;

    ~OpInputsChecker() = default;

    const ShapeConstraint &GetShapeConstraint() const;
    void CheckShape() const;
    const TypeConstraint &GetTypeConstraint() const;
    void CheckInputTypeConsistency(const TypeSet &types, const std::vector<int> &indexs) const;
    void CheckType() const;

    const Opcode opcode_;
    std::vector<Shape> inputs_shape_;
    TypeSet inputs_type_;

    static const auto dtype_fp32 = {DT_FP32};
    static const auto dtype_u8_b = {DT_UINT8, DT_BOOL};
    static const auto dtype_i64_32 = {DT_INT64, DT_INT32};
    static const auto dtype_base = {DT_FP32, DT_FP16, DT_BF16};
    static const auto dtype_binary = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const auto dtype_32_16 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_UINT32, DT_UINT16};
    static const auto dtype_i64_32_16 = {DT_INT64, DT_INT32, DT_INT16};
    static const auto dtype_i = {DT_INT64, DT_INT32, DT_INT16, DT_INT8};
    static const auto dtype_i_u = {DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT64, DT_UINT32, DT_UINT16, DT_UINT8};
    static const auto dtype_logical = {DT_FP32, DT_FP16, DT_BF16, DT_UINT8, DT_INT8, DT_BOOL};
    static const auto dtype_base_i32 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32};
    static const auto dtype_b_i3_1 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16};
    static const auto dtype_b_i3_1_8 = {DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8};
    static const auto dtype_base_i_u = {
        DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, DT_UINT32, DT_UINT16, DT_UINT8};
    static const auto dtype_base_i_u_b = {
        DT_FP32, DT_FP16, DT_BF16, DT_INT32, DT_INT16, DT_INT8, DT_UINT32, DT_UINT16, DT_UINT8, DT_BOOL};

    static const std::unordered_map<const Opcode, const ShapeConstraint> shape_constraints_;
    static const std::unordered_map<const Opcode, const TypeConstraint> dtype_constraints_;
};
} // namespace npu::tile_fwk
#endif // INTERFACE_MAIN_OPERATION_INPUTS_CHECKER_H
