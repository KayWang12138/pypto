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
using InputIndices = std::vector<std::vector<int>>;

struct TypeConstraint {
    InputTypeSet types_;
    InputIndices indices_;

    TypeConstraint(const InputTypeSet &types, const InputIndices &indices) : types_(types), indices_(indices) {}
};

class OpInputsChecker {
public:
    ~OpInputsChecker() = default;

    static OpInputsChecker GetInstance(const Opcode opcode) {
        OpInputsChecker instance(opcode);
        return instance;
    }

    void AddInput(const LogicalTensorPtr &tensor);

    void AddInput(const Element &scalar);

    void Check() const;

    void Check(const std::vector<LogicalTensorPtr> &inputs);

private:
    explicit OpInputsChecker(const Opcode opcode) : opcode_(opcode) {}
    OpInputsChecker() = delete;

    const ShapeConstraint &GetShapeConstraint() const;
    void CheckShape() const;
    const TypeConstraint &GetTypeConstraint() const;
    void CheckInputTypeConsistency(const TypeSet &types, const std::vector<int> &indexs) const;
    void CheckType() const;

    const Opcode opcode_;
    std::vector<Shape> inputs_shape_;
    TypeSet inputs_type_;
    static const std::unordered_map<const Opcode, const ShapeConstraint> shape_constraints_;
    static const std::unordered_map<const Opcode, const TypeConstraint> dtype_constraints_;
};
} // namespace npu::tile_fwk
#endif // INTERFACE_MAIN_OPERATION_INPUTS_CHECKER_H
