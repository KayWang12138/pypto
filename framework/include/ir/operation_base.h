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
 * \file operation_base.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_set>

#include "value.h"
#include "opcode.h"

namespace pto {

// Base class for operations inside statements.
class Operation : public Object {
public:
    /// Default: invalid / placeholder op
    Operation();
    explicit Operation(Opcode opcode);
    Operation(Opcode opcode, std::string name);

    /// Full construction
    Operation(Opcode opcode,
              std::vector<ValuePtr> ioprands,
              std::vector<ValuePtr> ooprands,
              std::string name = "");

    ~Operation() override = default;

    ObjectType GetObjectType() const override { return ObjectType::Operation; }
    // ---- OpCode ----
    Opcode GetOpcode() const { return opcode_; }

    // ---- IOprands ----
    void AppendInput(const ValuePtr& value) {
        ioprands_.push_back(value);
    }

    size_t GetNumInputOperand() const { return ioprands_.size(); }

    ValuePtr GetInputOperand(size_t idx) const {
        return ioprands_.at(idx);
    }

    void SetInputOperand(size_t idx, const ValuePtr& value) {
        ioprands_.at(idx) = value;
    }

    // ---- OOprands ----
    void AppendOutput(const ValuePtr& value) {
        ooprands_.push_back(value);
    }

    size_t GetNumOutputOperand() const { return ooprands_.size(); }

    ValuePtr GetOutputOperand(size_t idx) const {
        return ooprands_.at(idx);
    }

    void SetOutputOperand(size_t idx, const ValuePtr& value) {
        ooprands_.at(idx) = value;
    }

    // Pretty-print with the given indentation (in spaces).
    void Print(std::ostream& os, int indent = 0) const;
public:
    std::vector<ValuePtr> ioprands_;
    std::vector<ValuePtr> ooprands_;
    Opcode opcode_;
};

using OperationPtr = std::shared_ptr<Operation>;

class ScalarOp : public Operation {
public:
    ScalarOp(Opcode opcode,
             const std::vector<ScalarValuePtr> &inOperandList,
             const std::vector<ScalarValuePtr> &outOperandList)
        : Operation(opcode, CastScalarToValue(inOperandList), CastScalarToValue(outOperandList)) {}

    ScalarValuePtr GetInOperand(size_t index) const;
    ScalarValuePtr GetOutOperand(size_t index) const;
};
using ScalarOpPtr = std::shared_ptr<ScalarOp>;

} // namespace pto
