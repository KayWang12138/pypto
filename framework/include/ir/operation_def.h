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
 * \file operation_def.h
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

class Operation {
public:
    Operation(Opcode opcode,
              const std::vector<ValuePtr> &inOperandList,
              const std::vector<ValuePtr> &outOperandList)
        : opcode_(opcode),
          inOperandList_(inOperandList),
          outOperandList_(outOperandList) {}

    Opcode GetOpcode() const { return opcode_; }
    size_t GetInOperandSize() const { return inOperandList_.size(); }
    ValuePtr GetInOperand(size_t index) const { return inOperandList_[index]; }

    size_t GetOutOperandSize() const { return outOperandList_.size(); }
    ValuePtr GetOutOperand(size_t index) const { return outOperandList_[index]; }
private:
    Opcode opcode_;
    std::vector<ValuePtr> inOperandList_;
    std::vector<ValuePtr> outOperandList_;
};
class ScalarOp : public Operation {
public:
    ScalarOp(Opcode opcode,
             const std::vector<ScalarPtr> &inOperandList,
             const std::vector<ScalarPtr> &outOperandList)
        : Operation(opcode, CastScalarToValue(inOperandList), CastScalarToValue(outOperandList)) {}

    ScalarPtr GetInOperand(size_t index) const;
    ScalarPtr GetOutOperand(size_t index) const;
};

} // namespace pto
