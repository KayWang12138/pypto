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
 * \file operation.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_set>
#include "value.h"

namespace pto {

class Operation {
public:
    size_t GetInOperandSize() const;
    std::shared_ptr<Value> GetInOperand(size_t index) const;

    size_t GetOutOperandSize() const;
    std::shared_ptr<Value> GetOutOperand(size_t index) const;
private:
    std::vector<std::shared_ptr<Value>> inOperandList_;
    std::vector<std::shared_ptr<Value>> outOperandList_;
};

class ScalarOp : public Operation {
public:
    std::shared_ptr<ScalarValue> GetInOperand(size_t index) const;
    std::shared_ptr<ScalarValue> GetOutOperand(size_t index) const;
};

} // namespace pto
