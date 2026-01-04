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
 * \file value.h
 * \brief
 */

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <unordered_set>
#include "type.h"

namespace pto {

class Value {
    std::shared_ptr<Type> GetType();
protected:
    std::shared_ptr<Type> type;
};

using ValuePtr = std::shared_ptr<Value>;

class Scalar : public Value {
public:
    std::shared_ptr<ScalarType> GetType();
};

using ScalarPtr = std::shared_ptr<Scalar>;

static inline std::vector<ValuePtr> CastScalarToValue(const std::vector<ScalarPtr> &scalarList) {
    std::vector<ValuePtr> valueList;
    for (auto scalar : scalarList) {
        valueList.emplace_back(std::static_pointer_cast<ValuePtr>(scalar));
    }
    return valueList;
}

} // namespace pto
