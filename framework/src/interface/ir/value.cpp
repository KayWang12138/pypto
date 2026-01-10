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
 * \file value.cpp
 * \brief
 */
#include "ir/utils.h"
#include "ir/value.h"

#include <ostream>
#include <variant>

namespace pto {
// ========== Value System Implementation ==========

int64_t ScalarValue::GetInt64Value() const {
    if (!HasImmediateValue()) {
        throw std::runtime_error("ScalarValue does not hold a constant value");
    }
    return std::visit([](const auto& val) -> int64_t {
        return static_cast<int64_t>(val);
    }, immediateValue_);
}

void ScalarValue::PrintSSAName(std::ostream& os) const {
    switch (valueKind_) {
    case ScalarValueKind::Immediate:
        // For immediate values, print the actual constant value
        std::visit([&os](const auto& val) {
            os << val;
        }, immediateValue_);
        break;
    case ScalarValueKind::Symbolic:
        os << GetSSAName();
        break;
    default:
        os << "Unknown ScalarValue";
    }
}

void ScalarValue::PrintValue(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    PrintSSAName(os);
    os << ": ";
    PrintType(os);
}

void ScalarValue::PrintType(std::ostream& os) const {
    if (type_) {
        type_->Print(os);
    } else {
        os << "unknown";
    }
}

void TensorValue::PrintSSAName(std::ostream& os) const {
    os << GetSSAName();
}

void TensorValue::PrintValue(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    PrintSSAName(os);
    os << ": ";
    PrintType(os);
}

void TensorValue::PrintType(std::ostream& os) const {
    os << "tensor<";
    // ====== shape ======
    os << "[";
    auto shape = GetShape();
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i]->PrintSSAName(os);
        if (i + 1 < shape.size()) {
            os << ", ";
        }
    }
    os << "]";
    // ====== type ======
    os << ", ";
    os << DataTypeToString(GetDataType());
    os << ">";
}

void TileValue::PrintSSAName(std::ostream& os) const {
    os << GetSSAName();
}

void TileValue::PrintValue(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    PrintSSAName(os);
    os << ": ";
    PrintType(os);
}

void TileValue::PrintType(std::ostream& os) const {
    os << "tile<[";
    // ====== valid shape ======
    const auto& shape = GetShape();
    for (size_t i = 0; i < validShapes_.size(); ++i) {
        validShapes_[i]->PrintSSAName(os);
        if (i + 1 < shape.size()) {
            os << ", ";
        }
    }
    os << "], [";
    // ====== tile shapes ======
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i + 1 < shape.size()) {
            os << ", ";
        }
    }
    os << "], ";
    // ====== type ======
    os << DataTypeToString(GetDataType());
    os << ">";
}

} // namespace pto
