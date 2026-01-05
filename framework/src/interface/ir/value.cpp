// PTO-IR prototype: type system structures implementation.

#include "ir/utils.h"
#include "ir/value.h"

#include <ostream>
#include <variant>

namespace pto {
// ========== Value System Implementation ==========

const std::string& ScalarValue::GetSymbolicExpr() const {
    if (symbolicExpr_.empty()) {
        symbolicExpr_ = DataTypeToString(GetDataType());
    }
    return symbolicExpr_;
}

int64_t ScalarValue::GetInt64Value() const {
    if (!HasConstantValue()) {
        throw std::runtime_error("ScalarValue does not hold a constant value");
    }
    return std::visit([](const auto& val) -> int64_t {
        return static_cast<int64_t>(val);
    }, constantValue_);
}

void ScalarValue::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);

    switch (valueKind_) {
    case ScalarValueKind::Constant:
        // Print the actual constant value
        std::visit([&os](const auto& val) {
            os << val;
        }, constantValue_);
        break;
    case ScalarValueKind::Symbolic:
        os << GetSSAName();
        break;
    default:
        os << "Unknown ScalarValue";
    }
}

void TensorValue::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "tensor<";

    // ====== shape ======
    os << "[";
    auto shape = GetShape();
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i]->Print(os);
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

void TileValue::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "tile<[";

    // ====== valid shape ======
    const auto& shape = GetShape();
    for (size_t i = 0; i < validShapes_.size(); ++i) {
        validShapes_[i]->Print(os, 0);
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
