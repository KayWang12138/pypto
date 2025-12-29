// PTO-IR prototype: type system structures implementation.

#include "ir/type.h"
#include "ir/utils.h"

#include <algorithm>
#include <ostream>
#include <variant>

namespace pto {


const std::string& Scalar::GetSymbolicExpr() const {
    if (symbolicExpr_.empty()) {
        symbolicExpr_ = DataTypeToString(GetDataType());
    }
    return symbolicExpr_;
}

int64_t Scalar::GetInt64Value() const {
    if (!HasConstantValue()) {
        throw std::runtime_error("Scalar does not hold a constant value");
    }
    return std::visit([](const auto& val) -> int64_t {
        return static_cast<int64_t>(val);
    }, constantValue_);
}

void Scalar::Print(std::ostream& os, int indent) const {
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
    }
}

void Tensor::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "tensor<";

    // ====== shape ======
    os << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        shape_[i].Print(os);
        if (i + 1 < shape_.size()) {
            os << ", ";
        }
    }
    os << "]";

    // ====== type ======
    os << ", ";
    os << DataTypeToString(GetDataType());

    os << ">";
}

bool Tile::isDense() const {
    if (strides_.empty()) {
        return true;
    }
    std::vector<std::pair<size_t, size_t>> ss;
    for (size_t i = 0; i < strides_.size(); i++) {
        // shape == 1 have no effect
        if (shape_[i] != 1) {
            ss.push_back({strides_[i], shape_[i]});
        }
    }
    std::sort(ss.begin(), ss.end());
    size_t expected = 1;
    for (auto [st, sp] : ss) {
        if (st != expected) {
            return false;
        }
        expected *= sp;
    }
    return true;
}

bool Tile::isContiguous() const {
    if (strides_.empty()) {
        return true;
    }
    size_t expected = 1;
    for (size_t i = strides_.size() - 1; i >= 0; i--) {
        if (strides_[i] != expected) {
            return false;
        }
        expected *= shape_[i];
    }
    return true;
}

void Tile::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);
    os << "tile<[";

    // ====== valid shape ======
    for (size_t i = 0; i < validShapes_.size(); ++i) {
        validShapes_[i].Print(os, 0);
        if (i + 1 < shape_.size()) {
            os << ", ";
        }
    }
    os << "], [";

    // ====== tile shapes ======
    for (size_t i = 0; i < shape_.size(); ++i) {
        os << shape_[i];
        if (i + 1 < shape_.size()) {
            os << ", ";
        }
    }
    os << "], ";

    // // ====== strides ======
    // for (size_t i = 0; i < strides_.size(); ++i) {
    //     os << strides_[i];
    //     if (i + 1 < strides_.size()) {
    //         os << ", ";
    //     }
    // }
    // os << "], ";

    // // ====== offset ======
    // if (startOffset_.has_value()) {
    //     (*startOffset_).Print(os, 0);
    // }
    // os << ", ";

    // ====== type ======
    os << DataTypeToString(GetDataType());

    os << ">";
}

} // namespace pto

