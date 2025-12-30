// ir/builder/ir_builder_value.cpp
// PTO-IR prototype: IRBuilder value helpers (explicit value creation only).

#include "ir/builder/ir_builder.h"

#include <stdexcept>
#include <utility>

namespace pto {

ValuePtr IRBuilder::AddToScope(ValuePtr v) {
    if (!scope_) throw std::runtime_error("IRBuilder::AddToScope: scope is null");
    if (!v) throw std::runtime_error("IRBuilder::AddToScope: value is null");
    // Use SSA name as the key in environment table
    std::string key = v->GetName();
    scope_->SetEnvVar(key, v);
    return v;
}

std::shared_ptr<Tensor> IRBuilder::CreateTensor(
    const std::vector<Scalar>& shape, DataType dt, std::string name) {
    // Create a Tensor value object directly without creating a TensorCreate operation.
    // TensorCreate operations should only be created explicitly by the user code or parser,
    // not implicitly by helper methods.
    auto t = std::make_shared<Tensor>(shape, dt, std::move(name));
    AddToScope(t);
    return t;
}

std::shared_ptr<Tile> IRBuilder::CreateTile(
    const std::vector<size_t>& shape, DataType dt, std::string name) {
    auto t = std::make_shared<Tile>(shape, dt, std::move(name));
    AddToScope(t);
    return t;
}

std::shared_ptr<Scalar> IRBuilder::CreateScalar(DataType dt, std::string name) {
    auto s = std::make_shared<Scalar>(dt, std::move(name), ScalarValueKind::Symbolic);
    AddToScope(s);
    return s;
}

std::shared_ptr<Scalar> IRBuilder::CreateConst(int64_t v, std::string name) {
    auto s = std::make_shared<Scalar>(v, std::move(name));
    AddToScope(s);
    return s;
}

std::shared_ptr<Scalar> IRBuilder::CreateConst(double v, std::string name) {
    auto s = std::make_shared<Scalar>(v, std::move(name));
    AddToScope(s);
    return s;
}

} // namespace pto
