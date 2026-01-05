// ir/builder/ir_builder_op.cpp
// PTO-IR prototype: IRBuilder op emission + schema-driven Create().

#include "ir/builder/ir_builder.h"
#include <utility>

namespace pto {

OperationPtr IRBuilder::Emit(OperationPtr op) {
    auto opStmt = GetOrCreateActiveOpStmt();
    opStmt->Operations().push_back(std::move(op));
    return opStmt->Operations().back();
}

} // namespace pto
