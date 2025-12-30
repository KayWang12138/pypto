
#pragma once

#include "ir/scope.h"
#include "ir/type.h"
#include "ir/utils.h"
#include "ir/operation.h"

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pto {

// Kinds of statement nodes supported in the prototype.
enum class StatementKind {
    Block,
    For,
    If,
    Yield,
    Call,
    Return
    // Memory operations (load/store/alloc/...) can be added later.
};

// Base class for all statement nodes.
class Statement : public Object, public AttributeHolder {
public:
    Statement() : Object(ObjectType::Statement) {}
    virtual ~Statement() = default;

    ObjectType GetObjectType() const override { return ObjectType::Statement; }

    virtual StatementKind GetKind() const = 0;

    // Pretty-print with the given indentation (in spaces).
    virtual void Print(std::ostream& os, int indent) const = 0;
};

using StatementPtr = std::shared_ptr<Statement>;

// A linear basic block of operations with nested statements as children.
class BlockStatement : public Statement {
public:
    StatementKind GetKind() const override { return StatementKind::Block; }

    // Free-form textual operations representing concrete IR ops.
    std::vector<OperationPtr>& Operations() { return operations_; }
    const std::vector<OperationPtr>& Operations() const { return operations_; }

    void Print(std::ostream& os, int indent) const override;

private:
    std::vector<OperationPtr> operations_;
};

using BlockStatementPtr = std::shared_ptr<BlockStatement>;

}