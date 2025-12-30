// PTO-IR prototype: function-level IR structures.
// All comments must remain in English for consistency.

#pragma once

#include "ir/statement.h"
#include "ir/scope.h"
#include "ir/type.h"
#include "ir/utils.h"

#include <ostream>
#include <string>
#include <vector>

namespace pto {

// High-level classification of PTO functions.
enum class FunctionKind {
    ControlFlow, // control-flow functions using statement dialect
    DataFlow,    // pure data-flow graphs at tensor/tile level
    Kernel       // low-level kernels near instruction/memory level
};

// Signature of a function: arguments and results.
// Arguments are Data objects where the name field stores the argument name (e.g. "%A").
struct FunctionSignature {
    std::vector<ValuePtr> arguments; // argument types with names stored in Value::name
    std::vector<ValuePtr> results;  // return types
};

// Minimal container for a PTO function.
// This prototype focuses on structural information and simple printing,
// not on detailed statement/tensor/tile bodies.
class Function : public Object, public AttributeHolder {
public:
    Function(std::string name, FunctionKind kind, FunctionSignature signature);

    ObjectType GetObjectType() const override { return ObjectType::Function; }

    FunctionKind GetKind() const { return kind_; }
    const FunctionSignature& GetSignature() const { return signature_; }

    // Top-level statement sequence forming the function body.
    std::vector<StatementPtr>& Body() { return scope_.GetStatements(); }
    const std::vector<StatementPtr> Body() const { return scope_.GetStatements(); }

    // Scope for Data objects and statements created in this function.
    Scope& GetScope() { return scope_; }
    const Scope& GetScope() const { return scope_; }

    // Scope containing function arguments. This scope is the parent of the function body scope.
    Scope& GetInputScope() { return input_scope_; }
    const Scope& GetInputScope() const { return input_scope_; }

    // Convenience to append a top-level statement.
    void AddStatement(StatementPtr stmt);

    // Pretty-print a standalone function in PTO-IR-like syntax.
    void Print(std::ostream& os, int indent = 0) const;

private:
    FunctionKind kind_;
    FunctionSignature signature_;
    Scope input_scope_; // Scope holding function arguments (inputs)
    Scope scope_;  // Scope for Data objects and statements created in this function
};

// Helper for convenient streaming: std::cout << func;
std::ostream& operator<<(std::ostream& os, const Function& func);

} // namespace pto


