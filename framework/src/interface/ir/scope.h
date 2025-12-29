// PTO-IR prototype: scope management for Data objects.
// All comments must remain in English for consistency.

#pragma once

#include "ir/type.h"
#include <vector>
#include <unordered_map>

namespace pto {

// Forward declaration for statements so Scope can store StatementPtr.
class Statement;
using StatementPtr = std::shared_ptr<Statement>;

// Scope class for managing Value objects and statements in nested scopes.
// Each scope has a pointer to its parent scope, forming a scope chain.
class Scope {
public:
    // Create a root scope (no parent).
    Scope() : parent_(nullptr) {}

    // Create a child scope with a parent scope.
    explicit Scope(Scope* parent) : parent_(parent) {}

    // Get the parent scope (nullptr if this is a root scope).
    Scope* GetParent() const { return parent_; }
    void SetParent(Scope* parent) { parent_ = parent; }
    
    // Get the list of statements in this scope.
    std::vector<StatementPtr>& GetStatements() { return statements_; }
    // const overload return non-reference, avoid iterator invalidation if modify statements_ in iteration.
    const std::vector<StatementPtr> GetStatements() const { return statements_; }

    // Add a Statement to this scope.
    void AddStatement(StatementPtr stmt) { statements_.push_back(std::move(stmt)); }

    // Set a variable in the environment table (by name)
    void SetEnvVar(const std::string& name, ValuePtr value);
    
    // Get a variable from the environment table (by name), searching up the scope chain
    ValuePtr GetEnvVar(const std::string& name) const;
    

private:
    Scope* parent_;                         // Pointer to parent scope (nullptr for root)
    std::vector<StatementPtr> statements_;  // Statements in this scope
    std::unordered_map<std::string, ValuePtr> envTable_;  // Environment table: variable name -> latest SSA Value
};

} // namespace pto

