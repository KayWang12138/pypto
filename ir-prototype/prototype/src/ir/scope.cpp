// PTO-IR prototype: scope management implementation.

#include "ir/scope.h"
#include <string>

namespace pto {

ValuePtr Scope::FindValue(const std::string& name) const {
    // Use GetEnvVar which already searches through the scope chain
    return GetEnvVar(name);
}

void Scope::RemoveValue(ValuePtr val) {
    for (auto it = envTable_.begin(); it != envTable_.end();) {
        if (it->second == val) {
            it = envTable_.erase(it);
            return;
        } else {
            ++it;
        }
    }
}

std::unordered_map<std::string, ValuePtr> Scope::GetAncestorValues() const {
    std::unordered_map<std::string, ValuePtr> ancestor_values;
    
    // Traverse all ancestor scopes (parent, grandparent, etc.)
    Scope* current_parent = parent_;
    while (current_parent != nullptr) {
        // Collect all values from current ancestor scope's environment table
        const auto& parent_env = current_parent->GetEnvTable();
        for (const auto& pair : parent_env) {
            if (pair.second) {
                // If a variable with the same name exists in multiple ancestor scopes,
                // the closer one (more recent ancestor) takes precedence
                if (ancestor_values.find(pair.first) == ancestor_values.end()) {
                    ancestor_values[pair.first] = pair.second;
                }
            }
        }
        
        // Move to next ancestor
        current_parent = current_parent->GetParent();
    }
    
    return ancestor_values;
}

void Scope::SetEnvVar(const std::string& name, ValuePtr value) {
    if (!value) {
        throw std::runtime_error("Scope::SetEnvVar: value is null");
    }
    envTable_[name] = value;
}

ValuePtr Scope::GetEnvVar(const std::string& name) const {
    // First, search in current scope
    auto it = envTable_.find(name);
    if (it != envTable_.end()) {
        return it->second;
    }
    
    // If not found, search in parent scope (recursively)
    if (parent_) {
        return parent_->GetEnvVar(name);
    }
    
    // Not found in any scope
    return nullptr;
}

} // namespace pto

