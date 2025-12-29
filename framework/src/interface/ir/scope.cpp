// PTO-IR prototype: scope management implementation.

#include "ir/scope.h"
#include <string>

namespace pto {

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

}