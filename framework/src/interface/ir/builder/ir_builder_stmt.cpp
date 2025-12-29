#include "ir/builder/ir_builder.h"

#include <stdexcept>
#include <utility>
#include <unordered_set>
#include <functional>

namespace pto {

BlockStatement& IRBuilder::CreateBlockStmt() {
    if (!scope_) throw std::runtime_error("IRBuilder::CreateBlockStmt: scope is null");

    auto blk = std::make_shared<BlockStatement>();
    auto& ref = *blk;
    scope_->AddStatement(std::move(blk));
    block_ = &ref;
    return ref;
}

// ===== Enter nested scopes =====
std::shared_ptr<ScopeGuard> IRBuilder::EnterFunctionBody(Function& func) {
    
    Scope& scope = func.GetScope();
    
    return std::make_shared<ScopeGuard>(*this, &scope, &func);
}


} // namespace pto
