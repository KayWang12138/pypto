#include "ir/builder/ir_builder.h"
#include "ir/function.h"
#include "ir/builder/guard.h"

namespace pto {

ScopeGuard::ScopeGuard(IRBuilder& builder, std::shared_ptr<CompoundStatement> new_scope, std::shared_ptr<Function> new_func)
    : builder_(builder),
      prev_compound_(builder.compound_),
      prev_func_(builder_.func_) {
    builder_.compound_ = new_scope;
    builder_.opStmt_ = nullptr;
    if (new_func) {
      builder_.func_ = new_func;
    }
}

ScopeGuard::~ScopeGuard() {
    builder_.compound_ = prev_compound_;
    builder_.opStmt_ = nullptr;
    builder_.func_ = prev_func_;
}
}
