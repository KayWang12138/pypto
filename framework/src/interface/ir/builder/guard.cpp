#include "ir/builder/ir_builder.h"
#include "ir/function.h"
#include "ir/builder/guard.h"

namespace pto {

ScopeGuard::ScopeGuard(IRBuilder& builder, Scope* new_scope, Function* new_func)
    : builder_(builder),
      prev_scope_(builder.scope_),
      prev_func_(builder_.func_) {
    builder_.scope_ = new_scope;
    builder_.opStmt_ = nullptr;
    if (new_func) {
      builder_.func_ = new_func;
    }
}

ScopeGuard::~ScopeGuard() {
    builder_.scope_ = prev_scope_;
    builder_.opStmt_ = nullptr;
    builder_.func_ = prev_func_;
}
}
