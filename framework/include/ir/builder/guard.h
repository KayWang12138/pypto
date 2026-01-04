#pragma once

#include "ir/function.h"
namespace pto {

class IRBuilder;
class CompoundStatement;

class ScopeGuard {
public:
    ScopeGuard(IRBuilder& builder, std::shared_ptr<CompoundStatement> new_scope, std::shared_ptr<Function> new_func = nullptr);
    ~ScopeGuard();

private:
    IRBuilder& builder_;
    std::shared_ptr<CompoundStatement> prev_compound_;
    std::shared_ptr<Function> prev_func_;
};
}
