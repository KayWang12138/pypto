#pragma once

#include "ir/function.h"
namespace pto {

class IRBuilder;
class Scope;

class ScopeGuard {
public:
    ScopeGuard(IRBuilder& builder, Scope* new_scope, Function* new_func = nullptr);
    ~ScopeGuard();

private:
    IRBuilder& builder_;
    Scope* prev_scope_;
    Function* prev_func_;
};
}
