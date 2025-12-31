#pragma once

#include "ir/function.h"
namespace pto {

class IRBuilder;
class CompoundStatement;

class ScopeGuard {
public:
    ScopeGuard(IRBuilder& builder, CompoundStatement* new_scope, Function* new_func = nullptr);
    ~ScopeGuard();

private:
    IRBuilder& builder_;
    CompoundStatement* prev_compound_;
    Function* prev_func_;
};
}
