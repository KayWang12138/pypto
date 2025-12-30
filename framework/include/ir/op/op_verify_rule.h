#pragma once

#include <string>
#include "ir/op/op_opcode.h"

namespace pto {

class Operation;

class VerifyRule {
public:
    virtual ~VerifyRule() = default;

    virtual bool Apply(const Operation& op, std::string* err) const = 0;
};

const VerifyRule* GetVerifyRuleForOpcode(Opcode op);

} // namespace pto
