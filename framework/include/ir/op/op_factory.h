// ir/op_factory.h
#pragma once

#include "ir/operation.h"
#include "ir/op/op_opcode.h"

namespace pto {

OperationPtr CreateOp(
    Opcode opcode,
    ValuePtrs inputs,
    ValuePtrs outputs,
    std::shared_ptr<OpPayload> payload = nullptr,
    std::string name = ""
);

} // namespace pto
