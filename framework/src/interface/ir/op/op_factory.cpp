// ir/op_factory.cpp
#include "ir/op/op_factory.h"

#include <sstream>

namespace pto {

OperationPtr CreateOp(
    Opcode opcode,
    ValuePtrs inputs,
    ValuePtrs outputs,
    std::shared_ptr<OpPayload> payload,
    std::string name) {

    auto op = std::make_shared<Operation>(
        opcode,
        std::move(inputs),
        std::move(outputs),
        std::move(name)
    );

    if (payload) {
        op->SetPayload(std::move(payload));
    }

    // 3. Verify（schema + payload）
    std::string err;
    if (!op->Verify(&err)) {
        std::ostringstream oss;
        oss << "CreateOp failed for opcode `"
            << ToString(opcode)
            << "`: " << err;
        throw std::runtime_error(oss.str());
    }

    return op;
}

} // namespace pto
