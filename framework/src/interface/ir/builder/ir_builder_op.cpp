// ir/builder/ir_builder_op.cpp
// PTO-IR prototype: IRBuilder op emission + schema-driven Create().

#include "ir/builder/ir_builder.h"
#include <utility>

namespace pto {

Operation& IRBuilder::Emit(OperationPtr op) {
    auto& blk = GetOrCreateActiveBlock();
    blk.Operations().push_back(std::move(op));
    return *blk.Operations().back();
}

ValuePtrs IRBuilder::CreateOp(Opcode opcode,
                              ValuePtrs inputs,
                              std::shared_ptr<OpPayload> payload,
                              std::string name) {
    // Delegate to schema/traits but keep caller-provided outputs.
    return BuildBySchema(*this,
                         opcode,
                         std::move(inputs),
                         std::move(payload),
                         name);
}

} // namespace pto
