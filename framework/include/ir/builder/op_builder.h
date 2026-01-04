// ir/op_builder.h
#pragma once

#include "ir/value.h"
#include "ir/op/op_opcode.h"
#include "ir/op/op_payload.h"

namespace pto {

class IRBuilder; // forward decl (avoid include cycle)

ValuePtrs BuildBySchema(
    IRBuilder& builder,
    Opcode opcode,
    ValuePtrs inputs,
    std::shared_ptr<OpPayload> payload = nullptr,
    std::string name = "");
} // namespace pto
