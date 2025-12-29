#pragma once

#include "ir/op/op_opcode.h"
#include "ir/op/op_payload.h"

#include <cstdint>

namespace pto {

enum class OpArity : uint8_t {
    Nullary = 0,
    Unary   = 1,
    Binary  = 2,
    Variadic
};

enum OpTrait : uint32_t {
    None            = 0,
    Elementwise     = 1u << 0,
    Commutative     = 1u << 1,
    SupportsScalar  = 1u << 2,

    ProducesScalar  = 1u << 3,

    // NEW
    ProducesTile    = 1u << 4,
    ProducesTensor  = 1u << 5,
    Pure            = 1u << 6,
};

constexpr OpTrait operator|(OpTrait a, OpTrait b) noexcept {
    return static_cast<OpTrait>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    );
}

inline bool HasTrait(uint32_t traits, OpTrait t) {
    return (traits & static_cast<uint32_t>(t)) != 0;
}

struct OpSchema {
    // required
    Opcode opcode;
    const char* name;

    OpArity arity;
    uint8_t numResults;

    uint32_t traits;
    PayloadKind requiredPayload;
};


const OpSchema& GetOpSchema(Opcode opcode);

} // namespace pto
