#pragma once

namespace pto {

enum class Opcode {
    OP_INVALID,
    
    // Binary tensor ops
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MAXIMUM,
    OP_MINIMUM,

    // Type conversion
    OP_CAST,

    // Unary element-wise ops
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_RSQRT,

    // Reduction 
    OP_ROWMAX_SINGLE,
    OP_ROWSUM_SINGLE,

    // View
    OP_RESHAPE,
    OP_ASSEMBLE,
    OP_VIEW,

    // Matrix operations
    OP_MATMUL,
    OP_BATCH_MATMUL,

    // Tensor creation
    OP_TENSOR_CREATE
};

inline const char* ToString(Opcode op) {
    switch (op) {
        case Opcode::OP_INVALID: return "invalid";
        case Opcode::OP_ADD:     return "add";
        case Opcode::OP_SUB:     return "sub";
        case Opcode::OP_MUL:     return "mul";
        case Opcode::OP_DIV:     return "div";
        case Opcode::OP_MAXIMUM: return "maximum";
        case Opcode::OP_MINIMUM: return "minimum";

        case Opcode::OP_CAST:    return "cast";

        case Opcode::OP_EXP:     return "exp";
        case Opcode::OP_LOG:     return "log";
        case Opcode::OP_SQRT:    return "sqrt";
        case Opcode::OP_RSQRT:   return "rsqrt";

        case Opcode::OP_ROWMAX_SINGLE:    return "row_max";
        case Opcode::OP_ROWSUM_SINGLE:    return "row_sum";

        case Opcode::OP_RESHAPE:    return "reshape";
        case Opcode::OP_ASSEMBLE:   return "assemble";
        case Opcode::OP_VIEW:       return "view";

        case Opcode::OP_MATMUL:         return "matmul";
        case Opcode::OP_BATCH_MATMUL:   return "batch_matmul";

        case Opcode::OP_TENSOR_CREATE:  return "tensor.create";

        default:              return "unknown";
    }
}

} // namespace pto
