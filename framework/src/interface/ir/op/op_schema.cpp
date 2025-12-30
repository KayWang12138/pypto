#include "ir/op/op_schema.h"

#include <array>
#include <cassert>

namespace pto {

static constexpr std::array<OpSchema, 20> kOpSchemaTable = {{
    // ===== Binary tensor ops =====
    {Opcode::OP_ADD, "add",
        OpArity::Binary, 1,
        OpTrait::Elementwise | OpTrait::Commutative | OpTrait::SupportsScalar,
        PayloadKind::None},

    {Opcode::OP_SUB, "sub",
        OpArity::Binary, 1,
        OpTrait::Elementwise | OpTrait::SupportsScalar,
        PayloadKind::None},

    {Opcode::OP_MUL, "mul",
        OpArity::Binary, 1,
        OpTrait::Elementwise | OpTrait::Commutative | OpTrait::SupportsScalar,
        PayloadKind::None},

    {Opcode::OP_DIV, "div",
        OpArity::Binary, 1,
        OpTrait::Elementwise | OpTrait::SupportsScalar,
        PayloadKind::None},

    {Opcode::OP_MAXIMUM, "maximum",
        OpArity::Binary, 1,
        OpTrait::Elementwise | OpTrait::Commutative | OpTrait::SupportsScalar,
        PayloadKind::None},

    {Opcode::OP_MINIMUM, "minimum",
        OpArity::Binary, 1,
        OpTrait::Elementwise | OpTrait::Commutative | OpTrait::SupportsScalar,
        PayloadKind::None},

    // ===== Type conversion =====
    {Opcode::OP_CAST, "cast",
        OpArity::Unary, 1,
        OpTrait::Elementwise,
        PayloadKind::Cast},

    // ===== Unary element-wise ops =====
    {Opcode::OP_EXP, "exp",
        OpArity::Unary, 1,
        OpTrait::Elementwise,
        PayloadKind::None},

    {Opcode::OP_LOG, "log",
        OpArity::Unary, 1,
        OpTrait::Elementwise,
        PayloadKind::None},

    {Opcode::OP_SQRT, "sqrt",
        OpArity::Unary, 1,
        OpTrait::Elementwise,
        PayloadKind::None},

    {Opcode::OP_RSQRT, "rsqrt",
        OpArity::Unary, 1,
        OpTrait::Elementwise,
        PayloadKind::None},

    // ===== Reduction =====
    {Opcode::OP_ROWSUM_SINGLE, "row_sum",
        OpArity::Unary, 1,
        OpTrait::ProducesTensor,
        PayloadKind::Reduce},

    {Opcode::OP_ROWMAX_SINGLE, "row_max",
        OpArity::Unary, 1,
        OpTrait::ProducesTensor,
        PayloadKind::Reduce},

    // ===== View-like =====
    {Opcode::OP_RESHAPE, "reshape",
        OpArity::Unary, 1,
        OpTrait::ProducesTile | OpTrait::ProducesTensor,
        PayloadKind::Reshape},

    {Opcode::OP_ASSEMBLE, "assemble",
        OpArity::Binary, 1,
        OpTrait::ProducesTile | OpTrait::ProducesTensor | OpTrait::Pure,
        PayloadKind::Assemble},

    {Opcode::OP_VIEW, "view",
        OpArity::Unary, 1,
        OpTrait::ProducesTile | OpTrait::ProducesTensor | OpTrait::Pure,
        PayloadKind::View},

    // ===== Matrix operations =====
    {Opcode::OP_MATMUL, "matmul",
        OpArity::Binary, 1,
        OpTrait::ProducesTensor,
        PayloadKind::Matmul},

    {Opcode::OP_BATCH_MATMUL, "batch_matmul",
        OpArity::Binary, 1,
        OpTrait::ProducesTensor,
        PayloadKind::Matmul},

    // ===== Tensor creation =====
    {Opcode::OP_TENSOR_CREATE, "tensor.create",
        OpArity::Nullary, 1,
        OpTrait::ProducesTensor,
        PayloadKind::TensorCreate}
}};


const OpSchema& GetOpSchema(Opcode opcode) {
    for (const auto& schema : kOpSchemaTable) {
        if (schema.opcode == opcode)
            return schema;
    }
    assert(false && "Unknown opcode in GetOpSchema");
    __builtin_unreachable();
}

} // namespace pto
