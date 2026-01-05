#include "ir/operation.h"

namespace pto {

Operation::Operation()
    : Object(ObjectType::Operation),
      opcode_(Opcode::OP_INVALID) {}

Operation::Operation(Opcode opcode)
    : Object(ObjectType::Operation),
      opcode_(opcode) {}

Operation::Operation(Opcode opcode, std::string name)
    : Object(ObjectType::Operation, std::move(name)),
      opcode_(opcode) {}

Operation::Operation(Opcode opcode,
                     ValuePtrs inputs,
                     ValuePtrs outputs,
                     std::string name)
    : Object(ObjectType::Operation, std::move(name)),
      ioprands_(std::move(inputs)),
      ooprands_(std::move(outputs)),
      opcode_(opcode) {}


void Operation::Print(std::ostream& os, int indent) const {
    PrintIndent(os, indent);

    // ===== results =====
    if (GetNumOutputOperand()) {
        for (size_t i = 0; i < GetNumOutputOperand(); ++i) {
            os << GetOutputOperand(i)->GetSSAName();
            if (i + 1 < GetNumOutputOperand())
                os << ", ";
        }
        os << " = ";
    }

    // ===== op name =====
    // Determine dialect prefix based on output type
    // - Tensor outputs -> tensor. prefix (Python frontend operations)
    // - Tile outputs -> tile. prefix (compiler optimization operations)
    // - Scalar outputs -> tensor. prefix (element-wise operations)
    std::string opcodeName = GetOpcodeName(GetOpcode());
    if (GetNumOutputOperand()) {
        const auto& firstOutput = GetOutputOperand(0);
        if (firstOutput->GetValueKind() == ValueKind::Tile) {
            os << "tile." << opcodeName << "";
        } else {
            // Tensor or Scalar -> use tensor. prefix
            os << "tensor." << opcodeName << "";
        }
    } else {
        // No outputs (shouldn't happen, but fallback to tensor.)
        os << "tensor." << opcodeName << "";
    }

    // ===== operands =====
    os << " ";
    for (size_t i = 0; i < GetNumInputOperand(); ++i) {
        os << GetInputOperand(i)->GetSSAName();
        if (i + 1 < GetNumInputOperand())
            os << ", ";
    }

    // ===== type signature =====
    os << " : (";
    for (size_t i = 0; i < GetNumInputOperand(); ++i) {
        GetInputOperand(i)->Print(os, 0);
        if (i + 1 < GetNumInputOperand())
            os << ", ";
    }
    os << ")";

    os << " -> ";
    if (GetNumOutputOperand() == 1) {
        GetOutputOperand(0)->Print(os, 0);
    } else {
        os << "(";
        for (size_t i = 0; i < GetNumOutputOperand(); ++i) {
            GetOutputOperand(i)->Print(os, 0);
            if (i + 1 < GetNumOutputOperand())
                os << ", ";
        }
        os << ")";
    }

    os << "\n";
}

} // namespace pto
