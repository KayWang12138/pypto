#include "ir/operation.h"
#include "ir/op/op_verify_rule.h"

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

    const auto& schema = GetSchema();

    // ===== results =====
    if (!GetOutputs().empty()) {
        for (size_t i = 0; i < GetOutputs().size(); ++i) {
            os << GetOutputs()[i]->GetSSAName();
            if (i + 1 < GetOutputs().size())
                os << ", ";
        }
        os << " = ";
    }

    // ===== op name =====
    // Determine dialect prefix based on output type
    // - Tensor outputs -> tensor. prefix (Python frontend operations)
    // - Tile outputs -> tile. prefix (compiler optimization operations)
    // - Scalar outputs -> tensor. prefix (element-wise operations)
    if (!GetOutputs().empty()) {
        const auto& firstOutput = GetOutputs()[0];
        if (firstOutput->GetValueKind() == ValueKind::Tile) {
            os << "tile." << schema.name << "";
        } else {
            // Tensor or Scalar -> use tensor. prefix
            os << "tensor." << schema.name << "";
        }
    } else {
        // No outputs (shouldn't happen, but fallback to tensor.)
        os << "tensor." << schema.name << "";
    }

    // ===== operands =====
    os << " ";
    for (size_t i = 0; i < GetInputs().size(); ++i) {
        os << GetInputs()[i]->GetSSAName();
        if (i + 1 < GetInputs().size())
            os << ", ";
    }

    // ===== type signature =====
    os << " : (";
    for (size_t i = 0; i < GetInputs().size(); ++i) {
        GetInputs()[i]->Print(os, 0);
        if (i + 1 < GetInputs().size())
            os << ", ";
    }
    os << ")";

    os << " -> ";
    if (GetOutputs().size() == 1) {
        GetOutputs()[0]->Print(os, 0);
    } else {
        os << "(";
        for (size_t i = 0; i < GetOutputs().size(); ++i) {
            GetOutputs()[i]->Print(os, 0);
            if (i + 1 < GetOutputs().size())
                os << ", ";
        }
        os << ")";
    }

    // ===== payload / attributes =====
    if (payload_) {
        os << " ";
        payload_->Print(os);
    }

    os << "\n";
}



bool Operation::Verify(std::string* err) const {
    const auto& schema = GetSchema();
    // ---- num results ----
    if (GetNumOutputs() != schema.numResults) {
        if (err) *err = "Result count mismatch for op " + std::string(schema.name);
        return false;
    }

    // ---- opcode-specific verify rule ----
    if (const VerifyRule* rule = GetVerifyRuleForOpcode(opcode_)) {
        return rule->Apply(*this, err);
    }
    return true;
}


} // namespace pto
