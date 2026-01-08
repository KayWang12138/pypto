// ir/ir_opcode_formatter.h
#pragma once

#include <ostream>

#include "ir/ir_visitor.h"
#include "ir/operation.h"
#include "ir/value.h"
#include "ir/opcode.h"

namespace pto {

class IROpcodeFormatter : public IRVisitor<IROpcodeFormatter> {
public:
    explicit IROpcodeFormatter(std::ostream& os)
        : os_(os) {}

    void Print(Operation& op) {
        Visit(op);   // CRTP dispatch via IRVisitor
    }

    void VisitOperation(Operation& op) {
        PrintGeneric(op);
    }

private:

    // ===== generic printer =====
    void PrintGeneric(Operation& op) {
        // ===== results =====
        if (op.GetNumOutputOperand()) {
            for (size_t i = 0; i < op.GetNumOutputOperand(); ++i) {
                op.GetOutputOperand(i)->PrintSSAName(os_);
                if (i + 1 < op.GetNumOutputOperand()) {
                    os_ << ", ";
                }
            }
            os_ << " = ";
        }

        // ===== op name =====
        // Determine dialect prefix based on output type
        // - Tensor outputs -> tensor. prefix (Python frontend operations)
        // - Tile outputs -> tile. prefix (compiler optimization operations)
        // - Scalar outputs -> tensor. prefix (element-wise operations)
        std::string opcodeName = GetOpcodeName(op.GetOpcode());
        if (op.GetNumOutputOperand()) {
            const auto& firstOutput = op.GetOutputOperand(0);
            if (firstOutput->GetValueKind() == ValueKind::Tile) {
                os_ << "tile." << opcodeName;
            } else {
                // Tensor or Scalar -> use tensor. prefix
                os_ << "tensor." << opcodeName;
            }
        } else {
            // No outputs (shouldn't happen, but fallback to tensor.)
            os_ << "tensor." << opcodeName;
        }

        // ===== operands =====
        os_ << " ";
        for (size_t i = 0; i < op.GetNumInputOperand(); ++i) {
            op.GetInputOperand(i)->PrintSSAName(os_);
            if (i + 1 < op.GetNumInputOperand()) {
                os_ << ", ";
            }
        }

        // ===== type signature =====
        os_ << " : (";
        for (size_t i = 0; i < op.GetNumInputOperand(); ++i) {
            op.GetInputOperand(i)->PrintType(os_);
            if (i + 1 < op.GetNumInputOperand()) {
                os_ << ", ";
            }
        }
        os_ << ")";

        os_ << " -> ";
        if (op.GetNumOutputOperand() == 1) {
            op.GetOutputOperand(0)->PrintType(os_);
        } else {
            os_ << "(";
            for (size_t i = 0; i < op.GetNumOutputOperand(); ++i) {
                op.GetOutputOperand(i)->PrintType(os_);
                if (i + 1 < op.GetNumOutputOperand()) {
                    os_ << ", ";
                }
            }
            os_ << ")";
        }

        os_ << "\n";
    }

private:
    std::ostream& os_;
};

} // namespace pto