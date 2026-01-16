/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file printer.cpp
 * \brief Implementation of IR printer
 */

#include "ir/transform/printer.h"
#include "ir/opcode.h"
#include "ir/function.h"
#include "ir/value.h"
#include "ir/utils.h"

namespace pto {

// ===== program =====
void IRPrinter::VisitImplProgram(ProgramModulePtr &pm) {
    if (!pm)return;

    // Print module header
    PrintIndent();
    os_ << "program.module " << pm->GetPrefixedName() << " {\n";
    // Print program.entry + program-level attrs
    Indent();

    PrintIndent();
    if (auto entry = pm->GetProgramEntry()) {
        os_ << "program.entry " << entry->GetPrefixedName() << "\n";
    }

    const auto &attrs = pm->Attributes();
    for (const auto &kv : attrs) {
        PrintIndent();
        os_ << "attr " << kv.first << " = " << kv.second << "\n";
    }
    
    for (const auto &func : pm->GetFunctions()) {
        FunctionPtr f = func;
        VisitImplFunction(f);
        os_ << "\n";
    }
    Dedent();

    // Closing brace
    PrintIndent();
    os_ << "}\n";
}

// ===== function =====
void IRPrinter::VisitImplFunction(FunctionPtr &func) {
    // Print function header
    PrintIndent();
    os_ << "func.func " << func->GetPrefixedName() << "(";

    const auto &signature = func->GetSignature();
    for (size_t i = 0; i < signature.arguments.size(); ++i) {
        const auto &arg = signature.arguments[i];
        PrintValue(arg);
        if (i + 1 < signature.arguments.size()) {
            os_ << ", ";
        }
    }
    os_ << ")";
    
    os_ << " -> (";
    for (size_t i = 0; i < signature.results.size(); ++i) {
        PrintType(signature.results[i]);
        if (i + 1 < signature.results.size()) {
            os_ << ", ";
        }
    }
    os_ << ") {\n";
    // Print function body statements
    Indent();
    auto compound = func->GetCompound();
    VisitImplStmt(compound);
    Dedent();

    // Closing brace
    PrintIndent();
    os_ << "}\n";

    // Print function attributes
    std::string attrPrefix = "func.attr ";
    PrintAttributes(func->Attributes(), attrPrefix);

    // Print function kind comment
    PrintIndent();
    os_ << "// func.kind = " << ToString(func->GetKind()) << "\n";
}

// ===== statements =====

void IRPrinter::VisitImplStmt(OpStatementPtr &st) {
    PrintIndent();
    os_ << "statement.op {\n";

    // Print linear operations.
    Indent();
    const auto &operations = st->Operations();
    for (const auto &op : operations) {
        PrintIndent();
        OperationPtr opPtr = op;
        VisitOp(opPtr);
    }
    Dedent();

    PrintIndent();
    os_ << "}\n";
}

void IRPrinter::VisitImplStmt(IfStatementPtr &is) {
    PrintIndent();

    const auto &results = is->Results();
    for (size_t i = 0; i < results.size(); ++i) {
        PrintSSAName(results[i]);
        if (i + 1 < results.size()) {
            os_ << ", ";
        }
    }
    os_ << " = ";
    
    os_ << "statement.if ";
    PrintSSAName(is->GetCondition());
    os_ << " {\n";

    Indent();
    auto thenCompound = is->GetThenCompound();
    VisitImplStmt(thenCompound);
    Dedent();

    PrintIndent();
    os_ << "} else {\n";

    Indent();
    auto elseCompound = is->GetElseCompound();
    VisitImplStmt(elseCompound);
    Dedent();

    PrintIndent();
    os_ << "}\n";
}

void IRPrinter::VisitImplStmt(ForStatementPtr &fs) {
    PrintIndent();
    const auto &results = fs->Results();
    
    for (size_t i = 0; i < results.size(); ++i) {
        PrintSSAName(results[i]);
        if (i + 1 < results.size()) {
            os_ << ", ";
        }
    }
    os_ << " = ";

    os_ << "statement.for ";
    PrintSSAName(fs->GetIterationVar());
    os_ << " = ";
    auto range = fs->GetRange();
    PrintSSAName(range->GetStart());
    os_ << " to ";
    PrintSSAName(range->GetEnd());
    os_ << " step ";
    PrintSSAName(range->GetStep());

    const auto &iterArgs = fs->IterArgs();
    
    os_ << "\n";
    PrintIndentWithExtra(2);
    os_ << "iter_args(";
    for (size_t i = 0; i < iterArgs.size(); ++i) {
        const auto &arg = iterArgs[i];
        PrintSSAName(arg.value);
        os_ << " = ";
        PrintValue(arg.initValue);
        
        if (i + 1 < iterArgs.size()) {
            os_ << ", ";
        }
    }
    os_ << ")";
    

    const auto &attrs = fs->Attributes();
    os_ << "\n";
    PrintIndentWithExtra(2);
    os_ << "attributes {";
    bool first = true;
    for (const auto &kv : attrs) {
        if (!first)
            os_ << ", ";
        os_ << kv.first << " = " << kv.second;
        first = false;
    }
    os_ << "}";

    os_ << " {\n";

    Indent();
    auto compound = fs->GetCompound();
    VisitImplStmt(compound);
    Dedent();

    PrintIndent();
    os_ << "}\n";
}

void IRPrinter::VisitImplStmt(YieldStatementPtr &ys) {
    PrintIndent();
    os_ << "statement.yield";
    const auto &values = ys->Values();
    if (!values.empty()) {
        os_ << " ";
        PrintValueList(values);
    }
    os_ << "\n";
}

void IRPrinter::VisitImplStmt(ReturnStatementPtr &rs) {
    PrintIndent();
    os_ << "statement.return";
    const auto &values = rs->Values();
    if (!values.empty()) {
        os_ << " ";
        PrintValueList(values);
    }
    os_ << "\n";
}

void IRPrinter::VisitImplOp(OperationPtr &op) {

    // ===== results =====
    for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
        PrintSSAName(op->GetOutputOperand(i));
        if (i + 1 < op->GetNumOutputOperand()) {
            os_ << ", ";
        }
    }
    os_ << " = ";
    
    // ===== op name =====
    // Determine dialect prefix based on output type
    // - Tensor outputs -> tensor. prefix (Python frontend operations)
    // - Tile outputs -> tile. prefix (compiler optimization operations)
    // - Scalar outputs -> tensor. prefix (element-wise operations)
    std::string opcodeName = GetOpcodeName(op->GetOpcode());
    
    const auto &firstOutput = op->GetOutputOperand(0);
    if (firstOutput->GetValueKind() == ValueKind::Tile) {
        os_ << "tile." << opcodeName;
    } else if (firstOutput->GetValueKind() == ValueKind::Scalar){
        // Tensor or Scalar -> use tensor. prefix
        os_ << "scalar." << opcodeName;
    }else {
        os_ << "tensor." << opcodeName;
    }

    // ===== operands =====
    os_ << " ";
    for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
        PrintSSAName(op->GetInputOperand(i));
        if (i + 1 < op->GetNumInputOperand()) {
            os_ << ", ";
        }
    }

    // ===== type signature =====
    os_ << " : (";
    for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
        PrintType(op->GetInputOperand(i));
        if (i + 1 < op->GetNumInputOperand()) {
            os_ << ", ";
        }
    }
    os_ << ")";

    os_ << " -> (";
    for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
        PrintType(op->GetOutputOperand(i));
        if (i + 1 < op->GetNumOutputOperand()) {
            os_ << ", ";
        }
    }
    os_ << ")";
    

    os_ << "\n";
}

// ===== values =====
void IRPrinter::VisitImplValue(ScalarValuePtr &value) {
    PrintValue(value);
}

void IRPrinter::VisitImplValue(TileValuePtr &value) {
    PrintValue(value);
}

void IRPrinter::VisitImplValue(TensorValuePtr &value) {
    PrintValue(value);
}

void IRPrinter::VisitImplValue(ValuePtr &value) {
    PrintValue(value);
}

// ===== helpers =====
void IRPrinter::PrintIndent() {
    for (int i = 0; i < indent_; ++i)
        os_ << "  ";
}

void IRPrinter::PrintIndentWithExtra(int extra) {
    for (int i = 0; i < indent_ + extra; ++i)
        os_ << "  ";
}

void IRPrinter::Indent() {
    indent_++;
}
void IRPrinter::Dedent() {
    indent_--;
}

void IRPrinter::PrintValueList(const std::vector<ValuePtr> &vals) {
    for (size_t i = 0; i < vals.size(); ++i) {
        if (i > 0)
            os_ << ", ";
        PrintSSAName(vals[i]);
    }
}

void IRPrinter::PrintAttributes(const AttributeMap &attrs, const std::string &prefix) {
    for (const auto &kv : attrs) {
        os_ << prefix << kv.first << " = " << kv.second << "\n";
    }
}

const char *IRPrinter::ToString(FunctionKind kind) {
    switch (kind) {
        case FunctionKind::ControlFlow: return "control_flow";
        case FunctionKind::DataFlow: return "data_flow";
        case FunctionKind::Block: return "block";
        default: return "unknown";
    }
}

// ===== value printing methods =====
void IRPrinter::PrintSSAName(ValuePtr value) {
    if (auto scalar = std::dynamic_pointer_cast<ScalarValue>(value)) {
        switch (scalar->GetScalarValueKind()) {
            case ScalarValueKind::Immediate:
                // For immediate values, print the actual constant value
                std::visit([this](const auto &val) { os_ << val; }, scalar->GetImmediateValue());
                break;
            case ScalarValueKind::Symbolic: os_ << scalar->GetSSAName(); break;
            default: os_ << "Unknown ScalarValue";
        }
    } else if (auto tensor = std::dynamic_pointer_cast<TensorValue>(value)) {
        os_ << tensor->GetSSAName();
    } else if (auto tile = std::dynamic_pointer_cast<TileValue>(value)) {
        os_ << tile->GetSSAName();
    } else {
        os_ << value->GetSSAName();
    }
}

void IRPrinter::PrintValue(ValuePtr value) {
    PrintSSAName(value);
    os_ << ": ";
    PrintType(value);
}

void IRPrinter::PrintType(ValuePtr value) {
    if (auto scalar = std::dynamic_pointer_cast<ScalarValue>(value)) {
        scalar->GetType()->Print(os_);
    } else if (auto tensor = std::dynamic_pointer_cast<TensorValue>(value)) {
        os_ << "tensor<";
        // ====== shape ======
        os_ << "[";
        const auto &shape = tensor->GetShape();
        for (size_t i = 0; i < shape.size(); ++i) {
            PrintSSAName(shape[i]);
            if (i + 1 < shape.size()) {
                os_ << ", ";
            }
        }
        os_ << "]";
        // ====== type ======
        os_ << ", ";
        os_ << DTypeInfoOf(tensor->GetDataType()).name;
        os_ << ">";
    } else if (auto tile = std::dynamic_pointer_cast<TileValue>(value)) {
        os_ << "tile<[";
        // ====== valid shape ======
        const auto &validShapes = tile->GetValidShape();
        const auto &shape = tile->GetShape();
        for (size_t i = 0; i < validShapes.size(); ++i) {
            PrintSSAName(validShapes[i]);
            if (i + 1 < shape.size()) {
                os_ << ", ";
            }
        }
        os_ << "], [";
        // ====== tile shapes ======
        for (size_t i = 0; i < shape.size(); ++i) {
            os_ << shape[i];
            if (i + 1 < shape.size()) {
                os_ << ", ";
            }
        }
        os_ << "], ";
        // ====== type ======
        os_ << DTypeInfoOf(tile->GetDataType()).name;
        os_ << ">";
    } else {
        if (auto type = value->GetType()) {
            type->Print(os_);
        } else {
            os_ << "unknown";
        }
    }
}

} // namespace pto
