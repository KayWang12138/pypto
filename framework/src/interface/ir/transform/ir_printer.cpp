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
 * \file ir_printer.cpp
 * \brief Implementation of IR printer
 */

#include "ir/transform/ir_printer.h"
#include "ir/opcode.h"
#include "ir/function.h"

#include <unordered_set>

namespace pto {

// ===== program =====
void IRPrinter::VisitProgram_(ProgramModulePtr& pm) {
  if (!pm) return;

  // Print module header
  PrintIndent();
  os_ << "program.module " << pm->GetPrefixedName() << " {\n";

  // Print program.entry + program-level attrs
  Indent();

  PrintIndent();
  if (auto entry = pm->GetProgramEntry()) {
    os_ << "program.entry " << entry->GetPrefixedName() << "\n";
  }

  const auto& attrs = pm->Attributes();
  if (!attrs.empty()) {
    for (const auto& kv : attrs) {
      PrintIndent();
      os_ << "attr " << kv.first << " = " << kv.second << "\n";
    }
  }

  // Visit all functions (dedup entry to avoid double-print)
  std::unordered_set<const Function*> visited;

  if (auto entry = pm->GetProgramEntry()) {
    if (visited.insert(entry.get()).second) {
      FunctionPtr f = entry;
      VisitFunction_(f);
      os_ << "\n";
    }
  }

  for (const auto& func : pm->GetFunctions()) {
    if (!func) continue;
    if (!visited.insert(func.get()).second) continue;
    FunctionPtr f = func;
    VisitFunction_(f);
    os_ << "\n";
  }

  Dedent();

  // Closing brace
  PrintIndent();
  os_ << "}\n";
}

// ===== function =====
void IRPrinter::VisitFunction_(FunctionPtr& func) {
  if (!func) return;

  // Print function header
  PrintIndent();
  os_ << "func.func " << func->GetPrefixedName() << "(";

  const auto& signature = func->GetSignature();
  for (size_t i = 0; i < signature.arguments.size(); ++i) {
    const auto& arg = signature.arguments[i];
    if (arg) {
      arg->PrintValue(os_);
    }
    if (i + 1 < signature.arguments.size()) {
      os_ << ", ";
    }
  }
  os_ << ")";

  if (!signature.results.empty()) {
    os_ << " -> (";
    for (size_t i = 0; i < signature.results.size(); ++i) {
      if (signature.results[i]) {
        signature.results[i]->PrintType(os_);
      }
      if (i + 1 < signature.results.size()) {
        os_ << ", ";
      }
    }
    os_ << ")";
  }
  os_ << " {\n";

  // Print function body statements
  Indent();
  if (auto compound = func->GetCompound()) {
    StatementPtr st = std::static_pointer_cast<Statement>(compound);
    VisitStmt(st); // will go to VisitStmt_(CompoundStatementPtr&)
  }
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

void IRPrinter::VisitStmt_(OpStatementPtr& st) {
  if (!st) return;

  PrintIndent();
  os_ << "statement.op {\n";

  // Print linear operations.
  Indent();
  const auto& operations = st->Operations();
  for (const auto& op : operations) {
    if (!op) continue;
    PrintIndent();
    OperationPtr opPtr = op;
    VisitOp(opPtr);
  }
  Dedent();

  PrintIndent();
  os_ << "}\n";
}

void IRPrinter::VisitStmt_(IfStatementPtr& is) {
  if (!is) return;

  PrintIndent();

  const auto& results = is->Results();
  if (!results.empty()) {
    for (size_t i = 0; i < results.size(); ++i) {
      if (results[i]) {
        results[i]->PrintSSAName(os_);
      }
      if (i + 1 < results.size()) {
        os_ << ", ";
      }
    }
    os_ << " = ";
  }

  os_ << "statement.if ";
  if (auto cond = is->GetCondition()) {
    cond->PrintSSAName(os_);
  }
  os_ << " {\n";

  Indent();
  if (auto thenCompound = is->GetThenCompound()) {
    StatementPtr thenSt = std::static_pointer_cast<Statement>(thenCompound);
    VisitStmt(thenSt);
  }
  Dedent();

  PrintIndent();
  os_ << "} else {\n";

  Indent();
  if (auto elseCompound = is->GetElseCompound()) {
    StatementPtr elseSt = std::static_pointer_cast<Statement>(elseCompound);
    VisitStmt(elseSt);
  }
  Dedent();

  PrintIndent();
  os_ << "}\n";
}

void IRPrinter::VisitStmt_(ForStatementPtr& fs) {
  if (!fs) return;

  PrintIndent();

  const auto& results = fs->Results();
  if (!results.empty()) {
    for (size_t i = 0; i < results.size(); ++i) {
      if (results[i]) {
        results[i]->PrintSSAName(os_);
      }
      if (i + 1 < results.size()) {
        os_ << ", ";
      }
    }
    os_ << " = ";
  }

  os_ << "statement.for ";
  if (auto iv = fs->GetIterationVar()) {
    iv->PrintSSAName(os_);
  } else {
    os_ << "<null>";
  }

  os_ << " = ";
  if (auto range = fs->GetRange()) {
    if (auto start = range->GetStart()) start->PrintSSAName(os_);
    else os_ << "<null>";

    os_ << " to ";

    if (auto end = range->GetEnd()) end->PrintSSAName(os_);
    else os_ << "<null>";

    os_ << " step ";

    if (auto step = range->GetStep()) step->PrintSSAName(os_);
    else os_ << "<null>";
  } else {
    os_ << "<null> to <null> step <null>";
  }

  const auto& iterArgs = fs->IterArgs();
  if (!iterArgs.empty()) {
    os_ << "\n";
    PrintIndentWithExtra(2);
    os_ << "iter_args(";
    for (size_t i = 0; i < iterArgs.size(); ++i) {
      const auto& arg = iterArgs[i];

      if (arg.value) {
        arg.value->PrintSSAName(os_);
        os_ << " = ";
      } else {
        os_ << "<no-value> = ";
      }

      if (arg.initValue) {
        arg.initValue->PrintValue(os_);
      } else {
        os_ << "<null> : <unknown>";
      }

      if (i + 1 < iterArgs.size()) {
        os_ << ", ";
      }
    }
    os_ << ")";
  }

  const auto& attrs = fs->Attributes();
  if (!attrs.empty()) {
    os_ << "\n";
    PrintIndentWithExtra(2);
    os_ << "attributes {";
    bool first = true;
    for (const auto& kv : attrs) {
      if (!first) os_ << ", ";
      os_ << kv.first << " = " << kv.second;
      first = false;
    }
    os_ << "}";
  }

  os_ << " {\n";

  Indent();
  if (auto compound = fs->GetCompound()) {
    StatementPtr body = std::static_pointer_cast<Statement>(compound);
    VisitStmt(body);
  }
  Dedent();

  PrintIndent();
  os_ << "}\n";
}

void IRPrinter::VisitStmt_(YieldStatementPtr& ys) {
  if (!ys) return;

  PrintIndent();
  os_ << "statement.yield";
  const auto& values = ys->Values();
  if (!values.empty()) {
    os_ << " ";
    PrintValueList(values);
  }
  os_ << "\n";
}

void IRPrinter::VisitStmt_(ReturnStatementPtr& rs) {
  if (!rs) return;

  PrintIndent();
  os_ << "statement.return";
  const auto& values = rs->Values();
  if (!values.empty()) {
    os_ << " ";
    PrintValueList(values);
  }
  os_ << "\n";
}

void IRPrinter::DefaultVisitOp(OperationPtr& op) {
  // ===== results =====
  if (op->GetNumOutputOperand()) {
    for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
      op->GetOutputOperand(i)->PrintSSAName(os_);
      if (i + 1 < op->GetNumOutputOperand()) {
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
  std::string opcodeName = GetOpcodeName(op->GetOpcode());
  if (op->GetNumOutputOperand()) {
    const auto& firstOutput = op->GetOutputOperand(0);
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
  for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
    op->GetInputOperand(i)->PrintSSAName(os_);
    if (i + 1 < op->GetNumInputOperand()) {
      os_ << ", ";
    }
  }

  // ===== type signature =====
  os_ << " : (";
  for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
    op->GetInputOperand(i)->PrintType(os_);
    if (i + 1 < op->GetNumInputOperand()) {
      os_ << ", ";
    }
  }
  os_ << ")";

  os_ << " -> ";
  if (op->GetNumOutputOperand() == 1) {
    op->GetOutputOperand(0)->PrintType(os_);
  } else {
    os_ << "(";
    for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
      op->GetOutputOperand(i)->PrintType(os_);
      if (i + 1 < op->GetNumOutputOperand()) {
        os_ << ", ";
      }
    }
    os_ << ")";
  }

  os_ << "\n";
}

// ===== helpers =====
void IRPrinter::PrintIndent() {
  for (int i = 0; i < indent_; ++i) os_ << "  ";
}

void IRPrinter::PrintIndentWithExtra(int extra) {
  for (int i = 0; i < indent_ + extra; ++i) os_ << "  ";
}

void IRPrinter::Indent() { indent_++; }
void IRPrinter::Dedent() { indent_--; }

void IRPrinter::PrintValueList(const std::vector<ValuePtr>& vals) {
  for (size_t i = 0; i < vals.size(); ++i) {
    if (i > 0) os_ << ", ";
    if (vals[i]) {
      vals[i]->PrintSSAName(os_);
    } else {
      os_ << "<null>";
    }
  }
}

void IRPrinter::PrintAttributes(const AttributeMap& attrs, const std::string& prefix) {
  for (const auto& kv : attrs) {
    os_ << prefix << kv.first << " = " << kv.second << "\n";
  }
}

const char* IRPrinter::ToString(FunctionKind kind) {
  switch (kind) {
    case FunctionKind::ControlFlow: return "control_flow";
    case FunctionKind::DataFlow:    return "data_flow";
    case FunctionKind::Kernel:      return "kernel";
    default:                        return "unknown";
  }
}

} // namespace pto

