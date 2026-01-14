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
 * \file visitor.cpp
 * \brief Implementation of IRVisitor methods
 */

#include "ir/transform/visitor.h"

namespace pto {

// ----------------- Public Visit* entrypoints -----------------

void IRVisitor::VisitImplProgram(ProgramModulePtr &program) {
    if (!program)
        return;

    // Visit all functions
    for (auto func : program->GetFunctions()) {
        FunctionPtr funcPtr = func;
        VisitFunction(funcPtr);
    }
}

void IRVisitor::VisitImplFunction(FunctionPtr &func) {
    if (!func)
        return;

    // Visit signature arguments
    const auto &sig = func->GetSignature();
    for (const auto &arg : sig.arguments) {
        ValuePtr argPtr = arg;
        VisitValue(argPtr);
    }

    // Visit signature results
    for (const auto &result : sig.results) {
        ValuePtr resultPtr = result;
        VisitValue(resultPtr);
    }

    // Visit input compound
    if (auto inputCompound = func->GetInputCompound()) {
        VisitImplStmt(inputCompound);
    }

    // Visit function body compound
    if (auto compound = func->GetCompound()) {
        VisitImplStmt(compound);
    }
}

void IRVisitor::VisitImplStmt(CompoundStatementPtr &stmt) {
    if (!stmt)
        return;

    for (size_t i = 0; i < stmt->GetStatementsNum(); ++i) {
        StatementPtr stmtPtr = stmt->GetStatement(i);
        VisitStmt(stmtPtr);
    }
}

void IRVisitor::VisitImplStmt(OpStatementPtr &stmt) {
    if (!stmt)
        return;

    const auto &ops = stmt->Operations();
    for (const auto &op : ops) {
        OperationPtr opPtr = op;
        VisitOp(opPtr);
    }
}

void IRVisitor::VisitImplStmt(ForStatementPtr &stmt) {
    if (!stmt)
        return;

    // Visit iteration variable
    if (auto iterVar = stmt->GetIterationVar()) {
        VisitImplValue(iterVar);
    }

    // Visit range values
    if (auto range = stmt->GetRange()) {
        if (auto start = range->GetStart()) {
            VisitImplValue(start);
        }
        if (auto end = range->GetEnd()) {
            VisitImplValue(end);
        }
        if (auto step = range->GetStep()) {
            VisitImplValue(step);
        }
    }

    // Visit iter args
    const auto &iterArgs = stmt->IterArgs();
    for (const auto &iterArg : iterArgs) {
        if (iterArg.initValue) {
            ValuePtr initPtr = iterArg.initValue;
            VisitValue(initPtr);
        }
        if (iterArg.value) {
            ValuePtr valuePtr = iterArg.value;
            VisitValue(valuePtr);
        }
    }

    // Visit loop body compound
    if (auto compound = stmt->GetCompound()) {
        VisitImplStmt(compound);
    }

    // Visit results
    const auto &results = stmt->Results();
    for (const auto &result : results) {
        ValuePtr resultPtr = result;
        VisitValue(resultPtr);
    }
}

void IRVisitor::VisitImplStmt(IfStatementPtr &stmt) {
    if (!stmt)
        return;

    // Visit condition
    if (auto condition = stmt->GetCondition()) {
        VisitImplValue(condition);
    }

    // Visit then branch
    if (auto thenCompound = stmt->GetThenCompound()) {
        VisitImplStmt(thenCompound);
    }

    // Visit else branch
    if (auto elseCompound = stmt->GetElseCompound()) {
        VisitImplStmt(elseCompound);
    }

    // Visit results
    const auto &results = stmt->Results();
    for (const auto &result : results) {
        ValuePtr resultPtr = result;
        VisitValue(resultPtr);
    }
}

void IRVisitor::VisitImplStmt(YieldStatementPtr &stmt) {
    if (!stmt)
        return;

    const auto &values = stmt->Values();
    for (const auto &value : values) {
        ValuePtr valuePtr = value;
        VisitValue(valuePtr);
    }
}

void IRVisitor::VisitImplStmt(ReturnStatementPtr &stmt) {
    if (!stmt)
        return;

    const auto &values = stmt->Values();
    for (const auto &value : values) {
        ValuePtr valuePtr = value;
        VisitValue(valuePtr);
    }
}

void IRVisitor::VisitImplStmt(StatementPtr &stmt) {
    // Fallback for unknown statement types: no children to visit.
    (void)stmt;
}

// ---- Concrete ops (auto-generated from *.def) ----
#define DEFOP(name, inherit, opcode, ...)                             \
    void IRVisitor::VisitImplOp(name##Ptr &op) {                         \
        OperationPtr opPtr = std::static_pointer_cast<Operation>(op); \
        VisitImplOp(opPtr);                                              \
    }
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

void IRVisitor::VisitImplOp(OperationPtr &op) {
    if (!op)
        return;

    // Visit input operands
    for (size_t i = 0; i < op->GetNumInputOperand(); ++i) {
        ValuePtr operandPtr = op->GetInputOperand(i);
        VisitValue(operandPtr);
    }

    // Visit output operands
    for (size_t i = 0; i < op->GetNumOutputOperand(); ++i) {
        ValuePtr operandPtr = op->GetOutputOperand(i);
        VisitValue(operandPtr);
    }
}

void IRVisitor::VisitImplValue(ScalarValuePtr &value) {
    // ScalarValue is a leaf node
    (void)value;
}

void IRVisitor::VisitImplValue(TileValuePtr &value) {
    // TileValue treated as leaf in this IR
    (void)value;
}

void IRVisitor::VisitImplValue(TensorValuePtr &value) {
    if (!value)
        return;

    const auto &shape = value->GetShape();
    for (const auto &shapeElem : shape) {
        ValuePtr shapePtr = std::static_pointer_cast<Value>(shapeElem);
        VisitValue(shapePtr);
    }
}

void IRVisitor::VisitImplValue(ValuePtr &value) {
    // Fallback for unknown value types
    (void)value;
}

} // namespace pto
