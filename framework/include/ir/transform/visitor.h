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
 * \file visitor.h
 * \brief Visitor classes for read-only traversal of IR nodes
 */

#pragma once

#include "ir/transform/functor.h"

namespace pto {

/**
 * @brief Read-only visitor for IR nodes
 *
 * Provides default empty implementations for all visit methods.
 * Subclasses can override specific methods to implement custom traversal logic.
 */
class IRVisitor : public IRFunctor<void, void, void, void, void> {
public:
    virtual ~IRVisitor() = default;

    // Program visitor methods
    void VisitImplProgram(ProgramModulePtr &program) override;

    // Function visitor methods
    void VisitImplFunction(FunctionPtr &func) override;

    // Statement visitor methods
    void VisitImplStmt(CompoundStatementPtr &stmt) override;
    void VisitImplStmt(OpStatementPtr &stmt) override;
    void VisitImplStmt(ForStatementPtr &stmt) override;
    void VisitImplStmt(IfStatementPtr &stmt) override;
    void VisitImplStmt(YieldStatementPtr &stmt) override;
    void VisitImplStmt(ReturnStatementPtr &stmt) override;
    void VisitImplStmt(StatementPtr &stmt) override;

// Operation visitor methods
// Concrete ops (auto-generated from *.def)
#define DEFOP(name, inherit, opcode, ...) void VisitImplOp(name##Ptr &op) override;
#include "ir/operation.def"
#include "ir/tile_graph.def"
#undef DEFOP

    void VisitImplOp(OperationPtr &op) override;

    // Value visitor methods
    void VisitImplValue(ScalarValuePtr &value) override;
    void VisitImplValue(TileValuePtr &value) override;
    void VisitImplValue(TensorValuePtr &value) override;
    void VisitImplValue(ValuePtr &value) override;
};

} // namespace pto
