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
 * \file printer.h
 * \brief
 */

#pragma once

#include <ostream>

#include "ir/transform/visitor.h"
#include "ir/object.h"

namespace pto {

class IRPrinter : public IRVisitor {
public:
    explicit IRPrinter(std::ostream &os) : os_(os) {}

    // Bring in all overloads from the base class to prevent hiding
    using IRVisitor::VisitImplOp;
    using IRVisitor::VisitImplStmt;
    using IRVisitor::VisitImplValue;

    // ===== program =====
    void VisitImplProgram(ProgramModulePtr &pm) override;

    // ===== function =====
    void VisitImplFunction(FunctionPtr &func) override;

    // ===== statements =====
    void VisitImplStmt(OpStatementPtr &st) override;
    void VisitImplStmt(IfStatementPtr &is) override;
    void VisitImplStmt(ForStatementPtr &fs) override;
    void VisitImplStmt(YieldStatementPtr &ys) override;
    void VisitImplStmt(ReturnStatementPtr &rs) override;

    // ===== operations =====
    void VisitImplOp(OperationPtr &op) override;

    // ===== values =====
    void VisitImplValue(ScalarValuePtr &value) override;
    void VisitImplValue(TileValuePtr &value) override;
    void VisitImplValue(TensorValuePtr &value) override;
    void VisitImplValue(ValuePtr &value) override;

private:
    // ===== helpers =====
    void PrintIndent();
    void PrintIndentWithExtra(int extra);
    void Indent();
    void Dedent();
    void PrintValueList(const std::vector<ValuePtr> &vals);
    void PrintAttributes(const AttributeMap &attrs, const std::string &prefix);

    // ===== value printing methods =====
    void PrintSSAName(ValuePtr value);
    void PrintValue(ValuePtr value);
    void PrintType(ValuePtr value);

    const char *ToString(FunctionKind kind);

private:
    std::ostream &os_;
    int indent_{0};
};

} // namespace pto
