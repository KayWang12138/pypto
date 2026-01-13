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
    using IRVisitor::VisitOp_;
    using IRVisitor::VisitStmt_;
    using IRVisitor::VisitValue_;

    // ===== program =====
    void VisitProgram_(ProgramModulePtr &pm) override;

    // ===== function =====
    void VisitFunction_(FunctionPtr &func) override;

    // ===== statements =====
    void VisitStmt_(OpStatementPtr &st) override;
    void VisitStmt_(IfStatementPtr &is) override;
    void VisitStmt_(ForStatementPtr &fs) override;
    void VisitStmt_(YieldStatementPtr &ys) override;
    void VisitStmt_(ReturnStatementPtr &rs) override;

    // ===== operations =====
    void VisitOp_(OperationPtr &op) override;

    // ===== values =====
    void VisitValue_(ScalarValuePtr &value) override;
    void VisitValue_(TileValuePtr &value) override;
    void VisitValue_(TensorValuePtr &value) override;
    void VisitValue_(ValuePtr &value) override;

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
