/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
  void VisitProgram_(ProgramModulePtr& program) override;

  // Function visitor methods
  void VisitFunction_(FunctionPtr& func) override;

  // Statement visitor methods
  void VisitStmt_(CompoundStatementPtr& stmt) override;
  void VisitStmt_(OpStatementPtr& stmt) override;
  void VisitStmt_(ForStatementPtr& stmt) override;
  void VisitStmt_(IfStatementPtr& stmt) override;
  void VisitStmt_(YieldStatementPtr& stmt) override;
  void VisitStmt_(ReturnStatementPtr& stmt) override;
  void VisitStmt_(StatementPtr& stmt) override;

  // Operation visitor methods
  void VisitOp_(ScalarBaseOpPtr& op) override;
  void VisitOp_(OperationPtr& op) override;

  // Value visitor methods
  void VisitValue_(ScalarValuePtr& value) override;
  void VisitValue_(TileValuePtr& value) override;
  void VisitValue_(TensorValuePtr& value) override;
  void VisitValue_(ValuePtr& value) override;
};

} // namespace pto
