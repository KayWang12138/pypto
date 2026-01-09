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
 * \file mutator.h
 * \brief Mutator classes for transforming IR nodes
 */

#pragma once

#include "ir/transform/functor.h"

namespace pto {

/**
 * @brief Mutator for IR nodes
 *
 * Provides default implementations that return the original object.
 * Subclasses can override specific methods to return modified objects.
 */
class IRMutator : public IRFunctor<ProgramModulePtr, FunctionPtr, StatementPtr,
                                   OperationPtr, ValuePtr> {
 public:
  virtual ~IRMutator() = default;

  // Program mutator methods
  ProgramModulePtr VisitProgram_(ProgramModulePtr& program) override;

  // Function mutator methods
  FunctionPtr VisitFunction_(FunctionPtr& func) override;

  // Statement mutator methods
  StatementPtr VisitStmt_(CompoundStatementPtr& stmt) override;
  StatementPtr VisitStmt_(OpStatementPtr& stmt) override;
  StatementPtr VisitStmt_(ForStatementPtr& stmt) override;
  StatementPtr VisitStmt_(IfStatementPtr& stmt) override;
  StatementPtr VisitStmt_(YieldStatementPtr& stmt) override;
  StatementPtr VisitStmt_(ReturnStatementPtr& stmt) override;
  StatementPtr VisitStmt_(StatementPtr& stmt) override;

  // Operation mutator methods
  OperationPtr VisitOp_(ScalarBaseOpPtr& op) override;
  OperationPtr VisitOp_(OperationPtr& op) override;

  // Value mutator methods
  ValuePtr VisitValue_(ScalarValuePtr& value) override;
  ValuePtr VisitValue_(TileValuePtr& value) override;
  ValuePtr VisitValue_(TensorValuePtr& value) override;
  ValuePtr VisitValue_(ValuePtr& value) override;
};

} // namespace pto
