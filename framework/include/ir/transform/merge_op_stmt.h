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
 * \file merge_op_stmt.h
 * \brief Pass to merge consecutive OpStatements in CompoundStatements
 */

#pragma once

#include "ir/transform/functor.h"
#include "ir/statement.h"

namespace pto {

/**
 * @brief Functor to merge consecutive OpStatements in CompoundStatements
 *
 * This pass traverses CompoundStatements and merges consecutive OpStatements
 * into a single OpStatement by combining their operations.
 */
class MergeOpStmt : public StatementFunctor<StatementPtr> {
 public:
  MergeOpStmt() = default;
  virtual ~MergeOpStmt() = default;

  // Override VisitStmt_ to handle CompoundStatement
  StatementPtr VisitStmt_(CompoundStatementPtr& stmt) override;
  
  // Override other statement types to recursively visit nested statements
  StatementPtr VisitStmt_(OpStatementPtr& stmt) override;
  StatementPtr VisitStmt_(ForStatementPtr& stmt) override;
  StatementPtr VisitStmt_(IfStatementPtr& stmt) override;
  StatementPtr VisitStmt_(YieldStatementPtr& stmt) override;
  StatementPtr VisitStmt_(ReturnStatementPtr& stmt) override;
  StatementPtr VisitStmt_(StatementPtr& stmt) override;

 private:
  /**
   * @brief Merge consecutive OpStatements in a list of statements
   *
   * @param statements The list of statements to process
   * @return A new list with consecutive OpStatements merged
   */
  std::vector<StatementPtr> MergeConsecutiveOpStmts(const std::vector<StatementPtr>& statements);
};

} // namespace pto

