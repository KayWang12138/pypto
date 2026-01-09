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
 * \file unroll_static_for.h
 * \brief Pass to unroll static for loops (loops with constant bounds)
 */

#pragma once

#include "ir/transform/mutator.h"
#include "ir/statement.h"
#include "ir/value.h"

namespace pto {

/**
 * @brief Mutator to unroll static for loops
 *
 * This pass identifies for loops with constant bounds (static loops) and
 * unrolls them by replacing the loop with multiple copies of the loop body.
 */
class UnrollStaticFor : public IRMutator {
 public:
  UnrollStaticFor() = default;
  virtual ~UnrollStaticFor() = default;

  // Override VisitFunction_ to handle function body and ensure unrolled statements are added
  FunctionPtr VisitFunction_(FunctionPtr& func) override;

  // Override VisitStmt_ to handle ForStatement, CompoundStatement, and OpStatement
  StatementPtr VisitStmt_(ForStatementPtr& stmt) override;
  StatementPtr VisitStmt_(CompoundStatementPtr& stmt) override;
  StatementPtr VisitStmt_(OpStatementPtr& stmt) override;
  
  // Explicitly declare base class method to avoid hiding
  using IRMutator::VisitStmt_;

 private:
  // Track current compound during traversal to access envTable
  CompoundStatementPtr currentCompound_{nullptr};

  /**
   * @brief Check if a for loop is static (has constant bounds)
   *
   * @param stmt The for statement to check
   * @return true if the loop is static, false otherwise
   */
  bool IsStaticForLoop(ForStatementPtr& stmt);

  /**
   * @brief Get the iteration count for a static for loop
   *
   * @param stmt The for statement
   * @return The number of iterations, or -1 if not static
   */
  int64_t GetIterationCount(ForStatementPtr& stmt);

  /**
   * @brief Unroll a static for loop by creating multiple copies of the body
   *
   * @param stmt The for statement to unroll
   * @param iterationCount The number of iterations
   * @return A compound statement containing the unrolled loop body
   */
  StatementPtr UnrollLoop(ForStatementPtr& stmt, int64_t iterationCount);

  /**
   * @brief Create a new value for unrolled iteration
   *
   * @param originalValue The original value to clone
   * @param iteration The iteration number
   * @param outputIndex The output index in the operation
   * @return A new value with updated name
   */
  ValuePtr CreateNewValue(ValuePtr originalValue, int64_t iteration, size_t outputIndex);

  /**
   * @brief Update environment variables after unrolling a for loop
   *
   * @param compound The compound statement containing the unrolled loop
   * @param originalForStmt The original for statement that was unrolled
   * @param unrolledOpStmt The unrolled OpStatement
   */
  void UpdateEnvVarsAfterUnroll(CompoundStatementPtr compound, ForStatementPtr originalForStmt, OpStatementPtr unrolledOpStmt);
};

} // namespace pto

