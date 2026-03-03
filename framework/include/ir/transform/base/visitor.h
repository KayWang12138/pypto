/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PYPTO_IR_TRANSFORMS_BASE_VISITOR_H_
#define PYPTO_IR_TRANSFORMS_BASE_VISITOR_H_

#include "ir/stmt.h"
#include "ir/transform/base/functor.h"

namespace pypto {
namespace ir {

/**
 * @brief Read-only IR visitor for both expressions and statements
 *
 * Provides default implementations that recursively traverse the IR tree.
 * Subclasses can override specific VisitExpr_ or VisitStmt_ methods to implement custom behavior.
 * All methods don't modify the visited IR nodes.
 */
class IRVisitor : public IRFunctor<void> {
 public:
  ~IRVisitor() override = default;

  void VisitExpr(const ExprPtr& expr) override;
  void VisitStmt(const StmtPtr& stmt) override;

 protected:
  PYPTO_DECLARE_ALL_VISITOR_OVERRIDES

 private:
  /**
   * @brief Helper to visit both children of a binary expression
   */
  void VisitBinaryOp_(const BinaryExprPtr& op);

  /**
   * @brief Helper to visit the operand of a unary expression
   */
  void VisitUnaryOp_(const UnaryExprPtr& op);
};

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_BASE_VISITOR_H_
