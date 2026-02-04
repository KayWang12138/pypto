/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ir/transform/base/visitor.h"

#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

void IRVisitor::VisitExpr(const ExprPtr& expr) {
  ExprFunctor<void>::VisitExpr(expr);
}

void IRVisitor::VisitStmt(const StmtPtr& stmt) {
  StmtFunctor<void>::VisitStmt(stmt);
}

// Leaf nodes - no children to visit
void IRVisitor::VisitExpr_(const VarPtr& op) {
  (void)op;
}

void IRVisitor::VisitExpr_(const IterArgPtr& op) {
  (void)op;
}

void IRVisitor::VisitExpr_(const ConstIntPtr& op) {
  (void)op;
}

void IRVisitor::VisitExpr_(const ConstFloatPtr& op) {
  (void)op;
}

void IRVisitor::VisitExpr_(const ConstBoolPtr& op) {
  (void)op;
}

void IRVisitor::VisitExpr_(const CallPtr& op) {
  (void)op;
}

void IRVisitor::VisitExpr_(const TupleGetItemExprPtr& op) {
  (void)op;
}

// Helper methods for binary and unary operations
void IRVisitor::VisitBinaryOp_(const BinaryExprPtr& op) {
  (void)op;
}

void IRVisitor::VisitUnaryOp_(const UnaryExprPtr& op) {
  (void)op;
}

// Binary operations
void IRVisitor::VisitExpr_(const AddPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const SubPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const MulPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const FloorDivPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const FloorModPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const FloatDivPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const MinPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const MaxPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const PowPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const EqPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const NePtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const LtPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const LePtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const GtPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const GePtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const AndPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const OrPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const XorPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const BitAndPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const BitOrPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const BitXorPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const BitShiftLeftPtr& op) { VisitBinaryOp_(op); }
void IRVisitor::VisitExpr_(const BitShiftRightPtr& op) { VisitBinaryOp_(op); }

// Unary operations
void IRVisitor::VisitExpr_(const AbsPtr& op) { VisitUnaryOp_(op); }
void IRVisitor::VisitExpr_(const NegPtr& op) { VisitUnaryOp_(op); }
void IRVisitor::VisitExpr_(const NotPtr& op) { VisitUnaryOp_(op); }
void IRVisitor::VisitExpr_(const BitNotPtr& op) { VisitUnaryOp_(op); }
void IRVisitor::VisitExpr_(const CastPtr& op) { VisitUnaryOp_(op); }

// Statement types
void IRVisitor::VisitStmt_(const AssignStmtPtr& op) {
  (void)op;
}

void IRVisitor::VisitStmt_(const IfStmtPtr& op) {
  (void)op;
}

void IRVisitor::VisitStmt_(const YieldStmtPtr& op) {
  (void)op;
}

void IRVisitor::VisitStmt_(const ReturnStmtPtr& op) {
  (void)op;
}

void IRVisitor::VisitStmt_(const ForStmtPtr& op) {
  (void)op;
}

void IRVisitor::VisitStmt_(const SeqStmtsPtr& op) {
  (void)op;
}

void IRVisitor::VisitStmt_(const OpStmtsPtr& op) {
  (void)op;
}

void IRVisitor::VisitStmt_(const StmtPtr& op) {
  (void)op;
}

}  // namespace ir
}  // namespace pypto
