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

#include "ir/transform/base/mutator.h"

#include <memory>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

ExprPtr IRMutator::VisitExpr(const ExprPtr &expr) {
    // Call the base class VisitExpr which returns ExprPtr
    return ExprFunctor<ExprPtr>::VisitExpr(expr);
}

StmtPtr IRMutator::VisitStmt(const StmtPtr &stmt) {
    // Call the base class VisitStmt which returns StmtPtr
    return StmtFunctor<StmtPtr>::VisitStmt(stmt);
}

// Leaf nodes - return original shared_ptr (immutable)
ExprPtr IRMutator::VisitExpr_(const VarPtr &op) {
    // Var is immutable, return original
    return op;
}

ExprPtr IRMutator::VisitExpr_(const IterArgPtr &op) {
    // Visit initValue as Expr
    INTERNAL_CHECK(op->initValue_) << "IterArg has null initValue";
    auto newInitValue = ExprFunctor<ExprPtr>::VisitExpr(op->initValue_);
    INTERNAL_CHECK(newInitValue) << "IterArg initValue mutated to null";
    // Copy-on-write: only create new node if children changed
    if (newInitValue.get() != op->initValue_.get()) {
        return std::make_shared<const IterArg>(op->name_, op->GetType(), std::move(newInitValue), op->span_);
    } else {
        return op;
    }
}

ExprPtr IRMutator::VisitExpr_(const MemRefPtr &op) {
    // MemRef is immutable, return original
    return op;
}

ExprPtr IRMutator::VisitExpr_(const ConstIntPtr &op) {
    // ConstInt is immutable, return original
    return op;
}

ExprPtr IRMutator::VisitExpr_(const ConstFloatPtr &op) {
    // ConstFloat is immutable, return original
    return op;
}

ExprPtr IRMutator::VisitExpr_(const ConstBoolPtr &op) {
    // ConstBool is immutable, return original
    return op;
}

ExprPtr IRMutator::VisitExpr_(const CallPtr &op) {
    // Visit all arguments
    std::vector<ExprPtr> newArgs;
    bool changed = false;
    newArgs.reserve(op->args_.size());

    for (size_t i = 0; i < op->args_.size(); ++i) {
        INTERNAL_CHECK(op->args_[i]) << "Call has null argument at index " << i;
        auto newArg = ExprFunctor<ExprPtr>::VisitExpr(op->args_[i]);
        INTERNAL_CHECK(newArg) << "Call argument at index " << i << " mutated to null";
        newArgs.push_back(newArg);
        if (newArg.get() != op->args_[i].get()) {
            changed = true;
        }
    }

    // Copy-on-write: only create new node if arguments changed
    if (changed) {
        // Preserve original type and kwargs when reconstructing the Call node
        return std::make_shared<const Call>(op->op_, std::move(newArgs), op->kwargs_, op->GetType(), op->span_);
    } else {
        return op;
    }
}

ExprPtr IRMutator::VisitExpr_(const MakeTuplePtr &op) {
    // Visit all element expressions
    std::vector<ExprPtr> newElements;
    newElements.reserve(op->elements_.size());
    bool changed = false;

    for (const auto &elem : op->elements_) {
        INTERNAL_CHECK(elem) << "MakeTuple has null element";
        auto newElem = ExprFunctor<ExprPtr>::VisitExpr(elem);
        INTERNAL_CHECK(newElem) << "MakeTuple element mutated to null";
        newElements.push_back(newElem);
        if (newElem.get() != elem.get()) {
            changed = true;
        }
    }

    // Copy-on-write: only create new node if elements changed
    if (changed) {
        return std::make_shared<const MakeTuple>(std::move(newElements), op->span_);
    } else {
        return op;
    }
}

ExprPtr IRMutator::VisitExpr_(const TupleGetItemExprPtr &op) {
    // Visit the tuple expression
    INTERNAL_CHECK(op->tuple_) << "TupleGetItemExpr has null tuple";
    auto newTuple = ExprFunctor<ExprPtr>::VisitExpr(op->tuple_);
    INTERNAL_CHECK(newTuple) << "TupleGetItemExpr tuple mutated to null";

    // Copy-on-write: only create new node if tuple changed
    if (newTuple.get() != op->tuple_.get()) {
        return std::make_shared<const TupleGetItemExpr>(newTuple, op->index_, op->span_);
    } else {
        return op;
    }
}

// Macro to generate binary operation mutators with copy-on-write
#define DEFINE_BINARY_MUTATOR(OpType)                                                       \
    ExprPtr IRMutator::VisitExpr_(const OpType##Ptr &op) {                                  \
        INTERNAL_CHECK(op->left_) << #OpType " has null left operand";                      \
        INTERNAL_CHECK(op->right_) << #OpType " has null right operand";                    \
        auto new_left = ExprFunctor<ExprPtr>::VisitExpr(op->left_);                         \
        auto new_right = ExprFunctor<ExprPtr>::VisitExpr(op->right_);                       \
        INTERNAL_CHECK(new_left) << #OpType " left operand mutated to null";                \
        INTERNAL_CHECK(new_right) << #OpType " right operand mutated to null";              \
        auto scalar_type = As<ScalarType>(op->GetType());                                   \
        INTERNAL_CHECK(scalar_type) << #OpType " has null type";                            \
        if (new_left.get() != op->left_.get() || new_right.get() != op->right_.get()) {     \
            return std::make_shared<const OpType>(                                          \
                std::move(new_left), std::move(new_right), scalar_type->dtype_, op->span_); \
        } else {                                                                            \
            return op;                                                                      \
        }                                                                                   \
    }

// Binary operations
DEFINE_BINARY_MUTATOR(Add)
DEFINE_BINARY_MUTATOR(Sub)
DEFINE_BINARY_MUTATOR(Mul)
DEFINE_BINARY_MUTATOR(FloorDiv)
DEFINE_BINARY_MUTATOR(FloorMod)
DEFINE_BINARY_MUTATOR(FloatDiv)
DEFINE_BINARY_MUTATOR(Min)
DEFINE_BINARY_MUTATOR(Max)
DEFINE_BINARY_MUTATOR(Pow)
DEFINE_BINARY_MUTATOR(Eq)
DEFINE_BINARY_MUTATOR(Ne)
DEFINE_BINARY_MUTATOR(Lt)
DEFINE_BINARY_MUTATOR(Le)
DEFINE_BINARY_MUTATOR(Gt)
DEFINE_BINARY_MUTATOR(Ge)
DEFINE_BINARY_MUTATOR(And)
DEFINE_BINARY_MUTATOR(Or)
DEFINE_BINARY_MUTATOR(Xor)
DEFINE_BINARY_MUTATOR(BitAnd)
DEFINE_BINARY_MUTATOR(BitOr)
DEFINE_BINARY_MUTATOR(BitXor)
DEFINE_BINARY_MUTATOR(BitShiftLeft)
DEFINE_BINARY_MUTATOR(BitShiftRight)

#undef DEFINE_BINARY_MUTATOR

// Macro to generate unary operation mutators with copy-on-write
#define DEFINE_UNARY_MUTATOR(OpType)                                                                       \
    ExprPtr IRMutator::VisitExpr_(const OpType##Ptr &op) {                                                 \
        INTERNAL_CHECK(op->operand_) << #OpType " has null operand";                                       \
        auto new_operand = ExprFunctor<ExprPtr>::VisitExpr(op->operand_);                                  \
        INTERNAL_CHECK(new_operand) << #OpType " operand mutated to null";                                 \
        auto scalar_type = As<ScalarType>(op->GetType());                                                  \
        INTERNAL_CHECK(scalar_type) << #OpType " has null type";                                           \
        if (new_operand.get() != op->operand_.get()) {                                                     \
            return std::make_shared<const OpType>(std::move(new_operand), scalar_type->dtype_, op->span_); \
        } else {                                                                                           \
            return op;                                                                                     \
        }                                                                                                  \
    }

// Unary operations
DEFINE_UNARY_MUTATOR(Abs)
DEFINE_UNARY_MUTATOR(Neg)
DEFINE_UNARY_MUTATOR(Not)
DEFINE_UNARY_MUTATOR(BitNot)
DEFINE_UNARY_MUTATOR(Cast)

#undef DEFINE_UNARY_MUTATOR

// Statement types
StmtPtr IRMutator::VisitStmt_(const AssignStmtPtr &op) {
    INTERNAL_CHECK(op->var_) << "AssignStmt has null var";
    INTERNAL_CHECK(op->value_) << "AssignStmt has null value";
    auto newVarExpr = ExprFunctor<ExprPtr>::VisitExpr(op->var_);
    auto newValue = ExprFunctor<ExprPtr>::VisitExpr(op->value_);
    INTERNAL_CHECK(newVarExpr) << "AssignStmt var mutated to null";
    INTERNAL_CHECK(newValue) << "AssignStmt value mutated to null";
    // Cast new_var from ExprPtr to VarPtr (required by AssignStmt constructor)
    auto newVar = As<Var>(newVarExpr);
    INTERNAL_CHECK(newVar) << "AssignStmt var is not a Var after mutation";
    if (newVar.get() != op->var_.get() || newValue.get() != op->value_.get()) {
        return std::make_shared<const AssignStmt>(std::move(newVar), std::move(newValue), op->span_);
    } else {
        return op;
    }
}

StmtPtr IRMutator::VisitStmt_(const IfStmtPtr &op) {
    INTERNAL_CHECK(op->condition_) << "IfStmt has null condition";
    auto newCondition = ExprFunctor<ExprPtr>::VisitExpr(op->condition_);
    INTERNAL_CHECK(newCondition) << "IfStmt condition mutated to null";

    INTERNAL_CHECK(op->thenBody_) << "IfStmt has null then_body";
    auto newThenBody = StmtFunctor<StmtPtr>::VisitStmt(op->thenBody_);
    INTERNAL_CHECK(newThenBody) << "IfStmt then_body mutated to null";
    bool thenChanged = (newThenBody.get() != op->thenBody_.get());

    std::optional<StmtPtr> newElseBody;
    bool elseChanged = false;
    if (op->elseBody_.has_value()) {
        INTERNAL_CHECK(*op->elseBody_) << "IfStmt has null else_body";
        auto newStmt = StmtFunctor<StmtPtr>::VisitStmt(*op->elseBody_);
        INTERNAL_CHECK(newStmt) << "IfStmt else_body mutated to null";
        newElseBody = newStmt;
        if (newStmt.get() != op->elseBody_->get()) {
            elseChanged = true;
        }
    }

    std::vector<VarPtr> newReturnVars;
    bool returnVarsChanged = false;
    newReturnVars.reserve(op->returnVars_.size());
    for (size_t i = 0; i < op->returnVars_.size(); ++i) {
        INTERNAL_CHECK(op->returnVars_[i]) << "IfStmt has null return_vars at index " << i;
        auto newVarExpr = ExprFunctor<ExprPtr>::VisitExpr(op->returnVars_[i]);
        INTERNAL_CHECK(newVarExpr) << "IfStmt return_vars at index " << i << " mutated to null";
        // Cast new_var from ExprPtr to VarPtr (required by IfStmt constructor)
        auto newVar = As<Var>(newVarExpr);
        INTERNAL_CHECK(newVar) << "IfStmt return_vars at index " << i << " is not a Var after mutation";
        newReturnVars.push_back(newVar);
        if (newVar.get() != op->returnVars_[i].get()) {
            returnVarsChanged = true;
        }
    }

    if (newCondition.get() != op->condition_.get() || thenChanged || elseChanged || returnVarsChanged) {
        if (newElseBody.has_value()) {
            return std::make_shared<const IfStmt>(
                std::move(newCondition), std::move(newThenBody), *newElseBody, std::move(newReturnVars), op->span_);
        } else {
            return std::make_shared<const IfStmt>(
                std::move(newCondition), std::move(newThenBody), std::nullopt, std::move(newReturnVars), op->span_);
        }
    } else {
        return op;
    }
}

StmtPtr IRMutator::VisitStmt_(const YieldStmtPtr &op) {
    std::vector<ExprPtr> newValue;
    bool changed = false;
    newValue.reserve(op->value_.size());

    for (size_t i = 0; i < op->value_.size(); ++i) {
        INTERNAL_CHECK(op->value_[i]) << "YieldStmt has null value at index " << i;
        auto newExpr = ExprFunctor<ExprPtr>::VisitExpr(op->value_[i]);
        INTERNAL_CHECK(newExpr) << "YieldStmt value at index " << i << " mutated to null";
        newValue.push_back(newExpr);
        if (newExpr.get() != op->value_[i].get()) {
            changed = true;
        }
    }

    if (changed) {
        return std::make_shared<const YieldStmt>(std::move(newValue), op->span_);
    } else {
        return op;
    }
}

StmtPtr IRMutator::VisitStmt_(const ReturnStmtPtr &op) {
    std::vector<ExprPtr> newValue;
    bool changed = false;
    newValue.reserve(op->value_.size());

    for (size_t i = 0; i < op->value_.size(); ++i) {
        INTERNAL_CHECK(op->value_[i]) << "ReturnStmt has null value at index " << i;
        auto newExpr = ExprFunctor<ExprPtr>::VisitExpr(op->value_[i]);
        INTERNAL_CHECK(newExpr) << "ReturnStmt value at index " << i << " mutated to null";
        newValue.push_back(newExpr);
        if (newExpr.get() != op->value_[i].get()) {
            changed = true;
        }
    }

    if (changed) {
        return std::make_shared<const ReturnStmt>(std::move(newValue), op->span_);
    } else {
        return op;
    }
}

StmtPtr IRMutator::VisitStmt_(const ForStmtPtr &op) {
    INTERNAL_CHECK(op->loopVar_) << "ForStmt has null loop_var";
    INTERNAL_CHECK(op->start_) << "ForStmt has null start";
    INTERNAL_CHECK(op->stop_) << "ForStmt has null stop";
    INTERNAL_CHECK(op->step_) << "ForStmt has null step";
    auto newLoopVarExpr = ExprFunctor<ExprPtr>::VisitExpr(op->loopVar_);
    INTERNAL_CHECK(newLoopVarExpr) << "ForStmt loop_var mutated to null";
    auto newLoopVar = As<Var>(newLoopVarExpr);
    INTERNAL_CHECK(newLoopVar) << "ForStmt loop_var is not a Var after mutation";

    auto newStart = ExprFunctor<ExprPtr>::VisitExpr(op->start_);
    INTERNAL_CHECK(newStart) << "ForStmt start mutated to null";

    auto newStop = ExprFunctor<ExprPtr>::VisitExpr(op->stop_);
    INTERNAL_CHECK(newStop) << "ForStmt stop mutated to null";

    auto newStep = ExprFunctor<ExprPtr>::VisitExpr(op->step_);
    INTERNAL_CHECK(newStep) << "ForStmt step mutated to null";

    std::vector<IterArgPtr> newIterArgs;
    bool iterArgsChanged = false;
    newIterArgs.reserve(op->iterArgs_.size());
    for (size_t i = 0; i < op->iterArgs_.size(); ++i) {
        INTERNAL_CHECK(op->iterArgs_[i]) << "ForStmt has null iter_args at index " << i;
        auto newIterArgExpr = ExprFunctor<ExprPtr>::VisitExpr(op->iterArgs_[i]);
        INTERNAL_CHECK(newIterArgExpr) << "ForStmt iter_args at index " << i << " mutated to null";
        auto newIterArg = As<IterArg>(std::static_pointer_cast<const IRNode>(newIterArgExpr));
        INTERNAL_CHECK(newIterArg) << "ForStmt iter_args at index " << i << " is not an IterArg after mutation";
        newIterArgs.push_back(newIterArg);
        if (newIterArg.get() != op->iterArgs_[i].get()) {
            iterArgsChanged = true;
        }
    }

    INTERNAL_CHECK(op->body_) << "ForStmt has null body";
    auto newBody = StmtFunctor<StmtPtr>::VisitStmt(op->body_);
    INTERNAL_CHECK(newBody) << "ForStmt body mutated to null";
    bool bodyChanged = (newBody.get() != op->body_.get());

    std::vector<VarPtr> newReturnVars;
    bool returnVarsChanged = false;
    newReturnVars.reserve(op->returnVars_.size());
    for (size_t i = 0; i < op->returnVars_.size(); ++i) {
        INTERNAL_CHECK(op->returnVars_[i]) << "ForStmt has null return_vars at index " << i;
        auto newVarExpr = ExprFunctor<ExprPtr>::VisitExpr(op->returnVars_[i]);
        INTERNAL_CHECK(newVarExpr) << "ForStmt return_vars at index " << i << " mutated to null";
        // Cast new_var from ExprPtr to VarPtr (required by ForStmt constructor)
        auto newVar = As<Var>(newVarExpr);
        INTERNAL_CHECK(newVar) << "ForStmt return_vars at index " << i << " is not a Var after mutation";
        newReturnVars.push_back(newVar);
        if (newVar.get() != op->returnVars_[i].get()) {
            returnVarsChanged = true;
        }
    }

    if (newLoopVar.get() != op->loopVar_.get() || newStart.get() != op->start_.get() ||
        newStop.get() != op->stop_.get() || newStep.get() != op->step_.get() || iterArgsChanged || bodyChanged ||
        returnVarsChanged) {
        return std::make_shared<const ForStmt>(std::move(newLoopVar), std::move(newStart), std::move(newStop),
            std::move(newStep), std::move(newIterArgs), std::move(newBody), std::move(newReturnVars), op->span_);
    } else {
        return op;
    }
}

StmtPtr IRMutator::VisitStmt_(const SeqStmtsPtr &op) {
    std::vector<StmtPtr> newStmts;
    bool changed = false;
    newStmts.reserve(op->stmts_.size());
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
        INTERNAL_CHECK(op->stmts_[i]) << "SeqStmts has null statement at index " << i;
        auto newStmt = StmtFunctor<StmtPtr>::VisitStmt(op->stmts_[i]);
        INTERNAL_CHECK(newStmt) << "SeqStmts statement at index " << i << " mutated to null";
        newStmts.push_back(newStmt);
        if (newStmt.get() != op->stmts_[i].get()) {
            changed = true;
        }
    }

    if (changed) {
        return std::make_shared<const SeqStmts>(std::move(newStmts), op->span_);
    } else {
        return op;
    }
}

StmtPtr IRMutator::VisitStmt_(const OpStmtsPtr &op) {
    std::vector<StmtPtr> newStmts;
    bool changed = false;
    newStmts.reserve(op->stmts_.size());
    for (size_t i = 0; i < op->stmts_.size(); ++i) {
        INTERNAL_CHECK(op->stmts_[i]) << "OpStmts has null statement at index " << i;
        auto newStmt = StmtFunctor<StmtPtr>::VisitStmt(op->stmts_[i]);
        INTERNAL_CHECK(newStmt) << "OpStmts statement at index " << i << " mutated to null";
        // Verify it's still an AssignStmt or EvalStmt after mutation
        auto kind = newStmt->GetKind();
        INTERNAL_CHECK(kind == ObjectKind::AssignStmt || kind == ObjectKind::EvalStmt)
            << "OpStmts statement at index " << i << " is not an AssignStmt or EvalStmt after mutation";
        newStmts.push_back(newStmt);
        if (newStmt.get() != op->stmts_[i].get()) {
            changed = true;
        }
    }

    if (changed) {
        return std::make_shared<const OpStmts>(std::move(newStmts), op->span_);
    } else {
        return op;
    }
}

StmtPtr IRMutator::VisitStmt_(const EvalStmtPtr &op) {
    INTERNAL_CHECK(op->expr_) << "EvalStmt has null expr";
    auto newExpr = ExprFunctor<ExprPtr>::VisitExpr(op->expr_);
    INTERNAL_CHECK(newExpr) << "EvalStmt expr mutated to null";

    if (newExpr.get() != op->expr_.get()) {
        return std::make_shared<const EvalStmt>(std::move(newExpr), op->span_);
    } else {
        return op;
    }
}

StmtPtr IRMutator::VisitStmt_(const StmtPtr &op) {
    // Base Stmt is immutable, return original
    return op;
}

} // namespace ir
} // namespace pypto
