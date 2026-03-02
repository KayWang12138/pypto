/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mutator.cpp
 * \brief Unit tests for IR mutator (copy-on-write transformation)
 */

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/kind_traits.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/base/mutator.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

class IRMutatorTest : public testing::Test {};

// ============================================================================
// Identity Mutator Tests (no changes, should return same pointers)
// ============================================================================

TEST_F(IRMutatorTest, TestIdentityMutatorConstInt) {
    IRMutator mutator;
    auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto result = mutator.VisitExpr(expr);
    ASSERT_EQ(result.get(), expr.get()); // Same pointer (no copy)
}

TEST_F(IRMutatorTest, TestIdentityMutatorVar) {
    IRMutator mutator;
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
    auto result = mutator.VisitExpr(var);
    ASSERT_EQ(result.get(), var.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorAdd) {
    IRMutator mutator;
    auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto add = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());

    auto result = mutator.VisitExpr(add);
    ASSERT_EQ(result.get(), add.get()); // No change, same pointer
}

TEST_F(IRMutatorTest, TestIdentityMutatorEvalStmt) {
    IRMutator mutator;
    auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto stmt = std::make_shared<EvalStmt>(expr, Span::unknown());

    auto result = mutator.VisitStmt(stmt);
    ASSERT_EQ(result.get(), stmt.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorAssignStmt) {
    IRMutator mutator;
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
    auto val = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
    auto stmt = std::make_shared<AssignStmt>(var, val, Span::unknown());

    auto result = mutator.VisitStmt(stmt);
    ASSERT_EQ(result.get(), stmt.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorSeqStmts) {
    IRMutator mutator;
    auto expr1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto eval1 = std::make_shared<EvalStmt>(expr1, Span::unknown());
    auto expr2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto eval2 = std::make_shared<EvalStmt>(expr2, Span::unknown());

    std::vector<StmtPtr> stmts = {eval1, eval2};
    auto seq = std::make_shared<SeqStmts>(stmts, Span::unknown());

    auto result = mutator.VisitStmt(seq);
    ASSERT_EQ(result.get(), seq.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorReturnStmt) {
    IRMutator mutator;
    auto val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
    std::vector<ExprPtr> values = {val};
    auto ret = std::make_shared<ReturnStmt>(values, Span::unknown());

    auto result = mutator.VisitStmt(ret);
    ASSERT_EQ(result.get(), ret.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorYieldStmt) {
    IRMutator mutator;
    auto val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
    std::vector<ExprPtr> values = {val};
    auto yieldStmt = std::make_shared<YieldStmt>(values, Span::unknown());

    auto result = mutator.VisitStmt(yieldStmt);
    ASSERT_EQ(result.get(), yieldStmt.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorForStmt) {
    IRMutator mutator;
    auto loopVar = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto bodyExpr = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
    auto body = std::make_shared<EvalStmt>(bodyExpr, Span::unknown());

    std::vector<IterArgPtr> iterArgs;
    std::vector<VarPtr> returnVars;
    auto forStmt = std::make_shared<ForStmt>(loopVar, start, stop, step, iterArgs, body, returnVars, Span::unknown());

    auto result = mutator.VisitStmt(forStmt);
    ASSERT_EQ(result.get(), forStmt.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorIfStmt) {
    IRMutator mutator;
    auto cond = std::make_shared<ConstBool>(true, Span::unknown());
    auto thenExpr = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto thenBody = std::make_shared<EvalStmt>(thenExpr, Span::unknown());

    std::vector<VarPtr> returnVars;
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::nullopt, returnVars, Span::unknown());

    auto result = mutator.VisitStmt(ifStmt);
    ASSERT_EQ(result.get(), ifStmt.get());
}

// ============================================================================
// Mutator Binary/Unary Expression Tests
// ============================================================================

TEST_F(IRMutatorTest, TestIdentityMutatorSub) {
    IRMutator mutator;
    auto left = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
    auto right = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
    auto sub = std::make_shared<Sub>(left, right, DataType::INT32, Span::unknown());

    auto result = mutator.VisitExpr(sub);
    ASSERT_EQ(result.get(), sub.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorMul) {
    IRMutator mutator;
    auto left = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto right = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
    auto mul = std::make_shared<Mul>(left, right, DataType::INT32, Span::unknown());

    auto result = mutator.VisitExpr(mul);
    ASSERT_EQ(result.get(), mul.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorNeg) {
    IRMutator mutator;
    auto operand = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
    auto neg = std::make_shared<Neg>(operand, DataType::INT32, Span::unknown());

    auto result = mutator.VisitExpr(neg);
    ASSERT_EQ(result.get(), neg.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorConstFloat) {
    IRMutator mutator;
    auto expr = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::unknown());
    auto result = mutator.VisitExpr(expr);
    ASSERT_EQ(result.get(), expr.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorConstBool) {
    IRMutator mutator;
    auto expr = std::make_shared<ConstBool>(true, Span::unknown());
    auto result = mutator.VisitExpr(expr);
    ASSERT_EQ(result.get(), expr.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorMakeTuple) {
    IRMutator mutator;
    auto elem1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto elem2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    std::vector<ExprPtr> elements = {elem1, elem2};
    auto tuple = std::make_shared<MakeTuple>(elements, Span::unknown());

    auto result = mutator.VisitExpr(tuple);
    ASSERT_EQ(result.get(), tuple.get());
}

TEST_F(IRMutatorTest, TestIdentityMutatorOpStmts) {
    IRMutator mutator;
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
    auto val = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto assign = std::make_shared<AssignStmt>(var, val, Span::unknown());

    std::vector<StmtPtr> stmts = {assign};
    auto opStmts = std::make_shared<OpStmts>(stmts, Span::unknown());

    auto result = mutator.VisitStmt(opStmts);
    ASSERT_EQ(result.get(), opStmts.get());
}

// ============================================================================
// Additional Binary/Unary/Node Mutator Tests
// ============================================================================

TEST_F(IRMutatorTest, TestIdentityFloorDiv) {
    IRMutator mutator;
    auto l = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
    auto e = std::make_shared<FloorDiv>(l, r, DataType::INT32, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(e).get(), e.get());
}

TEST_F(IRMutatorTest, TestIdentityFloorMod) {
    IRMutator mutator;
    auto l = std::make_shared<ConstInt>(10, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
    auto e = std::make_shared<FloorMod>(l, r, DataType::INT32, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(e).get(), e.get());
}

TEST_F(IRMutatorTest, TestIdentityFloatDiv) {
    IRMutator mutator;
    auto l = std::make_shared<ConstFloat>(1.0, DataType::FP32, Span::unknown());
    auto r = std::make_shared<ConstFloat>(2.0, DataType::FP32, Span::unknown());
    auto e = std::make_shared<FloatDiv>(l, r, DataType::FP32, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(e).get(), e.get());
}

TEST_F(IRMutatorTest, TestIdentityMinMax) {
    IRMutator mutator;
    auto l = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto mn = std::make_shared<Min>(l, r, DataType::INT32, Span::unknown());
    auto mx = std::make_shared<Max>(l, r, DataType::INT32, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(mn).get(), mn.get());
    ASSERT_EQ(mutator.VisitExpr(mx).get(), mx.get());
}

TEST_F(IRMutatorTest, TestIdentityPow) {
    IRMutator mutator;
    auto l = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::unknown());
    auto e = std::make_shared<Pow>(l, r, DataType::INT32, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(e).get(), e.get());
}

TEST_F(IRMutatorTest, TestIdentityComparisonOps) {
    IRMutator mutator;
    auto l = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(2, DataType::INT32, Span::unknown());
    auto eq = std::make_shared<Eq>(l, r, DataType::BOOL, Span::unknown());
    auto ne = std::make_shared<Ne>(l, r, DataType::BOOL, Span::unknown());
    auto lt = std::make_shared<Lt>(l, r, DataType::BOOL, Span::unknown());
    auto le = std::make_shared<Le>(l, r, DataType::BOOL, Span::unknown());
    auto gt = std::make_shared<Gt>(l, r, DataType::BOOL, Span::unknown());
    auto ge = std::make_shared<Ge>(l, r, DataType::BOOL, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(eq).get(), eq.get());
    ASSERT_EQ(mutator.VisitExpr(ne).get(), ne.get());
    ASSERT_EQ(mutator.VisitExpr(lt).get(), lt.get());
    ASSERT_EQ(mutator.VisitExpr(le).get(), le.get());
    ASSERT_EQ(mutator.VisitExpr(gt).get(), gt.get());
    ASSERT_EQ(mutator.VisitExpr(ge).get(), ge.get());
}

TEST_F(IRMutatorTest, TestIdentityLogicalOps) {
    IRMutator mutator;
    auto l = std::make_shared<ConstBool>(true, Span::unknown());
    auto r = std::make_shared<ConstBool>(false, Span::unknown());
    auto a = std::make_shared<And>(l, r, DataType::BOOL, Span::unknown());
    auto o = std::make_shared<Or>(l, r, DataType::BOOL, Span::unknown());
    auto x = std::make_shared<Xor>(l, r, DataType::BOOL, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(a).get(), a.get());
    ASSERT_EQ(mutator.VisitExpr(o).get(), o.get());
    ASSERT_EQ(mutator.VisitExpr(x).get(), x.get());
}

TEST_F(IRMutatorTest, TestIdentityBitwiseOps) {
    IRMutator mutator;
    auto l = std::make_shared<ConstInt>(0xFF, DataType::INT32, Span::unknown());
    auto r = std::make_shared<ConstInt>(0x0F, DataType::INT32, Span::unknown());
    auto ba = std::make_shared<BitAnd>(l, r, DataType::INT32, Span::unknown());
    auto bo = std::make_shared<BitOr>(l, r, DataType::INT32, Span::unknown());
    auto bx = std::make_shared<BitXor>(l, r, DataType::INT32, Span::unknown());
    auto sl = std::make_shared<BitShiftLeft>(l, r, DataType::INT32, Span::unknown());
    auto sr = std::make_shared<BitShiftRight>(l, r, DataType::INT32, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(ba).get(), ba.get());
    ASSERT_EQ(mutator.VisitExpr(bo).get(), bo.get());
    ASSERT_EQ(mutator.VisitExpr(bx).get(), bx.get());
    ASSERT_EQ(mutator.VisitExpr(sl).get(), sl.get());
    ASSERT_EQ(mutator.VisitExpr(sr).get(), sr.get());
}

TEST_F(IRMutatorTest, TestIdentityUnaryOps) {
    IRMutator mutator;
    auto v = std::make_shared<ConstInt>(5, DataType::INT32, Span::unknown());
    auto abs_e = std::make_shared<Abs>(v, DataType::INT32, Span::unknown());
    auto not_e =
        std::make_shared<Not>(std::make_shared<ConstBool>(true, Span::unknown()), DataType::BOOL, Span::unknown());
    auto bn = std::make_shared<BitNot>(v, DataType::INT32, Span::unknown());
    auto cast_e = std::make_shared<Cast>(v, DataType::FP32, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(abs_e).get(), abs_e.get());
    ASSERT_EQ(mutator.VisitExpr(not_e).get(), not_e.get());
    ASSERT_EQ(mutator.VisitExpr(bn).get(), bn.get());
    ASSERT_EQ(mutator.VisitExpr(cast_e).get(), cast_e.get());
}

TEST_F(IRMutatorTest, TestIdentityTupleGetItem) {
    IRMutator mutator;
    auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1}, Span::unknown());
    auto get = std::make_shared<TupleGetItemExpr>(tup, 0, Span::unknown());
    ASSERT_EQ(mutator.VisitExpr(get).get(), get.get());
}

TEST_F(IRMutatorTest, TestIdentityCall) {
    IRMutator mutator;
    Span sp = Span::unknown();
    auto &reg = OpRegistry::GetInstance();
    auto a = std::make_shared<Var>("a",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto b = std::make_shared<Var>("b",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto call = reg.Create("tensor.add", {a, b}, sp);
    ASSERT_EQ(mutator.VisitExpr(call).get(), call.get());
}

TEST_F(IRMutatorTest, TestIdentityMemRef) {
    IRMutator mutator;
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::unknown());
    auto mr = std::make_shared<MemRef>(MemorySpace::UB, addr, 1024, 0);
    ASSERT_EQ(mutator.VisitExpr(mr).get(), mr.get());
}

TEST_F(IRMutatorTest, TestIdentityIfStmtWithElse) {
    IRMutator mutator;
    Span sp = Span::unknown();
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto thenBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto elseBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseBody), std::vector<VarPtr>{}, sp);
    ASSERT_EQ(mutator.VisitStmt(ifStmt).get(), ifStmt.get());
}

// ============================================================================
// Custom Mutator for Copy-on-Write Tests
// ============================================================================

// A custom mutator that replaces ConstInt(42) with ConstInt(99)
// and Var("replace_me") with a new Var("replaced")
class CopyOnWriteTestMutator : public IRMutator {
    using IRMutator::VisitExpr_;

protected:
    ExprPtr VisitExpr_(const ConstIntPtr &op) override {
        if (op->value_ == 42) {
            auto scalarType = As<ScalarType>(op->GetType());
            return std::make_shared<ConstInt>(99, scalarType->dtype_, op->span_);
        }
        return op;
    }

    ExprPtr VisitExpr_(const VarPtr &op) override {
        if (op->name_ == "replace_me") {
            return std::make_shared<Var>("replaced", op->GetType(), op->span_);
        }
        return op;
    }
};

// ============================================================================
// Copy-on-Write Tests (children change, new nodes created)
// ============================================================================

TEST_F(IRMutatorTest, TestCOWEvalStmt) {
    CopyOnWriteTestMutator mutator;
    auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto stmt = std::make_shared<EvalStmt>(expr, Span::unknown());
    auto result = mutator.VisitStmt(stmt);
    ASSERT_NE(result.get(), stmt.get());
    auto eval = As<EvalStmt>(result);
    ASSERT_NE(eval, nullptr);
    auto ci = As<ConstInt>(eval->expr_);
    ASSERT_NE(ci, nullptr);
    ASSERT_EQ(ci->value_, 99);
}

TEST_F(IRMutatorTest, TestCOWAssignStmt) {
    CopyOnWriteTestMutator mutator;
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::unknown());
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto stmt = std::make_shared<AssignStmt>(var, val, Span::unknown());
    auto result = mutator.VisitStmt(stmt);
    ASSERT_NE(result.get(), stmt.get());
    auto assign = As<AssignStmt>(result);
    ASSERT_NE(assign, nullptr);
    auto ci = As<ConstInt>(assign->value_);
    ASSERT_EQ(ci->value_, 99);
}

TEST_F(IRMutatorTest, TestCOWAddBinary) {
    CopyOnWriteTestMutator mutator;
    auto left = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto right = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto add = std::make_shared<Add>(left, right, DataType::INT32, Span::unknown());
    auto result = mutator.VisitExpr(add);
    ASSERT_NE(result.get(), add.get());
    auto new_add = As<Add>(result);
    ASSERT_NE(new_add, nullptr);
    auto newLeft = As<ConstInt>(new_add->left_);
    ASSERT_EQ(newLeft->value_, 99);
}

TEST_F(IRMutatorTest, TestCOWNegUnary) {
    CopyOnWriteTestMutator mutator;
    auto operand = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto neg = std::make_shared<Neg>(operand, DataType::INT32, Span::unknown());
    auto result = mutator.VisitExpr(neg);
    ASSERT_NE(result.get(), neg.get());
    auto new_neg = As<Neg>(result);
    ASSERT_NE(new_neg, nullptr);
    auto new_op = As<ConstInt>(new_neg->operand_);
    ASSERT_EQ(new_op->value_, 99);
}

TEST_F(IRMutatorTest, TestCOWMakeTuple) {
    CopyOnWriteTestMutator mutator;
    auto elem1 = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto elem2 = std::make_shared<ConstInt>(1, DataType::INT32, Span::unknown());
    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{elem1, elem2}, Span::unknown());
    auto result = mutator.VisitExpr(tuple);
    ASSERT_NE(result.get(), tuple.get());
}

TEST_F(IRMutatorTest, TestCOWTupleGetItemExpr) {
    CopyOnWriteTestMutator mutator;
    auto elem = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto tuple = std::make_shared<MakeTuple>(std::vector<ExprPtr>{elem}, Span::unknown());
    auto get = std::make_shared<TupleGetItemExpr>(tuple, 0, Span::unknown());
    auto result = mutator.VisitExpr(get);
    ASSERT_NE(result.get(), get.get());
}

TEST_F(IRMutatorTest, TestCOWIterArg) {
    CopyOnWriteTestMutator mutator;
    auto init = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto iterArg =
        std::make_shared<IterArg>("acc", std::make_shared<ScalarType>(DataType::INT32), init, Span::unknown());
    auto result = mutator.VisitExpr(iterArg);
    ASSERT_NE(result.get(), iterArg.get());
}

TEST_F(IRMutatorTest, TestCOWCall) {
    CopyOnWriteTestMutator mutator;
    Span sp = Span::unknown();
    auto &reg = OpRegistry::GetInstance();
    auto a = std::make_shared<Var>("replace_me",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto b = std::make_shared<Var>("b",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto call = reg.Create("tensor.add", {a, b}, sp);
    auto result = mutator.VisitExpr(call);
    ASSERT_NE(result.get(), call.get());
}

TEST_F(IRMutatorTest, TestCOWSeqStmts) {
    CopyOnWriteTestMutator mutator;
    auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto eval = std::make_shared<EvalStmt>(expr, Span::unknown());
    auto seq = std::make_shared<SeqStmts>(std::vector<StmtPtr>{eval}, Span::unknown());
    auto result = mutator.VisitStmt(seq);
    ASSERT_NE(result.get(), seq.get());
}

TEST_F(IRMutatorTest, TestCOWReturnStmt) {
    CopyOnWriteTestMutator mutator;
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{val}, Span::unknown());
    auto result = mutator.VisitStmt(ret);
    ASSERT_NE(result.get(), ret.get());
}

TEST_F(IRMutatorTest, TestCOWYieldStmt) {
    CopyOnWriteTestMutator mutator;
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto yieldStmt = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Span::unknown());
    auto result = mutator.VisitStmt(yieldStmt);
    ASSERT_NE(result.get(), yieldStmt.get());
}

TEST_F(IRMutatorTest, TestCOWIfStmtThenChanged) {
    CopyOnWriteTestMutator mutator;
    auto cond = std::make_shared<ConstBool>(true, Span::unknown());
    auto thenExpr = std::make_shared<ConstInt>(42, DataType::INT32, Span::unknown());
    auto thenBody = std::make_shared<EvalStmt>(thenExpr, Span::unknown());
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::nullopt, std::vector<VarPtr>{}, Span::unknown());
    auto result = mutator.VisitStmt(ifStmt);
    ASSERT_NE(result.get(), ifStmt.get());
}

TEST_F(IRMutatorTest, TestCOWIfStmtElseChanged) {
    CopyOnWriteTestMutator mutator;
    Span sp = Span::unknown();
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto thenBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto elseExpr = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto elseBody = std::make_shared<EvalStmt>(elseExpr, sp);
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseBody), std::vector<VarPtr>{}, sp);
    auto result = mutator.VisitStmt(ifStmt);
    ASSERT_NE(result.get(), ifStmt.get());
}

TEST_F(IRMutatorTest, TestCOWIfStmtReturnVarsChanged) {
    CopyOnWriteTestMutator mutator;
    Span sp = Span::unknown();
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto thenBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto returnVar = std::make_shared<Var>("replace_me", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::nullopt, std::vector<VarPtr>{returnVar}, sp);
    auto result = mutator.VisitStmt(ifStmt);
    ASSERT_NE(result.get(), ifStmt.get());
}

TEST_F(IRMutatorTest, TestCOWIfStmtWithElseAndReturnVars) {
    CopyOnWriteTestMutator mutator;
    Span sp = Span::unknown();
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto thenBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto elseBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(42, DataType::INT32, sp), sp);
    auto returnVar = std::make_shared<Var>("replace_me", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto ifStmt =
        std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseBody), std::vector<VarPtr>{returnVar}, sp);
    auto result = mutator.VisitStmt(ifStmt);
    ASSERT_NE(result.get(), ifStmt.get());
    // Verify the new IfStmt has else body
    auto new_if = As<IfStmt>(result);
    ASSERT_NE(new_if, nullptr);
    ASSERT_TRUE(new_if->elseBody_.has_value());
}

TEST_F(IRMutatorTest, TestCOWForStmt) {
    CopyOnWriteTestMutator mutator;
    Span sp = Span::unknown();
    auto loopVar = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto start = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, sp);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto body = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto forStmt = std::make_shared<ForStmt>(
        loopVar, start, stop, step, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{}, sp);
    auto result = mutator.VisitStmt(forStmt);
    ASSERT_NE(result.get(), forStmt.get());
}

TEST_F(IRMutatorTest, TestCOWForStmtIterArgsChanged) {
    CopyOnWriteTestMutator mutator;
    Span sp = Span::unknown();
    auto loopVar = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, sp);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto init = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto iterArg = std::make_shared<IterArg>("acc", std::make_shared<ScalarType>(DataType::INT32), init, sp);
    auto body = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto forStmt = std::make_shared<ForStmt>(
        loopVar, start, stop, step, std::vector<IterArgPtr>{iterArg}, body, std::vector<VarPtr>{}, sp);
    auto result = mutator.VisitStmt(forStmt);
    ASSERT_NE(result.get(), forStmt.get());
}

TEST_F(IRMutatorTest, TestCOWForStmtReturnVarsChanged) {
    CopyOnWriteTestMutator mutator;
    Span sp = Span::unknown();
    auto loopVar = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, sp);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto body = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(0, DataType::INT32, sp), sp);
    auto returnVar = std::make_shared<Var>("replace_me", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto forStmt = std::make_shared<ForStmt>(
        loopVar, start, stop, step, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{returnVar}, sp);
    auto result = mutator.VisitStmt(forStmt);
    ASSERT_NE(result.get(), forStmt.get());
}

TEST_F(IRMutatorTest, TestCOWOpStmts) {
    CopyOnWriteTestMutator mutator;
    Span sp = Span::unknown();
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto val = std::make_shared<ConstInt>(42, DataType::INT32, sp);
    auto assign = std::make_shared<AssignStmt>(var, val, sp);
    auto opStmts = std::make_shared<OpStmts>(std::vector<StmtPtr>{assign}, sp);
    auto result = mutator.VisitStmt(opStmts);
    ASSERT_NE(result.get(), opStmts.get());
}

} // namespace ir
} // namespace pypto
