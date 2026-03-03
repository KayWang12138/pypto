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
 * \file test_visitor.cpp
 * \brief Unit tests for IR visitor traversal
 */

#include "gtest/gtest.h"

#include <memory>
#include <optional>
#include <vector>

#include "core/dtype.h"
#include "ir/expr.h"
#include "ir/memref.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/base/visitor.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// A counting visitor that tracks how many nodes are visited
class CountingVisitor : public IRVisitor {
public:
    using IRVisitor::VisitExpr_;
    using IRVisitor::VisitStmt_;

    int exprCount = 0;
    int stmtCount = 0;

    void VisitExpr_(const ConstIntPtr &op) override {
        exprCount++;
        IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const VarPtr &op) override {
        exprCount++;
        IRVisitor::VisitExpr_(op);
    }

    void VisitExpr_(const AddPtr &op) override {
        exprCount++;
        IRVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const AssignStmtPtr &op) override {
        stmtCount++;
        IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const EvalStmtPtr &op) override {
        stmtCount++;
        IRVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const SeqStmtsPtr &op) override {
        stmtCount++;
        IRVisitor::VisitStmt_(op);
    }
};

class IRVisitorTest : public testing::Test {};

// ============================================================================
// Visitor Expression Tests
// ============================================================================

TEST_F(IRVisitorTest, TestVisitConstInt) {
    CountingVisitor visitor;
    auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(expr);
    ASSERT_EQ(visitor.exprCount, 1);
}

TEST_F(IRVisitorTest, TestVisitBinaryExpr) {
    CountingVisitor visitor;
    auto left = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto right = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    auto add = std::make_shared<Add>(left, right, DataType::INT32, Span::Unknown());

    visitor.VisitExpr(add);
    // Should visit Add + left ConstInt + right ConstInt
    ASSERT_EQ(visitor.exprCount, 3);
}

TEST_F(IRVisitorTest, TestVisitVar) {
    CountingVisitor visitor;
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::Unknown());
    visitor.VisitExpr(var);
    ASSERT_EQ(visitor.exprCount, 1);
}

// ============================================================================
// Visitor Statement Tests
// ============================================================================

TEST_F(IRVisitorTest, TestVisitEvalStmt) {
    CountingVisitor visitor;
    auto expr = std::make_shared<ConstInt>(42, DataType::INT32, Span::Unknown());
    auto stmt = std::make_shared<EvalStmt>(expr, Span::Unknown());
    visitor.VisitStmt(stmt);
    ASSERT_EQ(visitor.stmtCount, 1);
    ASSERT_EQ(visitor.exprCount, 1);
}

TEST_F(IRVisitorTest, TestVisitAssignStmt) {
    CountingVisitor visitor;
    auto var = std::make_shared<Var>("x", std::make_shared<ScalarType>(DataType::INT32), Span::Unknown());
    auto val = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    auto stmt = std::make_shared<AssignStmt>(var, val, Span::Unknown());
    visitor.VisitStmt(stmt);
    ASSERT_EQ(visitor.stmtCount, 1);
    ASSERT_EQ(visitor.exprCount, 2); // var + val
}

TEST_F(IRVisitorTest, TestVisitSeqStmts) {
    CountingVisitor visitor;
    auto expr1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto eval1 = std::make_shared<EvalStmt>(expr1, Span::Unknown());
    auto expr2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    auto eval2 = std::make_shared<EvalStmt>(expr2, Span::Unknown());

    std::vector<StmtPtr> stmts = {eval1, eval2};
    auto seq = std::make_shared<SeqStmts>(stmts, Span::Unknown());
    visitor.VisitStmt(seq);

    ASSERT_EQ(visitor.stmtCount, 3); // SeqStmts + 2 EvalStmts
    ASSERT_EQ(visitor.exprCount, 2); // 2 ConstInts
}

// ============================================================================
// Additional Expression Visitor Tests
// ============================================================================

TEST_F(IRVisitorTest, TestVisitConstFloat) {
    IRVisitor visitor;
    auto expr = std::make_shared<ConstFloat>(3.14, DataType::FP32, Span::Unknown());
    visitor.VisitExpr(expr);
}

TEST_F(IRVisitorTest, TestVisitConstBool) {
    IRVisitor visitor;
    auto expr = std::make_shared<ConstBool>(true, Span::Unknown());
    visitor.VisitExpr(expr);
}

TEST_F(IRVisitorTest, TestVisitSub) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto r = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    auto expr = std::make_shared<Sub>(l, r, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(expr);
}

TEST_F(IRVisitorTest, TestVisitMul) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstInt>(3, DataType::INT32, Span::Unknown());
    auto r = std::make_shared<ConstInt>(4, DataType::INT32, Span::Unknown());
    auto expr = std::make_shared<Mul>(l, r, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(expr);
}

TEST_F(IRVisitorTest, TestVisitFloorDiv) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::Unknown());
    auto expr = std::make_shared<FloorDiv>(l, r, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(expr);
}

TEST_F(IRVisitorTest, TestVisitFloorMod) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstInt>(10, DataType::INT32, Span::Unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::Unknown());
    auto expr = std::make_shared<FloorMod>(l, r, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(expr);
}

TEST_F(IRVisitorTest, TestVisitMinMax) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto r = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    auto mn = std::make_shared<Min>(l, r, DataType::INT32, Span::Unknown());
    auto mx = std::make_shared<Max>(l, r, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(mn);
    visitor.VisitExpr(mx);
}

TEST_F(IRVisitorTest, TestVisitComparisonOps) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto r = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(std::make_shared<Eq>(l, r, DataType::BOOL, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Ne>(l, r, DataType::BOOL, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Lt>(l, r, DataType::BOOL, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Le>(l, r, DataType::BOOL, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Gt>(l, r, DataType::BOOL, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Ge>(l, r, DataType::BOOL, Span::Unknown()));
}

TEST_F(IRVisitorTest, TestVisitLogicalOps) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstBool>(true, Span::Unknown());
    auto r = std::make_shared<ConstBool>(false, Span::Unknown());
    visitor.VisitExpr(std::make_shared<And>(l, r, DataType::BOOL, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Or>(l, r, DataType::BOOL, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Xor>(l, r, DataType::BOOL, Span::Unknown()));
}

TEST_F(IRVisitorTest, TestVisitBitwiseOps) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstInt>(0xFF, DataType::INT32, Span::Unknown());
    auto r = std::make_shared<ConstInt>(0x0F, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(std::make_shared<BitAnd>(l, r, DataType::INT32, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<BitOr>(l, r, DataType::INT32, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<BitXor>(l, r, DataType::INT32, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<BitShiftLeft>(l, r, DataType::INT32, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<BitShiftRight>(l, r, DataType::INT32, Span::Unknown()));
}

TEST_F(IRVisitorTest, TestVisitUnaryOps) {
    IRVisitor visitor;
    auto val = std::make_shared<ConstInt>(5, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(std::make_shared<Neg>(val, DataType::INT32, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Abs>(val, DataType::INT32, Span::Unknown()));
    auto bval = std::make_shared<ConstBool>(true, Span::Unknown());
    visitor.VisitExpr(std::make_shared<Not>(bval, DataType::BOOL, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<BitNot>(val, DataType::INT32, Span::Unknown()));
    visitor.VisitExpr(std::make_shared<Cast>(val, DataType::FP32, Span::Unknown()));
}

TEST_F(IRVisitorTest, TestVisitMakeTuple) {
    IRVisitor visitor;
    auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto e2 = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1, e2}, Span::Unknown());
    visitor.VisitExpr(tup);
}

TEST_F(IRVisitorTest, TestVisitTupleGetItem) {
    IRVisitor visitor;
    auto e1 = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto tup = std::make_shared<MakeTuple>(std::vector<ExprPtr>{e1}, Span::Unknown());
    auto get = std::make_shared<TupleGetItemExpr>(tup, 0, Span::Unknown());
    visitor.VisitExpr(get);
}

TEST_F(IRVisitorTest, TestVisitCall) {
    auto &reg = OpRegistry::GetInstance();
    Span sp = Span::Unknown();
    auto a = std::make_shared<Var>("a",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto b = std::make_shared<Var>("b",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{std::make_shared<ConstInt>(4, DataType::INT64, sp)}, DataType::FP32),
        sp);
    auto call = reg.Create("tensor.add", {a, b}, sp);
    IRVisitor visitor;
    visitor.VisitExpr(call);
}

TEST_F(IRVisitorTest, TestVisitMemRef) {
    IRVisitor visitor;
    auto addr = std::make_shared<ConstInt>(0, DataType::INT64, Span::Unknown());
    auto mr = std::make_shared<MemRef>(MemorySpace::UB, addr, 1024, 0);
    visitor.VisitExpr(mr);
}

// ============================================================================
// Additional Statement Visitor Tests
// ============================================================================

TEST_F(IRVisitorTest, TestVisitReturnStmt) {
    IRVisitor visitor;
    auto val = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto ret = std::make_shared<ReturnStmt>(std::vector<ExprPtr>{val}, Span::Unknown());
    visitor.VisitStmt(ret);
}

TEST_F(IRVisitorTest, TestVisitYieldStmt) {
    IRVisitor visitor;
    auto val = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{val}, Span::Unknown());
    visitor.VisitStmt(yield);
}

TEST_F(IRVisitorTest, TestVisitOpStmts) {
    IRVisitor visitor;
    auto expr = std::make_shared<ConstInt>(1, DataType::INT32, Span::Unknown());
    auto eval = std::make_shared<EvalStmt>(expr, Span::Unknown());
    auto ops = std::make_shared<OpStmts>(std::vector<StmtPtr>{eval}, Span::Unknown());
    visitor.VisitStmt(ops);
}

TEST_F(IRVisitorTest, TestVisitForStmt) {
    IRVisitor visitor;
    Span sp = Span::Unknown();
    auto loopVar = std::make_shared<Var>("i", std::make_shared<ScalarType>(DataType::INT32), sp);
    auto start = std::make_shared<ConstInt>(0, DataType::INT32, sp);
    auto stop = std::make_shared<ConstInt>(10, DataType::INT32, sp);
    auto step = std::make_shared<ConstInt>(1, DataType::INT32, sp);
    auto body = std::make_shared<EvalStmt>(start, sp);
    auto forStmt = std::make_shared<ForStmt>(
        loopVar, start, stop, step, std::vector<IterArgPtr>{}, body, std::vector<VarPtr>{}, sp);
    visitor.VisitStmt(forStmt);
}

TEST_F(IRVisitorTest, TestVisitIfStmt) {
    IRVisitor visitor;
    Span sp = Span::Unknown();
    auto cond = std::make_shared<ConstBool>(true, sp);
    auto thenBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(1, DataType::INT32, sp), sp);
    auto elseBody = std::make_shared<EvalStmt>(std::make_shared<ConstInt>(2, DataType::INT32, sp), sp);
    auto ifStmt = std::make_shared<IfStmt>(cond, thenBody, std::optional<StmtPtr>(elseBody), std::vector<VarPtr>{}, sp);
    visitor.VisitStmt(ifStmt);
}

TEST_F(IRVisitorTest, TestVisitPow) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstInt>(2, DataType::INT32, Span::Unknown());
    auto r = std::make_shared<ConstInt>(3, DataType::INT32, Span::Unknown());
    visitor.VisitExpr(std::make_shared<Pow>(l, r, DataType::INT32, Span::Unknown()));
}

TEST_F(IRVisitorTest, TestVisitFloatDiv) {
    IRVisitor visitor;
    auto l = std::make_shared<ConstFloat>(1.0, DataType::FP32, Span::Unknown());
    auto r = std::make_shared<ConstFloat>(2.0, DataType::FP32, Span::Unknown());
    visitor.VisitExpr(std::make_shared<FloatDiv>(l, r, DataType::FP32, Span::Unknown()));
}

} // namespace ir
} // namespace pypto
