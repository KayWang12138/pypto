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
 * \file test_symbolic_scalar_evaluate.cpp
 * \brief Unit tests for symbolic scalar evaluation
 */

#include "gtest/gtest.h"

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/configs/config_manager.h"
#include "interface/tensor/symbolic_scalar.h"

using namespace npu::tile_fwk;

class SymbolicScalarEvaluateTest : public testing::Test {
public:
    void SetUp() override {
        Program::GetInstance().Reset();
        config::Reset();
    }

    void TearDown() override { Program::GetInstance().Reset(); }
};

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarBasic) {
    SymbolicScalar a("a", 10);
    SymbolicScalar b("b", 20);

    EXPECT_TRUE(a.ConcreteValid());
    EXPECT_EQ(a.Concrete(), 10);
    EXPECT_TRUE(b.ConcreteValid());
    EXPECT_EQ(b.Concrete(), 20);
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarExpression) {
    SymbolicScalar a("a", 10);
    SymbolicScalar b("b", 20);

    SymbolicScalar c = a + b;
    EXPECT_TRUE(c.ConcreteValid());
    EXPECT_EQ(c.Concrete(), 30);

    SymbolicScalar d = a * b;
    EXPECT_TRUE(d.ConcreteValid());
    EXPECT_EQ(d.Concrete(), 200);
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarComparison) {
    SymbolicScalar a("a", 10);
    SymbolicScalar b("b", 20);

    SymbolicScalar c = a < b;
    EXPECT_TRUE(c.ConcreteValid());
    EXPECT_EQ(c.Concrete(), 1);

    SymbolicScalar d = a > b;
    EXPECT_TRUE(d.ConcreteValid());
    EXPECT_EQ(d.Concrete(), 0);
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarClosure) {
    SymbolicScalar a("a");
    SymbolicScalar b("b");
    SymbolicScalar c = a + b;

    EXPECT_FALSE(c.ConcreteValid());

    SymbolicClosure closure;
    closure.Insert("a", 100);
    closure.Insert("b", 50);

    EXPECT_EQ(150, closure.Evaluate(c));
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarUnaryOps) {
    SymbolicScalar a("a", 10);

    SymbolicScalar b = -a;
    EXPECT_TRUE(b.ConcreteValid());
    EXPECT_EQ(b.Concrete(), -10);

    SymbolicScalar c = +a;
    EXPECT_TRUE(c.ConcreteValid());
    EXPECT_EQ(c.Concrete(), 10);

    SymbolicScalar d = ~a;
    EXPECT_TRUE(d.ConcreteValid());
    EXPECT_EQ(d.Concrete(), ~10);
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarBinaryOps) {
    SymbolicScalar a("a", 10);
    SymbolicScalar b("b", 3);

    SymbolicScalar c = a + b;
    EXPECT_EQ(c.Concrete(), 13);

    SymbolicScalar d = a - b;
    EXPECT_EQ(d.Concrete(), 7);

    SymbolicScalar e = a * b;
    EXPECT_EQ(e.Concrete(), 30);

    SymbolicScalar f = a / b;
    EXPECT_EQ(f.Concrete(), 3);

    SymbolicScalar g = a % b;
    EXPECT_EQ(g.Concrete(), 1);
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarLogicalOps) {
    SymbolicScalar a("a", 10);
    SymbolicScalar b("b", 20);

    SymbolicScalar c = a && b;
    EXPECT_TRUE(c.ConcreteValid());
    EXPECT_EQ(c.Concrete(), 1);

    SymbolicScalar d = a || b;
    EXPECT_TRUE(d.ConcreteValid());
    EXPECT_EQ(d.Concrete(), 1);

    SymbolicScalar e = !a;
    EXPECT_TRUE(e.ConcreteValid());
    EXPECT_EQ(e.Concrete(), 0);
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarBitwiseOps) {
    SymbolicScalar a("a", 10);
    SymbolicScalar b("b", 6);

    SymbolicScalar c = a & b;
    EXPECT_TRUE(c.ConcreteValid());
    EXPECT_EQ(c.Concrete(), 10 & 6);

    SymbolicScalar d = a | b;
    EXPECT_TRUE(d.ConcreteValid());
    EXPECT_EQ(d.Concrete(), 10 | 6);

    SymbolicScalar e = a ^ b;
    EXPECT_TRUE(e.ConcreteValid());
    EXPECT_EQ(e.Concrete(), 10 ^ 6);
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarComplexExpression) {
    SymbolicScalar a("a", 10);
    SymbolicScalar b("b", 5);
    SymbolicScalar c("c", 2);

    SymbolicScalar result = (a + b) * c;
    EXPECT_TRUE(result.ConcreteValid());
    EXPECT_EQ(result.Concrete(), 30);

    SymbolicScalar result2 = a * b + c;
    EXPECT_TRUE(result2.ConcreteValid());
    EXPECT_EQ(result2.Concrete(), 52);
}

TEST_F(SymbolicScalarEvaluateTest, TestSymbolicScalarMaxMin) {
    SymbolicScalar a("a", 10);
    SymbolicScalar b("b", 20);

    SymbolicScalar c = a.Max(b);
    EXPECT_TRUE(c.ConcreteValid());
    EXPECT_EQ(c.Concrete(), 20);

    SymbolicScalar d = a.Min(b);
    EXPECT_TRUE(d.ConcreteValid());
    EXPECT_EQ(d.Concrete(), 10);
}