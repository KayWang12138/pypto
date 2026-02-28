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
 * \file test_verifier.cpp
 * \brief Unit tests for IR verifier framework
 */

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/program.h"
#include "ir/scalar_expr.h"
#include "ir/stmt.h"
#include "ir/transform/verifier.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// Helper to create a simple program
static ProgramPtr MakeSimpleProgram() {
  auto bodyVal = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(bodyVal, Span::unknown());
  std::vector<VarPtr> params;
  std::vector<TypePtr> returnTypes;
  auto func = std::make_shared<Function>("main", params, returnTypes, body, Span::unknown());
  std::vector<FunctionPtr> funcs = {func};
  return std::make_shared<Program>(funcs, "test", Span::unknown());
}

class IRVerifierTest : public testing::Test {};

// ============================================================================
// IRVerifier Basic Tests
// ============================================================================

TEST_F(IRVerifierTest, TestCreateDefault) {
  auto verifier = IRVerifier::CreateDefault();
  // Default verifier should have no rules
  auto program = MakeSimpleProgram();
  auto diags = verifier.Verify(program);
  ASSERT_TRUE(diags.empty());
}

TEST_F(IRVerifierTest, TestVerifyNullProgram) {
  auto verifier = IRVerifier::CreateDefault();
  auto diags = verifier.Verify(nullptr);
  ASSERT_TRUE(diags.empty());
}

TEST_F(IRVerifierTest, TestVerifyOrThrowNoErrors) {
  auto verifier = IRVerifier::CreateDefault();
  auto program = MakeSimpleProgram();
  ASSERT_NO_THROW(verifier.VerifyOrThrow(program));
}

// ============================================================================
// Rule Management Tests
// ============================================================================

TEST_F(IRVerifierTest, TestEnableDisableRule) {
  auto verifier = IRVerifier::CreateDefault();
  ASSERT_TRUE(verifier.IsRuleEnabled("test_rule"));

  verifier.DisableRule("test_rule");
  ASSERT_FALSE(verifier.IsRuleEnabled("test_rule"));

  verifier.EnableRule("test_rule");
  ASSERT_TRUE(verifier.IsRuleEnabled("test_rule"));
}

TEST_F(IRVerifierTest, TestAddNullRule) {
  auto verifier = IRVerifier::CreateDefault();
  verifier.AddRule(nullptr);
  // Should not crash
  auto program = MakeSimpleProgram();
  auto diags = verifier.Verify(program);
  ASSERT_TRUE(diags.empty());
}

// ============================================================================
// GenerateReport Tests
// ============================================================================

TEST_F(IRVerifierTest, TestGenerateReportEmpty) {
  auto verifier = IRVerifier::CreateDefault();
  std::vector<Diagnostic> emptyDiags;
  std::string report = verifier.GenerateReport(emptyDiags);
  ASSERT_NE(report.find("PASSED"), std::string::npos);
}

TEST_F(IRVerifierTest, TestGenerateReportWithErrors) {
  auto verifier = IRVerifier::CreateDefault();
  std::vector<Diagnostic> diags;
  diags.emplace_back(DiagnosticSeverity::ERROR, "TestRule", 1, "error msg", Span::unknown());
  diags.emplace_back(DiagnosticSeverity::WARNING, "TestRule", 2, "warn msg", Span::unknown());

  std::string report = verifier.GenerateReport(diags);
  ASSERT_NE(report.find("FAILED"), std::string::npos);
  ASSERT_NE(report.find("1 error"), std::string::npos);
  ASSERT_NE(report.find("1 warning"), std::string::npos);
}

TEST_F(IRVerifierTest, TestGenerateReportWarningsOnly) {
  auto verifier = IRVerifier::CreateDefault();
  std::vector<Diagnostic> diags;
  diags.emplace_back(DiagnosticSeverity::WARNING, "TestRule", 1, "warn msg", Span::unknown());

  std::string report = verifier.GenerateReport(diags);
  ASSERT_NE(report.find("PASSED"), std::string::npos);
  ASSERT_NE(report.find("warning"), std::string::npos);
}

// ============================================================================
// Custom Rule Tests
// ============================================================================

namespace {
class AlwaysErrorRule : public VerifyRule {
public:
  std::string GetName() const override { return "AlwaysError"; }
  void Verify(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) override {
    diagnostics.emplace_back(DiagnosticSeverity::ERROR, "AlwaysError", 100,
                             "function '" + func->name_ + "' always fails", func->span_);
  }
};

class AlwaysWarnRule : public VerifyRule {
public:
  std::string GetName() const override { return "AlwaysWarn"; }
  void Verify(const FunctionPtr& func, std::vector<Diagnostic>& diagnostics) override {
    diagnostics.emplace_back(DiagnosticSeverity::WARNING, "AlwaysWarn", 200,
                             "function '" + func->name_ + "' has a warning", func->span_);
  }
};
}  // namespace

TEST_F(IRVerifierTest, TestAddRuleAndVerify) {
  IRVerifier verifier;
  verifier.AddRule(std::make_shared<AlwaysErrorRule>());

  auto program = MakeSimpleProgram();
  auto diags = verifier.Verify(program);
  ASSERT_EQ(diags.size(), 1);
  ASSERT_EQ(diags[0].severity, DiagnosticSeverity::ERROR);
  ASSERT_NE(diags[0].message.find("main"), std::string::npos);
}

TEST_F(IRVerifierTest, TestVerifyOrThrowWithErrors) {
  IRVerifier verifier;
  verifier.AddRule(std::make_shared<AlwaysErrorRule>());

  auto program = MakeSimpleProgram();
  ASSERT_THROW(verifier.VerifyOrThrow(program), VerificationError);
}

TEST_F(IRVerifierTest, TestVerifyOrThrowWarningsOnly) {
  IRVerifier verifier;
  verifier.AddRule(std::make_shared<AlwaysWarnRule>());

  auto program = MakeSimpleProgram();
  ASSERT_NO_THROW(verifier.VerifyOrThrow(program));
}

TEST_F(IRVerifierTest, TestDisabledRuleSkipped) {
  IRVerifier verifier;
  verifier.AddRule(std::make_shared<AlwaysErrorRule>());
  verifier.DisableRule("AlwaysError");

  auto program = MakeSimpleProgram();
  auto diags = verifier.Verify(program);
  ASSERT_TRUE(diags.empty());
}

TEST_F(IRVerifierTest, TestReEnableRule) {
  IRVerifier verifier;
  verifier.AddRule(std::make_shared<AlwaysErrorRule>());
  verifier.DisableRule("AlwaysError");
  verifier.EnableRule("AlwaysError");

  auto program = MakeSimpleProgram();
  auto diags = verifier.Verify(program);
  ASSERT_EQ(diags.size(), 1);
}

TEST_F(IRVerifierTest, TestAddDuplicateRule) {
  IRVerifier verifier;
  verifier.AddRule(std::make_shared<AlwaysErrorRule>());
  verifier.AddRule(std::make_shared<AlwaysErrorRule>());  // Same name, should not add

  auto program = MakeSimpleProgram();
  auto diags = verifier.Verify(program);
  ASSERT_EQ(diags.size(), 1);  // Only one rule should run
}

TEST_F(IRVerifierTest, TestMultipleRules) {
  IRVerifier verifier;
  verifier.AddRule(std::make_shared<AlwaysErrorRule>());
  verifier.AddRule(std::make_shared<AlwaysWarnRule>());

  auto program = MakeSimpleProgram();
  auto diags = verifier.Verify(program);
  ASSERT_EQ(diags.size(), 2);
}

TEST_F(IRVerifierTest, TestGenerateReportMultipleErrors) {
  std::vector<Diagnostic> diags;
  diags.emplace_back(DiagnosticSeverity::ERROR, "Rule1", 1, "error 1", Span("test.py", 1, 1, 1, 10));
  diags.emplace_back(DiagnosticSeverity::ERROR, "Rule2", 2, "error 2", Span("test.py", 5, 1, 5, 10));
  diags.emplace_back(DiagnosticSeverity::WARNING, "Rule3", 3, "warn 1", Span("test.py", 10, 1, 10, 10));

  std::string report = IRVerifier::GenerateReport(diags);
  ASSERT_NE(report.find("FAILED"), std::string::npos);
  ASSERT_NE(report.find("2 error"), std::string::npos);
  ASSERT_NE(report.find("1 warning"), std::string::npos);
  ASSERT_NE(report.find("error 1"), std::string::npos);
  ASSERT_NE(report.find("error 2"), std::string::npos);
  ASSERT_NE(report.find("warn 1"), std::string::npos);
}

}  // namespace ir
}  // namespace pypto
