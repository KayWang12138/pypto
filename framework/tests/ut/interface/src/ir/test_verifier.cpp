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
  auto body_val = std::make_shared<ConstInt>(0, DataType::INT32, Span::unknown());
  auto body = std::make_shared<EvalStmt>(body_val, Span::unknown());
  std::vector<VarPtr> params;
  std::vector<TypePtr> return_types;
  auto func = std::make_shared<Function>("main", params, return_types, body, Span::unknown());
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
  std::vector<Diagnostic> empty_diags;
  std::string report = verifier.GenerateReport(empty_diags);
  ASSERT_NE(report.find("PASSED"), std::string::npos);
}

TEST_F(IRVerifierTest, TestGenerateReportWithErrors) {
  auto verifier = IRVerifier::CreateDefault();
  std::vector<Diagnostic> diags;
  diags.emplace_back(DiagnosticSeverity::Error, "TestRule", 1, "error msg", Span::unknown());
  diags.emplace_back(DiagnosticSeverity::Warning, "TestRule", 2, "warn msg", Span::unknown());

  std::string report = verifier.GenerateReport(diags);
  ASSERT_NE(report.find("FAILED"), std::string::npos);
  ASSERT_NE(report.find("1 error"), std::string::npos);
  ASSERT_NE(report.find("1 warning"), std::string::npos);
}

TEST_F(IRVerifierTest, TestGenerateReportWarningsOnly) {
  auto verifier = IRVerifier::CreateDefault();
  std::vector<Diagnostic> diags;
  diags.emplace_back(DiagnosticSeverity::Warning, "TestRule", 1, "warn msg", Span::unknown());

  std::string report = verifier.GenerateReport(diags);
  ASSERT_NE(report.find("PASSED"), std::string::npos);
  ASSERT_NE(report.find("warning"), std::string::npos);
}

}  // namespace ir
}  // namespace pypto
