/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "../bindings.h"
#include "core/error.h"
#include "ir/transform/passes.h"
#include "ir/transform/verification_error.h"
#include "ir/transform/verifier.h"

namespace py = pybind11;

namespace pypto {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindPass(py::module_ &m) {
  py::module_ passes = m.def_submodule("passes", "IR transformation passes");

  py::class_<Pass>(passes, "Pass")
      .def("__call__", &Pass::operator(), py::arg("program"));

  passes.def("init_mem_ref", &pass::InitMemRef);
  passes.def("basic_memory_reuse", &pass::BasicMemoryReuse);
  passes.def("insert_sync", &pass::InsertSync);
  passes.def("add_alloc", &pass::AddAlloc);

  py::enum_<ssa::ErrorType>(passes, "SSAErrorType")
      .value("MULTIPLE_ASSIGNMENT", ssa::ErrorType::MULTIPLE_ASSIGNMENT)
      .value("NAME_SHADOWING", ssa::ErrorType::NAME_SHADOWING)
      .value("MISSING_YIELD", ssa::ErrorType::MISSING_YIELD);

  passes.def("verify_ssa", &pass::VerifySSA);

  py::enum_<typecheck::ErrorType>(passes, "TypeCheckErrorType")
      .value("TYPE_KIND_MISMATCH", typecheck::ErrorType::TYPE_KIND_MISMATCH)
      .value("DTYPE_MISMATCH", typecheck::ErrorType::DTYPE_MISMATCH)
      .value("SHAPE_DIMENSION_MISMATCH", typecheck::ErrorType::SHAPE_DIMENSION_MISMATCH)
      .value("SHAPE_VALUE_MISMATCH", typecheck::ErrorType::SHAPE_VALUE_MISMATCH)
      .value("SIZE_MISMATCH", typecheck::ErrorType::SIZE_MISMATCH);

  passes.def("type_check", &pass::TypeCheck);
  passes.def("convert_to_ssa", &pass::ConvertToSSA);

  py::enum_<DiagnosticSeverity>(passes, "DiagnosticSeverity")
      .value("Error", DiagnosticSeverity::Error)
      .value("Warning", DiagnosticSeverity::Warning);

  py::class_<Diagnostic>(passes, "Diagnostic")
      .def_readonly("severity", &Diagnostic::severity)
      .def_readonly("rule_name", &Diagnostic::rule_name)
      .def_readonly("error_code", &Diagnostic::error_code)
      .def_readonly("message", &Diagnostic::message)
      .def_readonly("span", &Diagnostic::span);

  py::class_<IRVerifier>(passes, "IRVerifier")
      .def(py::init<>())
      .def_static("create_default", &IRVerifier::CreateDefault)
      .def("enable_rule", &IRVerifier::EnableRule, py::arg("name"))
      .def("disable_rule", &IRVerifier::DisableRule, py::arg("name"))
      .def("is_rule_enabled", &IRVerifier::IsRuleEnabled, py::arg("name"))
      .def("verify", &IRVerifier::Verify, py::arg("program"))
      .def("verify_or_throw", &IRVerifier::VerifyOrThrow, py::arg("program"))
      .def_static("generate_report", &IRVerifier::GenerateReport, py::arg("diagnostics"));

  passes.def("run_verifier", &pass::RunVerifier, py::arg("disabled_rules") = std::vector<std::string>{});
}

} // namespace pypto
