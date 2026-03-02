/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file passes.cpp
 * \brief Python bindings for IR transformation passes and verification
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "../bindings.h"
#include "core/error.h"
#include "ir/transform/passes.h"
#include "ir/transform/verifier.h"

namespace py = pybind11;

namespace pypto {

using namespace pypto::ir; // NOLINT(build/namespaces)

void BindPass(py::module_ &m) {
    py::module_ passes = m.def_submodule("passes", "IR transformation passes");

    py::class_<Pass>(passes, "Pass").def("__call__", &Pass::operator(), py::arg("program"));

    py::enum_<DiagnosticSeverity>(passes, "DiagnosticSeverity")
        .value("Error", DiagnosticSeverity::Error)
        .value("Warning", DiagnosticSeverity::Warning);

    py::class_<Diagnostic>(passes, "Diagnostic")
        .def_readonly("severity", &Diagnostic::severity)
        .def_readonly("rule_name", &Diagnostic::ruleName)
        .def_readonly("error_code", &Diagnostic::errorCode)
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
}

} // namespace pypto
