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

#include "bindings.h"
#include "core/error.h"

namespace py = pybind11;

namespace pypto {

[[noreturn]] static void raise_value_error(const std::string &message) { throw ir::ValueError(message); }
[[noreturn]] static void raise_type_error(const std::string &message) { throw ir::TypeError(message); }
[[noreturn]] static void raise_runtime_error(const std::string &message) { throw ir::RuntimeError(message); }
[[noreturn]] static void raise_not_implemented_error(const std::string &message) { throw ir::NotImplementedError(message); }
[[noreturn]] static void raise_index_error(const std::string &message) { throw ir::IndexError(message); }
[[noreturn]] static void raise_generic_error(const std::string &message) { throw ir::Error(message); }
[[noreturn]] static void raise_assertion_error(const std::string &message) { throw ir::AssertionError(message); }
[[noreturn]] static void raise_internal_error(const std::string &message) { throw ir::InternalError(message); }

void BindTesting(py::module_ &m) {
  py::module_ testing = m.def_submodule("testing", "Internal testing utilities (do not use in production)");

  testing.def("raise_value_error", &raise_value_error, py::arg("message"));
  testing.def("raise_type_error", &raise_type_error, py::arg("message"));
  testing.def("raise_runtime_error", &raise_runtime_error, py::arg("message"));
  testing.def("raise_not_implemented_error", &raise_not_implemented_error, py::arg("message"));
  testing.def("raise_index_error", &raise_index_error, py::arg("message"));
  testing.def("raise_generic_error", &raise_generic_error, py::arg("message"));
  testing.def("raise_assertion_error", &raise_assertion_error, py::arg("message"));
  testing.def("raise_internal_error", &raise_internal_error, py::arg("message"));
}

} // namespace pypto
