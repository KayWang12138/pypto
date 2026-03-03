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
 * \file testing.cpp
 * \brief Python bindings for internal testing utilities
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "../bindings.h"
#include "core/error.h"

namespace py = pybind11;

namespace pypto {

[[noreturn]] static void raiseValueError(const std::string &message) {
    throw ir::ValueError(message);
}
[[noreturn]] static void raiseTypeError(const std::string &message) {
    throw ir::TypeError(message);
}
[[noreturn]] static void raiseRuntimeError(const std::string &message) {
    throw ir::RuntimeError(message);
}
[[noreturn]] static void raiseNotImplementedError(const std::string &message) {
    throw ir::NotImplementedError(message);
}
[[noreturn]] static void raiseIndexError(const std::string &message) {
    throw ir::IndexError(message);
}
[[noreturn]] static void raiseGenericError(const std::string &message) {
    throw ir::Error(message);
}
[[noreturn]] static void raiseAssertionError(const std::string &message) {
    throw ir::AssertionError(message);
}
[[noreturn]] static void raiseInternalError(const std::string &message) {
    throw ir::InternalError(message);
}

void BindTesting(py::module_ &m) {
    py::module_ testing = m.def_submodule("testing", "Internal testing utilities (do not use in production)");

    testing.def("raise_value_error", &raiseValueError, py::arg("message"));
    testing.def("raise_type_error", &raiseTypeError, py::arg("message"));
    testing.def("raise_runtime_error", &raiseRuntimeError, py::arg("message"));
    testing.def("raise_not_implemented_error", &raiseNotImplementedError, py::arg("message"));
    testing.def("raise_index_error", &raiseIndexError, py::arg("message"));
    testing.def("raise_generic_error", &raiseGenericError, py::arg("message"));
    testing.def("raise_assertion_error", &raiseAssertionError, py::arg("message"));
    testing.def("raise_internal_error", &raiseInternalError, py::arg("message"));
}

} // namespace pypto
