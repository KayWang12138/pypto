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

#include <vector>

#include "../bindings.h"
#include "ir/builder.h"
#include "ir/core.h"
#include "ir/expr.h"
#include "ir/function.h"
#include "ir/stmt.h"
#include "ir/type.h"

namespace py = pybind11;

namespace pypto {

using namespace pypto::ir;  // NOLINT(build/namespaces)

void BindIRBuilder(py::module_ &m) {
  py::module_ ir = m.def_submodule("ir");

  py::class_<IRBuilder>(ir, "IRBuilder")
      .def(py::init<>())
      .def("begin_function", &IRBuilder::BeginFunction, py::arg("name"), py::arg("span"),
           py::arg("type") = FunctionType::Opaque)
      .def("func_arg", &IRBuilder::FuncArg, py::arg("name"), py::arg("type"), py::arg("span"))
      .def("return_type", &IRBuilder::ReturnType, py::arg("type"))
      .def("end_function", &IRBuilder::EndFunction, py::arg("end_span"))
      .def("begin_for_loop", &IRBuilder::BeginForLoop, py::arg("loop_var"), py::arg("start"),
           py::arg("stop"), py::arg("step"), py::arg("span"))
      .def("add_iter_arg", &IRBuilder::AddIterArg, py::arg("iter_arg"))
      .def("add_return_var", &IRBuilder::AddReturnVar, py::arg("var"))
      .def("end_for_loop", &IRBuilder::EndForLoop, py::arg("end_span"))
      .def("begin_if", &IRBuilder::BeginIf, py::arg("condition"), py::arg("span"))
      .def("begin_else", &IRBuilder::BeginElse, py::arg("span"))
      .def("add_if_return_var", &IRBuilder::AddIfReturnVar, py::arg("var"))
      .def("end_if", &IRBuilder::EndIf, py::arg("end_span"))
      .def("emit", &IRBuilder::Emit, py::arg("stmt"))
      .def("assign", &IRBuilder::Assign, py::arg("var"), py::arg("value"), py::arg("span"))
      .def("var", &IRBuilder::Var, py::arg("name"), py::arg("type"), py::arg("span"))
      .def("return_", py::overload_cast<const std::vector<ExprPtr> &, const Span &>(&IRBuilder::Return),
           py::arg("values"), py::arg("span"))
      .def("return_", py::overload_cast<const Span &>(&IRBuilder::Return), py::arg("span"))
      .def("in_function", &IRBuilder::InFunction)
      .def("in_loop", &IRBuilder::InLoop)
      .def("in_if", &IRBuilder::InIf)
      .def("in_program", &IRBuilder::InProgram)
      .def("begin_program", &IRBuilder::BeginProgram, py::arg("name"), py::arg("span"))
      .def("declare_function", &IRBuilder::DeclareFunction, py::arg("func_name"))
      .def("get_global_var", &IRBuilder::GetGlobalVar, py::arg("func_name"))
      .def("add_function", &IRBuilder::AddFunction, py::arg("func"))
      .def("end_program", &IRBuilder::EndProgram, py::arg("end_span"))
      .def("get_function_return_types", &IRBuilder::GetFunctionReturnTypes, py::arg("gvar"));
}

} // namespace pypto
