/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file symbolic_scalar.cpp
 * \brief
 */

#include "pybind_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void bind_symbolic_scalar(py::module &m) {
    py::class_<NotLessThan>(m, "not_less_than").def(py::init<int64_t>(), py::arg("threshold"));
    py::class_<NotGreaterThan>(m, "not_greater_than").def(py::init<int64_t>(), py::arg("threshold"));

    py::class_<SymbolicScalar> SymbolicScalar(m, "symbolic_scalar");

    SymbolicScalar
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("name"))
        .def(py::init<std::int64_t>(), py::arg("value"))
        .def(py::init<std::string, NotLessThan>(), py::arg("name"), py::arg(">"))
        .def(py::init<std::string, NotGreaterThan>(), py::arg("name"), py::arg("<"))
        .def(py::init<std::string, NotLessThan, NotGreaterThan>(), py::arg("name"), py::arg("<"), py::arg(">"))
        .def(py::init<std::string, int64_t>(), py::arg("name"), py::arg("value"));

    SymbolicScalar
        .def("is_immediate", &SymbolicScalar::IsImmediate)
        .def("is_symbol", &SymbolicScalar::IsSymbol)
        .def("is_expression", &SymbolicScalar::IsExpression)
        .def("is_valid", &SymbolicScalar::IsValid)
        .def("concrete_valid", &SymbolicScalar::ConcreteValid)
        .def("concrete", py::overload_cast<>(&SymbolicScalar::Concrete, py::const_))
        .def("__eq__", &SymbolicScalar::Eq) // Total ordering / comparisons
        .def("__ne__", &SymbolicScalar::Ne)
        .def("__lt__", &SymbolicScalar::Lt)
        .def("__leq__", &SymbolicScalar::Le)
        .def("__gt__", &SymbolicScalar::Gt)
        .def("__ge__", &SymbolicScalar::Ge)
        .def("__add__", &SymbolicScalar::Add) // Binary operators
        .def("__sub__", &SymbolicScalar::Sub)
        .def("__mul__", &SymbolicScalar::Mul)
        .def("__truediv__", &SymbolicScalar::Div)
        .def("__mod__", &SymbolicScalar::Mod);

    SymbolicScalar
        .def("as_intermediate_variable", &SymbolicScalar::AsIntermediateVariable)
        .def("is_intermediate_variable", &SymbolicScalar::IsIntermediateVariable)
        .def("dump", &SymbolicScalar::Dump)
        .def("min", &SymbolicScalar::Min, py::arg("other"))
        .def("max", &SymbolicScalar::Max, py::arg("other"));

    SymbolicScalar
        .def("__int__", &SymbolicScalar::operator int, "Convert to an integer if concrete value is valid.")
        .def("__str__", &SymbolicScalar::Dump, "String representation for print().")
        .def("__repr__", &SymbolicScalar::Dump, "String representation for display.");

    SymbolicScalar
        .def("__pos__", &SymbolicScalar::Pos)
        .def("__neg__", &SymbolicScalar::Neg)
        .def("__invert__", &SymbolicScalar::Not);

    SymbolicScalar
        .def(py::self + int())
        .def(int() + py::self)
        .def(py::self - int())
        .def(int() - py::self)
        .def(py::self * int())
        .def(int() * py::self)
        .def(py::self / int())
        .def(int() / py::self)
        .def(py::self % int())
        .def(int() % py::self)
        .def(py::self == int())
        .def(int() == py::self)
        .def(py::self != int())
        .def(int() != py::self)
        .def(py::self < int())
        .def(int() < py::self)
        .def(py::self <= int())
        .def(int() <= py::self)
        .def(py::self > int())
        .def(int() > py::self)
        .def(py::self >= int())
        .def(int() >= py::self);
}
} // namespace pypto