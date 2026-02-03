/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file symbolic_scalar.cpp
 * \brief
 */

#include "nb_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindSymbolicScalar(nb::module_ &m) {
    nb::class_<SymbolicScalar> _SymbolicScalar(m, "SymbolicScalar");

    _SymbolicScalar
        .def(nb::init<>())
        .def(nb::init<const SymbolicScalar &>(), nb::arg("val"))
        .def(nb::init<std::string>(), nb::arg("name"))
        .def(nb::init<std::int64_t>(), nb::arg("value"))
        .def(nb::init<std::string, int64_t>(), nb::arg("name"), nb::arg("value"));

    nb::implicitly_convertible<int64_t, SymbolicScalar>();
    nb::implicitly_convertible<int, SymbolicScalar>();

    _SymbolicScalar.def("IsImmediate", &SymbolicScalar::IsImmediate)
        .def("IsSymbol", &SymbolicScalar::IsSymbol)
        .def("IsExpression", &SymbolicScalar::IsExpression)
        .def("IsValid", &SymbolicScalar::IsValid)
        .def("ConcreteValid", &SymbolicScalar::ConcreteValid)
        .def("Concrete", nb::overload_cast<>(&SymbolicScalar::Concrete, nb::const_))
        .def("Eq", &SymbolicScalar::Eq) // Total ordering / comparisons
        .def("Ne", &SymbolicScalar::Ne)
        .def("Lt", &SymbolicScalar::Lt)
        .def("Le", &SymbolicScalar::Le)
        .def("Gt", &SymbolicScalar::Gt)
        .def("Ge", &SymbolicScalar::Ge)   // Total ordering / comparisons
        .def("Add", &SymbolicScalar::Add) // Binary operators
        .def("Sub", &SymbolicScalar::Sub)
        .def("Mul", &SymbolicScalar::Mul)
        .def("Div", &SymbolicScalar::Div)
        .def("Mod", &SymbolicScalar::Mod)
        .def("RAdd", [](const SymbolicScalar &self, int64_t other) { return other + self; })
        .def("RSub", [](const SymbolicScalar &self, int64_t other) { return other - self; })
        .def("RMul", [](const SymbolicScalar &self, int64_t other) { return other * self; })
        .def("RDiv", [](const SymbolicScalar &self, int64_t other) { return other / self; })
        .def("RMod", [](const SymbolicScalar &self, int64_t other) { return other % self; });

    _SymbolicScalar
        .def("AsIntermediateVariable", &SymbolicScalar::AsIntermediateVariable)
        .def("IsIntermediateVariable", &SymbolicScalar::IsIntermediateVariable)
        .def("Dump", &SymbolicScalar::Dump)
        .def("Min", &SymbolicScalar::Min, nb::arg("other"))
        .def("Max", &SymbolicScalar::Max, nb::arg("other"));

    _SymbolicScalar
        .def("Pos", &SymbolicScalar::Pos)
        .def("Neg", &SymbolicScalar::Neg)
        .def("Not", &SymbolicScalar::Not);
}
} // namespace pypto
