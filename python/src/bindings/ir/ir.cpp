/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/*!
 * \file error.cpp
 * \brief Python bindings for PyPTO error classes
 */

#include "bindings.h"

#include "ir/expr.h"
#include "core/error.h"

namespace pypto {
namespace ir {

void BindDType(py::module& m)
{
    py::class_<ir::DataType>(m, "DataType", "Enumeration of available data types")
        .def_readonly_static("BOOL", &ir::DataType::BOOL)
        .def_readonly_static("INT4", &ir::DataType::INT4)
        .def_readonly_static("INT8", &ir::DataType::INT8)
        .def_readonly_static("INT16", &ir::DataType::INT16)
        .def_readonly_static("INT32", &ir::DataType::INT32)
        .def_readonly_static("INT64", &ir::DataType::INT64)
        .def_readonly_static("UINT4", &ir::DataType::UINT4)
        .def_readonly_static("UINT8", &ir::DataType::UINT8)
        .def_readonly_static("UINT16", &ir::DataType::UINT16)
        .def_readonly_static("UINT32", &ir::DataType::UINT32)
        .def_readonly_static("UINT64", &ir::DataType::UINT64)
        .def_readonly_static("FP4", &ir::DataType::FP4)
        .def_readonly_static("FP8E4M3FN", &ir::DataType::FP8E4M3FN)
        .def_readonly_static("FP8E5M2", &ir::DataType::FP8E5M2)
        .def_readonly_static("FP16", &ir::DataType::FP16)
        .def_readonly_static("FP32", &ir::DataType::FP32)
        .def_readonly_static("BF16", &ir::DataType::BF16)
        .def_readonly_static("HF4", &ir::DataType::HF4)
        .def_readonly_static("HF8", &ir::DataType::HF8)
        .def_readonly_static("INDEX", &ir::DataType::INDEX)
        .def("bits", &ir::DataType::GetBit, "Get the size in bits of this data type.")
        .def("c_type", &ir::DataType::ToCTypeString, "Get C style type string for code generation.")
        .def("is_float", &ir::DataType::IsFloat, "Check if this data type is a floating point type.")
        .def("is_signed", &ir::DataType::IsSignedInt, "Check if this data type is a signed integer type.")
        .def("is_unsigned", &ir::DataType::IsUnsignedInt, "Check if this data type is an unsigned integer type.")
        .def("is_int", &ir::DataType::IsInt, "Check if this data type is an integer type.")
        .def("__int__", &ir::DataType::Code, "Get the underlying type code.")
        .def("__eq__", &ir::DataType::operator==, py::arg("other"))
        .def("__ne__", &ir::DataType::operator!=, py::arg("other"))
        .def("__repr__", &ir::DataType::ToString)
        .def("__str__", &ir::DataType::ToString);
}

void BindSpan(py::module& m)
{
    py::class_<ir::Span>(m, "Span", "Source location information tracking file, line, and column positions")
        .def(
            py::init<std::string, int, int, int, int>(), py::arg("filename"), py::arg("begin_line"),
            py::arg("begin_column"), py::arg("end_line") = -1, py::arg("end_column") = -1, "Create a source span")
        .def_static("is_unknown", &ir::Span::IsUnknown, "Check if the span is unknown")
        .def_static("unknown", &ir::Span::Unknown, "Create an unknown span", py::return_value_policy::reference)
        .def("__repr__", &ir::Span::ToString)
        .def("__str__", &ir::Span::ToString)
        .def_readonly("filename", &ir::Span::filename_, "Source filename")
        .def_readonly("begin_line", &ir::Span::beginLine_, "Beginning line (1-indexed)")
        .def_readonly("begin_column", &ir::Span::beginColumn_, "Beginning column (1-indexed)")
        .def_readonly("end_line", &ir::Span::endLine_, "Ending line (1-indexed)")
        .def_readonly("end_column", &ir::Span::endColumn_, "Ending column (1-indexed)");
}

} // namespace ir

void BindIR(py::module& m)
{
    auto m1 = m.def_submodule("ir");
    ir::BindDType(m1);
    ir::BindSpan(m1);
}
} // namespace pypto
