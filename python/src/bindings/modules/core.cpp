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

#include "../bindings.h"
#include "core/dtype.h"

namespace py = pybind11;

namespace pypto {

using pypto::ir::DataType;

void BindCore(py::module_ &m) {
  // Bind to ir submodule to avoid name collision with tilefwk DataType enum on root module
  py::module_ ir = m.def_submodule("ir");
  py::class_<DataType>(ir, "DataType")
      .def_readonly_static("BOOL", &DataType::BOOL)
      .def_readonly_static("INT4", &DataType::INT4)
      .def_readonly_static("INT8", &DataType::INT8)
      .def_readonly_static("INT16", &DataType::INT16)
      .def_readonly_static("INT32", &DataType::INT32)
      .def_readonly_static("INT64", &DataType::INT64)
      .def_readonly_static("UINT4", &DataType::UINT4)
      .def_readonly_static("UINT8", &DataType::UINT8)
      .def_readonly_static("UINT16", &DataType::UINT16)
      .def_readonly_static("UINT32", &DataType::UINT32)
      .def_readonly_static("UINT64", &DataType::UINT64)
      .def_readonly_static("FP4", &DataType::FP4)
      .def_readonly_static("FP8E4M3FN", &DataType::FP8E4M3FN)
      .def_readonly_static("FP8E5M2", &DataType::FP8E5M2)
      .def_readonly_static("FP8", &DataType::FP8)
      .def_readonly_static("FP16", &DataType::FP16)
      .def_readonly_static("FP32", &DataType::FP32)
      .def_readonly_static("BF16", &DataType::BF16)
      .def_readonly_static("HF4", &DataType::HF4)
      .def_readonly_static("HF8", &DataType::HF8)
      .def("get_bit", &DataType::GetBit)
      .def("to_string", &DataType::ToString)
      .def("to_c_type_string", &DataType::ToCTypeString)
      .def("is_float", &DataType::IsFloat)
      .def("is_signed_int", &DataType::IsSignedInt)
      .def("is_unsigned_int", &DataType::IsUnsignedInt)
      .def("is_int", &DataType::IsInt)
      .def("code", &DataType::Code)
      .def("__eq__", &DataType::operator==, py::arg("other"))
      .def("__ne__", &DataType::operator!=, py::arg("other"))
      .def("__repr__", &DataType::ToString)
      .def("__str__", &DataType::ToString);
}

} // namespace pypto
