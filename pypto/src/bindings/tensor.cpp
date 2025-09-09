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
 * \file tensor.cpp
 * \brief
 */

#include "pybind_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void bind_tensor(py::module &m){
    py::class_<Tensor>(m, "tensor")
        .def(py::init<DataType, std::vector<int64_t>, std::string>(), py::arg("dtype"), py::arg("shape"),
            py::arg("name") = "Unknown")
        .def(
            "__add__", [](Tensor &self, Tensor tensor) { return npu::tile_fwk::Add(self, tensor); }, "Tensor add.")
        .def("get_dtype", &Tensor::GetDataType)
        .def_property_readonly(
            "shape", py::overload_cast<>(&Tensor::GetShape, py::const_), py::return_value_policy::reference_internal)
        .def("get_shape", py::overload_cast<>(&Tensor::GetShape, py::const_),
            py::return_value_policy::reference_internal)
        .def("assign",
            py::overload_cast<const Tensor&>(&Tensor::operator=),
            "Assigns from another tensor by copying its content.",
            py::return_value_policy::reference_internal
        )
        .def("move",
            [](Tensor &self, Tensor &other) -> Tensor& {
                self = std::move(other);
                return self;
            },
            "Assigns from another tensor by moving its content. The source tensor is left in an empty state.",
            py::arg("other"),
            py::return_value_policy::reference_internal
        );

    py::class_<Element>(m, "element")
        .def(py::init<DataType, int64_t>(), py::arg("type"), py::arg("sData"))
        .def(py::init<DataType, uint64_t>(), py::arg("type"), py::arg("uData"))
        .def(py::init<DataType, double>(), py::arg("type"), py::arg("fData"))
        .def("get_data_type", &Element::GetDataType)
        .def("get_signed_data", &Element::GetSignedData)
        .def("get_unsigned_data", &Element::GetUnsignedData)
        .def("get_float_data", &Element::GetFloatData);
}
}