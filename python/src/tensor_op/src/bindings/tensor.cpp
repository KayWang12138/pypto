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
void BindTensor(py::module &m){
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init([](DataType dtype, const py::sequence& shape, const std::string& name, TileOpFormat format) {
            bool has_symbolic = false;
            for (const auto& item : shape) {
                if (py::isinstance<SymbolicScalar>(item)) {
                    has_symbolic = true;
                    break;
                }
            }
            if (has_symbolic) {
                std::vector<SymbolicScalar> symbolic_shape;
                symbolic_shape.reserve(py::len(shape));
                for (const auto& item : shape) {
                    symbolic_shape.push_back(item.cast<SymbolicScalar>());
                }
                return std::make_unique<Tensor>(dtype, symbolic_shape, name, format);
            } else {
                std::vector<int64_t> int_shape;
                int_shape.reserve(py::len(shape));
                for (const auto& item : shape) {
                    int_shape.push_back(item.cast<int64_t>());
                }
                return std::make_unique<Tensor>(dtype, int_shape, name, format);
            }
        }),
        py::arg("dtype"), py::arg("shape"), py::arg("name") = "", py::arg("format") = TileOpFormat::TILEOP_ND)
        .def(py::init<DataType, std::vector<int64_t>, uint8_t *, std::string, TileOpFormat>(),
            py::arg("dtype"), py::arg("shape"),  py::arg("data_ptr"), py::arg("name"), py::arg("format") = TileOpFormat::TILEOP_ND)
        .def(
            "__add__", [](Tensor &self, Tensor other) { return npu::tile_fwk::Add(self, other); }, "Tensor add.")
        .def("GetDataType", &Tensor::GetDataType)
        .def_property_readonly(
            "shape", py::overload_cast<>(&Tensor::GetShape, py::const_), py::return_value_policy::reference_internal)
        .def("GetShape", py::overload_cast<>(&Tensor::GetShape, py::const_),
            py::return_value_policy::reference_internal)
        .def("GetShapeAt", py::overload_cast<int>(&Tensor::GetShape, py::const_), py::arg("axis"))
        .def("Assign",
            py::overload_cast<const Tensor&>(&Tensor::operator=),
            "Assigns from another tensor by copying its content.",
            py::return_value_policy::reference_internal
        )
        .def("Move",
            [](Tensor &self, Tensor &other) -> Tensor& {
                self = std::move(other);
                return self;
            },
            "Assigns from another tensor by moving its content. The source tensor is left in an empty state.",
            py::arg("other"),
            py::return_value_policy::reference_internal
        )
        .def("SetCachePolicy", &Tensor::SetCachePolicy, py::arg("policy"), py::arg("value"))
        .def("GetCachePolicy", &Tensor::GetCachePolicy, py::arg("policy"))
        .def("GetStorage", [](const Tensor &self) { return self.GetStorage(false) != nullptr; })
        .def("Id", &Tensor::Id, "Get the index of the tensor.")
        .def_property_readonly("id", &Tensor::Id, "Get the index of the tensor.");
    m.def("GetInputShape", &GetInputShape, py::arg("tensor"), py::arg("axis"),
         "Get the shape of the input at the specified axis.");
    m.def("GetInputData", &GetInputData, py::arg("tensor"), py::arg("offset"),
        "Get the input data at the specified offsets.");
    m.def("GetTensorData", &GetTensorData, py::arg("tensor"), py::arg("offset"),
        "Get the tensor data at the specified offsets.");
    m.def("SetTensorData", &SetTensorData, py::arg("value"), py::arg("offset"), py::arg("dst"),
        "Set the tensor data at the destination offset from the source value.");

    py::class_<Element>(m, "element")
        .def(py::init<DataType, int64_t>(), py::arg("type"), py::arg("sData"))
        .def(py::init<DataType, uint64_t>(), py::arg("type"), py::arg("uData"))
        .def(py::init<DataType, double>(), py::arg("type"), py::arg("fData"))
        .def("get_data_type", &Element::GetDataType)
        .def("get_signed_data", &Element::GetSignedData)
        .def("get_unsigned_data", &Element::GetUnsignedData)
        .def("get_float_data", &Element::GetFloatData);

    m.def("get_input_data", &GetInputData, py::arg("tensor"), py::arg("offset"));
}
}