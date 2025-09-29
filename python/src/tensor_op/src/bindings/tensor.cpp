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
            "__add__", [](Tensor &self, Tensor tensor) { return npu::tile_fwk::Add(self, tensor); }, "Tensor add.")
        .def("get_dtype", &Tensor::GetDataType)
        .def_property_readonly(
            "shape", py::overload_cast<>(&Tensor::GetShape, py::const_), py::return_value_policy::reference_internal)
        .def("get_shape", py::overload_cast<>(&Tensor::GetShape, py::const_),
            py::return_value_policy::reference_internal)
        .def("get_shape_at",py::overload_cast<int>(&Tensor::GetShape, py::const_),py::arg("axis"))
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
        )
        .def("set_cache_policy", &Tensor::SetCachePolicy, py::arg("policy"), py::arg("value"))
        .def("get_cache_policy", &Tensor::GetCachePolicy, py::arg("policy"))
        .def("has_storage", [](const Tensor &self) { return self.GetStorage(false) != nullptr; })
        .def("id", &Tensor::Id, "Get the index of the tensor.")
        .def_property_readonly("id", &Tensor::Id, "Get the index of the tensor.");
    m.def("get_input_shape", &GetInputShape, py::arg("index"), py::arg("input_index"),
         "Get the shape of the input at the specified index.");
    m.def("get_input_data", &GetInputData, py::arg("index"), py::arg("data_offsets"),
        "Get the input data at the specified offsets.");
    m.def("get_tensor_data", &GetTensorData, py::arg("index"), py::arg("data_offsets"),
        "Get the tensor data at the specified offsets.");
    m.def("set_tensor_data", &SetTensorData, py::arg("value"), py::arg("src_offset"), py::arg("dst_offset"),
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