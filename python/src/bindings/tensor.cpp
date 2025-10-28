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

/**
 * @brief Return true if the slice is empty, i.e., b[:]
 *
 * @param slice Python slice
 * @return true If slice does not have any start, stop or step.
 * @return false Otherwise.
 */
bool IsEmptySlice(const py::slice &slice) {
    py::object start = slice.attr("start");
    py::object stop = slice.attr("stop");
    py::object step = slice.attr("step");

    return start.is_none() and stop.is_none() and step.is_none();
}

void TensorSetItem(Tensor &self, py::object key, py::object value) {
    if (!py::isinstance<Tensor>(value)) {
        throw std::runtime_error("Tensor.__setitem__ value must be a Tensor object.");
    }

    if (py::isinstance<py::slice>(key)) {
        py::slice slice = key.cast<py::slice>();
        if (!IsEmptySlice(slice)) {
            throw std::runtime_error(
                "Tensor.__setitem__ supports only [:]. Arbitrary slices are reserved for Tensor.Assemble.'");
        }

        // Move the input value tensor to self.
        Tensor &tensor_value = value.cast<Tensor &>();
        self = std::move(tensor_value);
    } else {
        throw std::runtime_error("Tensor.__setitem__ key type must be slice. Use 'tensor[:] = <source tensor>'");
    }
}

void BindTensor(py::module &m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init([](DataType dtype, const py::sequence &shape, const std::string &name, TileOpFormat format) {
            bool has_symbolic = false;
            for (const auto &item : shape) {
                if (py::isinstance<SymbolicScalar>(item)) {
                    has_symbolic = true;
                    break;
                }
            }
            if (has_symbolic) {
                std::vector<SymbolicScalar> symbolic_shape;
                symbolic_shape.reserve(py::len(shape));
                for (const auto &item : shape) {
                    symbolic_shape.push_back(item.cast<SymbolicScalar>());
                }
                return std::make_unique<Tensor>(dtype, symbolic_shape, name, format);
            } else {
                std::vector<int64_t> int_shape;
                int_shape.reserve(py::len(shape));
                for (const auto &item : shape) {
                    int_shape.push_back(item.cast<int64_t>());
                }
                return std::make_unique<Tensor>(dtype, int_shape, name, format);
            }
        }),
            py::arg("dtype"), py::arg("shape"), py::arg("name") = "", py::arg("format") = TileOpFormat::TILEOP_ND)
        .def(py::init<DataType, std::vector<int64_t>, uint8_t *, std::string, TileOpFormat>(), py::arg("dtype"),
            py::arg("shape"), py::arg("data_ptr"), py::arg("name"), py::arg("format") = TileOpFormat::TILEOP_ND)
        .def(py::init<DataType, std::vector<int64_t>, std::string>(), py::arg("dtype"), py::arg("shape"),
            py::arg("name") = "int_init")
        .def(py::init<DataType, std::vector<SymbolicScalar>, std::string>(), py::arg("dtype"), py::arg("shape"),
            py::arg("name") = "SymbolicScalar_init")
        .def(py::init<DataType, std::vector<int64_t>, std::string, TileOpFormat>(), py::arg("dtype"), py::arg("shape"),
            py::arg("name") = "", py::arg("format") = TileOpFormat::TILEOP_ND)
        .def(py::init<DataType, std::vector<int64_t>, uint8_t *, std::string, TileOpFormat>(), py::arg("dtype"),
            py::arg("shape"), py::arg("data_ptr"), py::arg("name"), py::arg("format") = TileOpFormat::TILEOP_ND)
        .def(py::init<DataType, std::vector<SymbolicScalar>, std::string, TileOpFormat>(), py::arg("dtype"),
            py::arg("shape"), py::arg("name") = "", py::arg("format") = TileOpFormat::TILEOP_ND)
        .def(
            "__add__", [](const Tensor &self, const Tensor &tensor) { return npu::tile_fwk::Add(self, tensor); },
            "Tensor add.")
        .def(
            "__add__", [](const Tensor &self, const Element &element) { return npu::tile_fwk::Add(self, element); },
            "Tensor-element add.")
        .def(
            "__radd__", [](const Tensor &self, const Element &element) { return npu::tile_fwk::Add(self, element); },
            "Tensor-element add.")
        .def(
            "__sub__", [](const Tensor &self, const Tensor &tensor) { return npu::tile_fwk::Sub(self, tensor); },
            "Tensor subtraction.")
        .def(
            "__sub__", [](const Tensor &self, const Element &element) { return npu::tile_fwk::Sub(self, element); },
            "Tensor-element subtraction.")
        .def("GetDataType", &Tensor::GetDataType)
        .def("get_dtype", &Tensor::GetDataType)
        .def_property_readonly(
            "dtype", py::overload_cast<>(&Tensor::GetDataType, py::const_), py::return_value_policy::reference_internal)
        .def_property_readonly(
            "shape", py::overload_cast<>(&Tensor::GetShape, py::const_), py::return_value_policy::reference_internal)
        .def(
            "GetShape", py::overload_cast<>(&Tensor::GetShape, py::const_), py::return_value_policy::reference_internal)
        .def("GetShapeAt", py::overload_cast<int>(&Tensor::GetShape, py::const_), py::arg("axis"))
        .def("Assign", py::overload_cast<const Tensor &>(&Tensor::operator=),
            "Assigns from another tensor by copying its content.", py::return_value_policy::reference_internal)
        .def(
            "Move",
            [](Tensor &self, Tensor &other) -> Tensor & {
                self = std::move(other);
                return self;
            },
            "Assigns from another tensor by moving its content. The source tensor is left in an empty state.",
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("__setitem__", &TensorSetItem,
            "Assigns from another tensor by moving its content. The source tensor is left in an empty state.",
            py::arg("key"), py::arg("value"))
        .def("SetCachePolicy", &Tensor::SetCachePolicy, py::arg("policy"), py::arg("value"))
        .def("GetCachePolicy", &Tensor::GetCachePolicy, py::arg("policy"))
        .def("GetStorage", [](const Tensor &self) { return self.GetStorage(false) != nullptr; })
        .def("Id", &Tensor::Id, "Get the index of the tensor.")
        .def("SetName", &Tensor::SetName, py::arg("name"))
        .def("GetName", &Tensor::GetName)
        .def("Dim", &Tensor::Dim, "Get the number of dimensions of the tensor.")
        .def_property_readonly("id", &Tensor::Id, "Get the index of the tensor.");
    m.def("GetInputShape",
        [](const Tensor& t, int axis){
            return npu::tile_fwk::GetInputShape(t, axis);
        },
        "Get the shape of the input at the specified axis.",
        py::arg("t"), py::arg("axis"));
    m.def("GetInputShape",
        [](const Tensor& t){
            return npu::tile_fwk::GetInputShape(t);
        },
        "Get the shape of the input.",
        py::arg("t"));
    m.def("GetTensorData", &GetTensorData, py::arg("tensor"), py::arg("offset"),
        "Get the tensor data at the specified offsets.");
    m.def("SetTensorData", &SetTensorData, py::arg("value"), py::arg("offset"), py::arg("dst"),
        "Set the tensor data at the destination offset from the source value.");
}
} // namespace pypto
