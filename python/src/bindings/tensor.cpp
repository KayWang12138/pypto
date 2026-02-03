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
 * \file tensor.cpp
 * \brief
 */

#include "tilefwk/tensor.h"
#include <nanobind/nanobind.h>
#include "nb_common.h"

using namespace npu::tile_fwk;

namespace pypto {
void BindTensor(nb::module_ &m) {
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Tensor *self, DataType dtype, const nb::sequence &shape, const std::string &name, TileOpFormat format) {
                bool has_symbolic = false;
                for (const auto &item : shape) {
                    if (nb::isinstance<SymbolicScalar>(item)) {
                        has_symbolic = true;
                        break;
                    }
                }
                if (has_symbolic) {
                    std::vector<SymbolicScalar> symbolic_shape;
                    symbolic_shape.reserve(nb::len(shape));
                    for (const auto &item : shape) {
                        symbolic_shape.push_back(nb::cast<SymbolicScalar>(item));
                    }
                    return new (self) Tensor(dtype, symbolic_shape, name, format);
                } else {
                    std::vector<int64_t> int_shape;
                    int_shape.reserve(nb::len(shape));
                    for (const auto &item : shape) {
                        int_shape.push_back(nb::cast<int64_t>(item));
                    }
                    return new (self) Tensor(dtype, int_shape, name, format);
                }
            }, nb::rv_policy::reference,
            nb::arg("dtype"), nb::arg("shape"), nb::arg("name") = "", nb::arg("format") = TileOpFormat::TILEOP_ND)
        .def(nb::init<DataType, std::vector<int64_t>, uint8_t *, std::string, TileOpFormat>(), nb::arg("dtype"),
            nb::arg("shape"), nb::arg("data_ptr"), nb::arg("name"), nb::arg("format") = TileOpFormat::TILEOP_ND)
        .def(nb::init<DataType, std::vector<int64_t>, std::string>(), nb::arg("dtype"), nb::arg("shape"),
            nb::arg("name") = "int_init")
        .def(nb::init<DataType, std::vector<SymbolicScalar>, std::string>(), nb::arg("dtype"), nb::arg("shape"),
            nb::arg("name") = "SymbolicScalar_init")
        .def(nb::init<DataType, std::vector<int64_t>, std::string, TileOpFormat>(), nb::arg("dtype"), nb::arg("shape"),
            nb::arg("name") = "", nb::arg("format") = TileOpFormat::TILEOP_ND)
        .def(nb::init<DataType, std::vector<int64_t>, uint8_t *, std::string, TileOpFormat>(), nb::arg("dtype"),
            nb::arg("shape"), nb::arg("data_ptr"), nb::arg("name"), nb::arg("format") = TileOpFormat::TILEOP_ND)
        .def(nb::init<DataType, std::vector<SymbolicScalar>, std::string, TileOpFormat>(), nb::arg("dtype"),
            nb::arg("shape"), nb::arg("name") = "", nb::arg("format") = TileOpFormat::TILEOP_ND)
        .def("IsEmpty", &Tensor::IsEmpty)
        .def("Id", &Tensor::Id)
        .def("GetDataType",
            [](const Tensor &t) -> DataType {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.GetDataType();
            })
        .def_prop_ro("dtype",
            [](const Tensor &t) -> DataType {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.GetDataType();
            })
        .def_prop_ro("shape",
            [](const Tensor &t) -> std::vector<int64_t> {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.GetShape();
            })
        .def(
            "GetShape",
            [](const Tensor &t) -> std::vector<int64_t> {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.GetShape();
            },
            "Get the shape of the tensor.")
        .def("GetValidShape",
            [](const Tensor &t) -> std::vector<SymbolicScalar> {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.GetValidShape();
            })
        .def(
            "Move",
            [](Tensor &self, Tensor &other) -> Tensor & {
                self = std::move(other);
                return self;
            },
            "Assigns from another tensor by moving its content. The source tensor is left in an empty state.",
            nb::arg("other"), nb::rv_policy::reference_internal)
        .def(
            "SetCachePolicy",
            [](Tensor &t, CachePolicy policy, bool value) {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                t.SetCachePolicy(policy, value);
            },
            nb::arg("policy"), nb::arg("value"))
        .def(
            "GetCachePolicy",
            [](const Tensor &t, CachePolicy policy) {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.GetCachePolicy(policy);
            },
            nb::arg("policy"))
        .def(
            "SetName",
            [](Tensor &t, const std::string &name) {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                t.SetName(name);
            },
            nb::arg("name"))
        .def(
            "GetName",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.GetName();
            },
            "Get the name of the tensor.")
        .def(
            "Dim",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.Dim();
            },
            "Get the number of dimensions of the tensor.")
        .def(
            "Format",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw nb::value_error("Empty tensor.");
                }
                return t.Format();
            },
            "Get the format of the tensor.");

    m.def("GetInputShape",
        [](const Tensor &t, int axis) {
            if (t.IsEmpty()) {
                throw nb::value_error("Empty tensor.");
            }
            return npu::tile_fwk::GetInputShape(t, axis);
        },
        nb::arg("t"), nb::arg("axis"),
        "Get the shape of the input at the specified axis.");
    m.def("GetInputShape",
        [](const Tensor &t) {
            if (t.IsEmpty()) {
                throw nb::value_error("Empty tensor.");
            }
            return npu::tile_fwk::GetInputShape(t);
        },
        "Get the shape of the input.", nb::arg("t"));
    m.def("GetTensorData",
        [](const Tensor &t, std::vector<SymbolicScalar> offset) {
            if (t.IsEmpty()) {
                throw nb::value_error("Empty tensor.");
            }
            return npu::tile_fwk::GetTensorData(t, offset);
        },
        nb::arg("tensor"), nb::arg("offset"),
        "Get the tensor data at the specified offsets.");
    m.def("SetTensorData",
        [](const SymbolicScalar &value, std::vector<SymbolicScalar> offset, Tensor &dst) {
            if (dst.IsEmpty()) {
                throw nb::value_error("Empty tensor.");
            }
            npu::tile_fwk::SetTensorData(value, offset, dst);
        },
        nb::arg("value"), nb::arg("offset"), nb::arg("dst"),
        "Set the tensor data at the destination offset from the source value.");
}
} // namespace pypto
