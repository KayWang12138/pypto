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

#include "pybind_common.h"
#include "passes/tensor_graph_pass/vjp_registry.h"
#include "interface/program/program.h"
#include "interface/utils/log.h"

using namespace npu::tile_fwk;

namespace {
constexpr const char* PROGRAM_ENTRY_FUNCTION_NAME = "PROGRAM_ENTRY";

bool IsGradientForMagic(const LogicalTensorPtr &tensor, int64_t magic) {
    if (tensor == nullptr) {
        return false;
    }
    bool isGradient = false;
    if (!tensor->GetAttr("is_gradient", isGradient) || !isGradient) {
        return false;
    }
    std::vector<int64_t> magics;
    if (tensor->GetAttr("gradient_of_magics", magics)) {
        for (int64_t value : magics) {
            if (value == magic) {
                return true;
            }
        }
        return false;
    }
    int64_t gradOfMagic = 0;
    return tensor->GetAttr("gradient_of_magic", gradOfMagic) && gradOfMagic == magic;
}

bool FindIncastMagic(const Tensor &t, const Function &func, int64_t *incastMagic) {
    if (incastMagic == nullptr) {
        return false;
    }
    auto storage = t.GetStorage(false);
    if (!storage) {
        return false;
    }
    auto dyndevAttr = func.GetDyndevAttribute();
    if (!dyndevAttr) {
        return false;
    }
    auto rawTensor = storage->GetRawTensor();
    if (!rawTensor) {
        return false;
    }
    int rawmagic = rawTensor->GetRawMagic();
    const auto &savedList = dyndevAttr->startArgsInputLogicalTensorList;
    const auto &incasts = func.GetIncast();
    for (size_t i = 0; i < savedList.size(); ++i) {
        const auto &savedStorage = savedList[i];
        if (!savedStorage || !savedStorage->GetRawTensor()) {
            continue;
        }
        if (savedStorage->GetRawTensor()->GetRawMagic() == rawmagic) {
            if (i >= incasts.size()) {
                ALOG_WARN_F("FindIncastMagic: RawMagic match at index %zu but incasts.size()=%zu",
                            i, incasts.size());
                return false;
            }
            if (!incasts[i]) {
                ALOG_WARN_F("FindIncastMagic: RawMagic match at index %zu but incast is nullptr", i);
                return false;
            }
            *incastMagic = incasts[i]->GetMagic();
            return true;
        }
    }
    return false;
}

int FindGradientOutputIndex(const Function &func, int64_t incastMagic, LogicalTensorPtr *gradTensor) {
    const auto &outcasts = func.GetOutcast();
    for (size_t i = 0; i < outcasts.size(); ++i) {
        if (IsGradientForMagic(outcasts[i], incastMagic)) {
            if (gradTensor != nullptr) {
                *gradTensor = outcasts[i];
            }
            return static_cast<int>(i);
        }
    }
    return -1;
}
} // namespace

namespace pypto {
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
        .def("IsEmpty", &Tensor::IsEmpty)
        .def("Id", &Tensor::Id)
        .def("GetDataType",
            [](const Tensor &t) -> DataType {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return t.GetDataType();
            })
        .def_property_readonly("dtype",
            [](const Tensor &t) -> DataType {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return t.GetDataType();
            })
        .def_property_readonly("shape",
            [] (const Tensor &t) -> std::vector<int64_t> {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return t.GetShape();
            })
        .def("GetShape",
            [](const Tensor &t) -> std::vector<int64_t> {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return t.GetShape();
            },
            "Get the shape of the tensor.")
        .def("get_storage_magic",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    return -1;
                }
                return t.GetStorage()->GetMagic();
            },
            "Get the magic number of the underlying storage (for debugging).")
        .def("Move",
            [](Tensor &self, Tensor &other) -> Tensor& {
                self = std::move(other);
                return self;
            },
            "Assigns from another tensor by moving its content. The source tensor is left in an empty state.",
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("SetCachePolicy",
            [](Tensor &t, CachePolicy policy, bool value) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                t.SetCachePolicy(policy, value);
            },
            py::arg("policy"), py::arg("value"))
        .def("GetCachePolicy",
            [](const Tensor &t, CachePolicy policy) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return t.GetCachePolicy(policy);
            },
            py::arg("policy"))
        .def("SetName",
            [](Tensor &t, const std::string &name) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                t.SetName(name);
            },
            py::arg("name"))
        .def("GetName",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return t.GetName();
            },
            "Get the name of the tensor.")
        .def("Dim",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return t.Dim();
            },
            "Get the number of dimensions of the tensor.")
        .def("Format",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return t.Format();
            },
            "Get the format of the tensor.")
        .def_property("requires_grad",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                bool value = false;
                t.GetStorage()->GetAttr(ATTR_REQUIRES_GRAD, value);
                return value;
            },
            [](Tensor &t, bool value) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                t.GetStorage()->SetAttr(ATTR_REQUIRES_GRAD, value);
            },
            "Whether this tensor requires gradients to be computed during backward pass.")
        .def_property("is_loss",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                bool value = false;
                t.GetStorage()->GetAttr(ATTR_IS_LOSS, value);
                return value;
            },
            [](Tensor &t, bool value) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                t.GetStorage()->SetAttr(ATTR_IS_LOSS, value);
            },
            "Whether this tensor is the loss tensor for backward pass.")
        .def("get_gradient_magic",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                // First try direct attribute lookup on storage
                auto storage = t.GetStorage(false);
                if (storage) {
                    int64_t magic = 0;
                    if (storage->GetAttr("gradient_magic", magic) && magic > 0) {
                        return magic;
                    }
                }
                // Fallback: use FindIncastMagic to find matching inCast (like get_gradient_tensor)
                Function *lastFunc = Program::GetInstance().GetLastFunction();
                if (lastFunc == nullptr) {
                    return static_cast<int64_t>(-1);
                }
                int64_t incastMagic = 0;
                if (!FindIncastMagic(t, *lastFunc, &incastMagic)) {
                    return static_cast<int64_t>(-1);
                }
                // Look up gradient from inCast
                LogicalTensorPtr gradTensor;
                if (FindGradientOutputIndex(*lastFunc, incastMagic, &gradTensor) >= 0 && gradTensor != nullptr) {
                    return static_cast<int64_t>(gradTensor->GetMagic());
                }
                return static_cast<int64_t>(-1);
            },
            "Get the magic number of the gradient tensor (returns -1 if no gradient).")
        .def("get_belong_function",
            [](const Tensor &t) -> Function* {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                return &t.GetStorage()->BelongFunction();
            },
            py::return_value_policy::reference,
            "Get the Function this tensor belongs to.")
        .def("get_gradient_tensor",
            [](const Tensor &t) -> py::object {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                auto storage = t.GetStorage(false);
                if (!storage) {
                    return py::none();
                }

                // Primary path: use gradient_magic attribute with Program's registry (O(1) lookup)
                int64_t gradMagic = 0;
                if (storage->GetAttr("gradient_magic", gradMagic) && gradMagic > 0) {
                    auto gradTensor = Program::GetInstance().GetGradientTensor(static_cast<int>(gradMagic));
                    if (gradTensor) {
                        return py::cast(Tensor(gradTensor));
                    }
                }

                // Fallback: search by is_gradient/gradient_of_magic attributes in lastFunc
                Function *lastFunc = Program::GetInstance().GetLastFunction();
                if (!lastFunc) {
                    return py::none();
                }
                int64_t incastMagic = 0;
                if (FindIncastMagic(t, *lastFunc, &incastMagic)) {
                    for (const auto &op : lastFunc->Operations().DuplicatedOpList()) {
                        for (const auto &output : op->GetOOperands()) {
                            if (IsGradientForMagic(output, incastMagic)) {
                                return py::cast(Tensor(output));
                            }
                        }
                    }
                }
                return py::none();
            },
            "Get the gradient Tensor. Returns None if no gradient.")
        .def("get_gradient_output_index",
            [](const Tensor &t) {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                auto storage = t.GetStorage(false);
                if (!storage) {
                    return -1;
                }
                int64_t gradMagic = 0;
                if (storage->GetAttr("gradient_magic", gradMagic) && gradMagic > 0) {
                    Function &func = storage->BelongFunction();
                    const auto &outcasts = func.GetOutcast();
                    for (size_t i = 0; i < outcasts.size(); ++i) {
                        if (outcasts[i] && outcasts[i]->GetMagic() == gradMagic) {
                            return static_cast<int>(i);
                        }
                    }
                }
                Function *lastFunc = Program::GetInstance().GetLastFunction();
                if (lastFunc == nullptr) {
                    return -1;
                }
                int64_t incastMagic = 0;
                if (!FindIncastMagic(t, *lastFunc, &incastMagic)) {
                    return -1;
                }
                return FindGradientOutputIndex(*lastFunc, incastMagic, nullptr);
            },
            "Get the output index where this tensor's gradient is stored (returns -1 if no gradient).")
        .def("get_gradient_info",
            [](const Tensor &t) -> py::object {
                if (t.IsEmpty()) {
                    throw py::value_error("Empty tensor.");
                }
                auto storage = t.GetStorage(false);
                if (!storage) {
                    return py::none();
                }
                int64_t gradMagic = 0;
                if (storage->GetAttr("gradient_magic", gradMagic) && gradMagic > 0) {
                    // Use Program's gradient registry (O(1) lookup)
                    auto gradTensor = Program::GetInstance().GetGradientTensor(static_cast<int>(gradMagic));
                    if (gradTensor != nullptr) {
                        py::dict info;
                        info["magic"] = gradTensor->GetMagic();
                        info["shape"] = std::vector<int64_t>(gradTensor->GetShape().begin(), gradTensor->GetShape().end());
                        info["dtype"] = gradTensor->Datatype();
                        info["output_index"] = -1;  // Not in outcasts
                        return info;
                    }
                    // Try outcasts
                    Function &func = storage->BelongFunction();
                    const auto &outcasts = func.GetOutcast();
                    for (size_t i = 0; i < outcasts.size(); ++i) {
                        if (outcasts[i] && outcasts[i]->GetMagic() == gradMagic) {
                            py::dict info;
                            info["magic"] = outcasts[i]->GetMagic();
                            info["shape"] = std::vector<int64_t>(outcasts[i]->GetShape().begin(), outcasts[i]->GetShape().end());
                            info["dtype"] = outcasts[i]->Datatype();
                            info["output_index"] = static_cast<int>(i);
                            return info;
                        }
                    }
                }

                Function *lastFunc = Program::GetInstance().GetLastFunction();
                if (!lastFunc) {
                    return py::none();
                }

                int64_t incastMagic = 0;
                if (!FindIncastMagic(t, *lastFunc, &incastMagic)) {
                    return py::none();
                }

                LogicalTensorPtr gradTensor;
                int outputIndex = FindGradientOutputIndex(*lastFunc, incastMagic, &gradTensor);
                if (outputIndex < 0 || gradTensor == nullptr) {
                    return py::none();
                }

                py::dict info;
                info["magic"] = gradTensor->GetMagic();
                info["shape"] = std::vector<int64_t>(gradTensor->GetShape().begin(), gradTensor->GetShape().end());
                info["dtype"] = gradTensor->Datatype();
                info["output_index"] = outputIndex;
                return info;
            },
            "Get gradient tensor info (magic, shape, dtype, output_index) or None if no gradient.");

    m.def("GetInputShape",
        [](const Tensor &t, int axis) {
            if (t.IsEmpty()) {
                throw py::value_error("Empty tensor.");
            }
            return npu::tile_fwk::GetInputShape(t, axis);
        },
        py::arg("t"), py::arg("axis"),
        "Get the shape of the input at the specified axis.");
    m.def("GetInputShape",
        [](const Tensor &t) {
            if (t.IsEmpty()) {
                throw py::value_error("Empty tensor.");
            }
            return npu::tile_fwk::GetInputShape(t);
        },
        "Get the shape of the input.", py::arg("t"));
    m.def("GetTensorData",
        [](const Tensor &t, std::vector<SymbolicScalar> offset) {
            if (t.IsEmpty()) {
                throw py::value_error("Empty tensor.");
            }
            return npu::tile_fwk::GetTensorData(t, offset);
        },
        py::arg("tensor"), py::arg("offset"),
        "Get the tensor data at the specified offsets.");
    m.def("SetTensorData",
        [](const SymbolicScalar &value, std::vector<SymbolicScalar> offset, Tensor &dst) {
            if (dst.IsEmpty()) {
                throw py::value_error("Empty tensor.");
            }
            npu::tile_fwk::SetTensorData(value, offset, dst);
        },
        py::arg("value"), py::arg("offset"), py::arg("dst"),
        "Set the tensor data at the destination offset from the source value.");
}
} // namespace pypto
