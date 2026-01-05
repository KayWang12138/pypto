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
 * \file function.cpp
 * \brief Python bindings for Function class
 */

#include "pybind_common.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "tilefwk/tensor.h"

using namespace npu::tile_fwk;

namespace pypto {
namespace {
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

std::vector<int64_t> GetGradientOfMagics(const LogicalTensorPtr &tensor) {
    std::vector<int64_t> magics;
    if (tensor == nullptr) {
        return magics;
    }
    if (tensor->GetAttr("gradient_of_magics", magics)) {
        return magics;
    }
    int64_t gradOfMagic = 0;
    if (tensor->GetAttr("gradient_of_magic", gradOfMagic)) {
        magics.push_back(gradOfMagic);
    }
    return magics;
}
} // namespace

void BindFunction(py::module &m) {
    // Bind the Function class
    py::class_<Function, std::shared_ptr<Function>>(m, "Function")
        .def("GetMagicName", &Function::GetMagicName, "Get the magic name of the function")
        .def("GetRawName", &Function::GetRawName, "Get the raw name of the function")
        .def("GetFuncMagic", &Function::GetFuncMagic, "Get the function magic number")
        .def("Dump", &Function::Dump, "Dump the function in brief format")
        .def("DumpSSA", &Function::DumpSSA, "Dump the function in SSA format")
        .def("GetFunctionType", &Function::GetFunctionType, "Get the function type")
        .def("GetFunctionTypeStr", &Function::GetFunctionTypeStr, "Get the function type as string")
        .def("GetGraphType", &Function::GetGraphType, "Get the graph type")
        .def("IsEager", &Function::IsEager, "Check if function is eager")
        .def("IsStatic", &Function::IsStatic, "Check if function is static")
        .def("IsExplicit", &Function::IsExplicit, "Check if function is explicit")
        .def("IsFunctionType", py::overload_cast<FunctionType>(&Function::IsFunctionType, py::const_), py::arg("type"),
            "Check if function is of given type")
        .def("IsGraphType", py::overload_cast<GraphType>(&Function::IsGraphType, py::const_), py::arg("type"),
            "Check if function has given graph type")
        .def("IsFunctionTypeAndGraphType",
            py::overload_cast<FunctionType, GraphType>(&Function::IsFunctionTypeAndGraphType, py::const_),
            py::arg("func_type"), py::arg("graph_type"), "Check if function has given function type and graph type")
        .def("HasParent", &Function::HasParent, "Check if function has a parent")
        .def("GetRootFunction", &Function::GetRootFunction, py::return_value_policy::reference, "Get the root function")
        .def(
            "GetIncast",
            [](const Function &self) -> std::vector<Tensor> {
                return std::vector<Tensor>(self.GetIncast().begin(), self.GetIncast().end());
            },
            py::return_value_policy::reference_internal, "Get input casts")
        .def(
            "GetOutcast",
            [](const Function &self) -> std::vector<Tensor> {
                return std::vector<Tensor>(self.GetOutcast().begin(), self.GetOutcast().end());
            },
            py::return_value_policy::reference_internal, "Get output casts")
        .def(
            "GetOriginIncast",
            [](const Function &self) -> std::vector<Tensor> {
                return std::vector<Tensor>(self.GetOriginIncast().begin(), self.GetOriginIncast().end());
            },
            py::return_value_policy::reference_internal, "Get original input casts")
        .def(
            "GetOriginOutcast",
            [](const Function &self) -> std::vector<Tensor> {
                return std::vector<Tensor>(self.GetOriginOutcast().begin(), self.GetOriginOutcast().end());
            },
            py::return_value_policy::reference_internal, "Get original output casts")
        .def("DumpFile", &Function::DumpFile, py::arg("file_path"), "Dump the function to a file")
        .def(
            "DumpJsonFile", [](Function &self, const std::string &fileName) { self.DumpJsonFile(fileName); },
            py::arg("file_name") = "", "Dump the function to a JSON file")
        .def("__repr__", [](const Function &self) {
            return "<Function '" + self.GetRawName() + "' (magic: " + std::to_string(self.GetFuncMagic()) + ")>";
        })
        .def(
            "GetTensorByMagic",
            [](const Function &self, int magic) -> py::object {
                // Use TensorMap for O(1) lookup
                auto logicalTensor = self.GetTensorMap().GetTensorByMagic(magic);
                if (logicalTensor != nullptr) {
                    return py::cast(Tensor(logicalTensor));
                }
                // Fallback: check outCasts (gradients are added to outCasts)
                const auto &outcasts = self.GetOutcast();
                for (const auto &outcast : outcasts) {
                    if (outcast && outcast->GetMagic() == magic) {
                        return py::cast(Tensor(outcast));
                    }
                }
                // Fallback: check inCasts
                const auto &incasts = self.GetIncast();
                for (const auto &incast : incasts) {
                    if (incast && incast->GetMagic() == magic) {
                        return py::cast(Tensor(incast));
                    }
                }
                return py::none();
            },
            py::arg("magic"),
            "Get a Tensor by its magic number using TensorMap (O(1) lookup). Returns None if not found.")
        .def(
            "get_gradient_output_index",
            [](const Function &self, int64_t tensorMagic) -> int {
                // Find the output index for the gradient of a tensor with given magic
                const auto &outcasts = self.GetOutcast();
                for (size_t i = 0; i < outcasts.size(); ++i) {
                    if (IsGradientForMagic(outcasts[i], tensorMagic)) {
                        return static_cast<int>(i);
                    }
                }
                return -1;  // Not found
            },
            py::arg("tensor_magic"),
            "Get the output index for the gradient of a tensor (returns -1 if not found)")
        .def(
            "get_gradient_tensors_info",
            [](const Function &self) -> py::dict {
                py::dict result;

                // Check incasts for gradient_magic and use Program's registry (O(1) lookup)
                const auto &incasts = self.GetIncast();
                for (size_t i = 0; i < incasts.size(); ++i) {
                    if (!incasts[i]) continue;
                    int64_t gradMagic = 0;
                    if (!incasts[i]->GetAttr("gradient_magic", gradMagic) || gradMagic <= 0) {
                        continue;
                    }
                    auto gradTensor = Program::GetInstance().GetGradientTensor(static_cast<int>(gradMagic));
                    if (!gradTensor) continue;

                    py::dict gradInfo;
                    gradInfo["magic"] = gradTensor->GetMagic();
                    gradInfo["output_index"] = -1;  // Not in outcasts
                    gradInfo["shape"] = std::vector<int64_t>(gradTensor->GetShape().begin(), gradTensor->GetShape().end());
                    gradInfo["dtype"] = gradTensor->Datatype();
                    result[py::int_(incasts[i]->GetMagic())] = gradInfo;
                }

                // Also check outcasts for backward compatibility
                const auto &outcasts = self.GetOutcast();
                for (size_t i = 0; i < outcasts.size(); ++i) {
                    bool isGradient = false;
                    if (!outcasts[i] || !outcasts[i]->GetAttr("is_gradient", isGradient) || !isGradient) {
                        continue;
                    }
                    auto gradOfMagics = GetGradientOfMagics(outcasts[i]);
                    if (gradOfMagics.empty()) {
                        continue;
                    }
                    py::dict gradInfo;
                    gradInfo["magic"] = outcasts[i]->GetMagic();
                    gradInfo["output_index"] = static_cast<int>(i);
                    gradInfo["shape"] = std::vector<int64_t>(outcasts[i]->GetShape().begin(), outcasts[i]->GetShape().end());
                    gradInfo["dtype"] = outcasts[i]->Datatype();
                    for (int64_t magic : gradOfMagics) {
                        result[py::int_(magic)] = gradInfo;
                    }
                }
                return result;
            },
            "Get a dictionary mapping original tensor magic to gradient tensor info (includes output_index)");

    // Add a function to get the last function from the Program
    m.def(
        "GetLastFunction", []() -> Function * { return Program::GetInstance().GetLastFunction(); },
        py::return_value_policy::reference, "Get the last compiled function from the Program");

    // Also add GetCurrentFunction for completeness
    m.def(
        "GetCurrentFunction", []() -> Function * { return Program::GetInstance().GetCurrentFunction(); },
        py::return_value_policy::reference, "Get the current function being built in the Program");
}

} // namespace pypto
