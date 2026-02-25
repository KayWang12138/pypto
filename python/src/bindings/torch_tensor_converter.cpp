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
 * \file torch_tensor_converter.cpp
 * \brief Implementation of PyTorch tensor to DeviceTensorData conversion.
 */

#include "bindings/torch_tensor_converter.h"

#include <stdexcept>
#include <string>

namespace pypto {

py::object GetTorchToDlpack() {
    try {
        return py::module::import("torch").attr("_C").attr("_to_dlpack");
    } catch (...) {
        return py::none();
    }
}

bool ParseDlpackCapsule(py::object cap, uintptr_t &data_ptr, std::vector<int64_t> &shape,
                        int *device_id, npu::tile_fwk::DataType *dtype_out) {
    if (cap.is_none()) return false;
    void *ptr = PyCapsule_GetPointer(cap.ptr(), "dltensor");
    if (!ptr) {
        PyErr_Clear();
        return false;
    }
    DLManagedTensor *tensor = static_cast<DLManagedTensor *>(ptr);
    DLManagedTensor::DLTensor &t = tensor->dl_tensor;

    data_ptr = reinterpret_cast<uintptr_t>(static_cast<char *>(t.data) + t.byte_offset);

    int32_t ndim = t.ndim;
    shape.clear();
    shape.reserve(ndim);
    for (int32_t i = 0; i < ndim; ++i) {
        shape.push_back(t.shape[i]);
    }

    if (device_id) *device_id = t.device_id;

    if (dtype_out) {
        DlpackDtypeToDataType(t.dtype.code, t.dtype.bits, t.dtype.lanes, dtype_out);
    }
    return true;
}

bool TryParseDlpack(py::object torch_tensor, uintptr_t &data_ptr, std::vector<int64_t> &shape,
                    int *device_id, py::object to_dlpack,
                    npu::tile_fwk::DataType *dtype_out) {
    if (to_dlpack.is_none()) to_dlpack = GetTorchToDlpack();
    if (to_dlpack.is_none()) return false;
    py::object cap;
    try {
        cap = to_dlpack(torch_tensor);
    } catch (...) {
        PyErr_Clear();
        return false;
    }
    return ParseDlpackCapsule(cap, data_ptr, shape, device_id, dtype_out);
}

void TorchTensorConverter::Convert(py::sequence tensors, py::sequence tensor_defs,
    std::vector<npu::tile_fwk::dynamic::DeviceTensorData> &tensors_data,
    std::vector<int> &device_ids) {
    using namespace npu::tile_fwk;
    using namespace npu::tile_fwk::dynamic;

    const size_t n = static_cast<size_t>(py::len(tensors));
    tensors_data.reserve(n);
    device_ids.reserve(n);

    py::object to_dlpack = GetTorchToDlpack();

    for (size_t i = 0; i < n; i++) {
        py::object torch_tensor = tensors[py::int_(i)];
        py::object tensor_def = tensor_defs[py::int_(i)];
        std::vector<int64_t> shape;
        uintptr_t data_ptr = 0;
        int device_id = -1;

        DataType dtype = DataType::DT_BOTTOM;
        if (!TryParseDlpack(torch_tensor, data_ptr, shape, &device_id, to_dlpack, &dtype)) {
            try {
                data_ptr = static_cast<uintptr_t>(py::cast<int64_t>(torch_tensor.attr("data_ptr")()));
                for (auto dim : torch_tensor.attr("shape")) {
                    shape.push_back(py::cast<int64_t>(dim));
                }
            } catch (...) {
                PyErr_Clear();
                throw std::runtime_error("Input tensor is not a valid torch tensor type");
            }
        }
        if (dtype == DataType::DT_BOTTOM) {
            auto base = py::getattr(tensor_def, "_base", py::none());
            if (!base.is_none() && py::isinstance<Tensor>(base)) {
                dtype = base.cast<Tensor &>().GetDataType();
            } else {
                dtype = tensor_def.attr("dtype").cast<DataType>();
            }
        }
        tensors_data.emplace_back(dtype, data_ptr, shape);
        device_ids.push_back(device_id);
    }
}

int DeviceValidator::ValidateAndGetDeviceId(const std::vector<int> &device_ids) {
    int device_id = -1;
    for (size_t i = 0; i < device_ids.size(); i++) {
        if (device_ids[i] < 0) continue;

        if (device_id < 0) {
            device_id = device_ids[i];
        } else if (device_ids[i] != device_id) {
            throw std::runtime_error(
                "Tensor at index " + std::to_string(i) +
                " is on device " + std::to_string(device_ids[i]) +
                ", expected device " + std::to_string(device_id));
        }
    }

    if (device_id < 0) {
        throw std::runtime_error(
            "Unable to determine device ID: ensure all tensors support DLPack");
    }
    return device_id;
}

size_t ValidateInputs(py::sequence tensors, py::sequence tensor_defs) {
    size_t n = static_cast<size_t>(py::len(tensors));
    if (n != static_cast<size_t>(py::len(tensor_defs))) {
        throw std::runtime_error(
            "Input length mismatch: tensors(" + std::to_string(n) +
            ") vs tensor_defs(" + std::to_string(py::len(tensor_defs)) + ")");
    }
    if (n == 0) {
        throw std::runtime_error("Empty tensor list");
    }
    return n;
}

}  // namespace pypto
