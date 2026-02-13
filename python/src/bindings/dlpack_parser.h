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
 * \file dlpack_parser.h
 * \brief DLPack capsule parsing for converting PyTorch/NumPy tensors to device tensor metadata.
 */

 #pragma once

 #include "Python.h"
 #include "pybind_common.h"
 
 #include <cstdint>
 #include <vector>
 
 namespace pypto {
 
 // DLPack ABI-compatible structs for parsing "dltensor" capsule.
 struct DLTensorView {
     void *data;
     int32_t device_type;
     int32_t device_id;
     int32_t ndim;
     uint8_t dtype_code;
     uint8_t dtype_bits;
     uint16_t dtype_lanes;
     int64_t *shape;
     int64_t *strides;
     uint64_t byte_offset;
 };
 struct DLManagedTensorView {
     DLTensorView dl_tensor;
     void *manager_ctx;
     void (*deleter)(void *);
 };
 
 namespace {
 
 inline py::object GetTorchToDlpack() {
     try {
         py::module torch = py::module::import("torch");
         return torch.attr("_C").attr("_to_dlpack");
     } catch (...) {
         return py::none();
     }
 }
 
 inline bool ParseDlpackCapsule(py::object cap, uintptr_t &data_ptr, std::vector<int64_t> &shape,
                                int *device_id = nullptr) {
     if (cap.is_none()) return false;
     void *ptr = PyCapsule_GetPointer(cap.ptr(), "dltensor");
     if (!ptr) {
         PyErr_Clear();
         return false;
     }
     auto &t = static_cast<DLManagedTensorView *>(ptr)->dl_tensor;
     data_ptr = reinterpret_cast<uintptr_t>(static_cast<char *>(t.data) + t.byte_offset);
     shape.assign(t.shape, t.shape + t.ndim);
     if (device_id) *device_id = t.device_id;
     return true;
 }
 
 inline py::object GetTorchToDlpackCapsule(py::object torch_tensor) {
     py::object to_dlpack = GetTorchToDlpack();
     if (to_dlpack.is_none()) return py::none();
     try {
         return to_dlpack(torch_tensor);
     } catch (...) {
         PyErr_Clear();
         return py::none();
     }
 }
 
 inline py::object GetNativeDlpackCapsule(py::object torch_tensor) {
     if (!py::hasattr(torch_tensor, "__dlpack__")) return py::none();
     try {
         return torch_tensor.attr("__dlpack__")(py::arg("stream") = -1);
     } catch (...) {
         PyErr_Clear();
         return py::none();
     }
 }
 
 }  // namespace
 
 /// Parse DLPack capsule from PyTorch or any __dlpack__-compatible tensor.
 /// Attempts torch._C._to_dlpack first, then falls back to __dlpack__().
 inline bool TryParseDlpack(py::object torch_tensor, uintptr_t &data_ptr, std::vector<int64_t> &shape,
                            int *device_id = nullptr) {
     py::object cap = GetTorchToDlpackCapsule(torch_tensor);
     if (!cap.is_none() && ParseDlpackCapsule(cap, data_ptr, shape, device_id)) return true;
     cap = GetNativeDlpackCapsule(torch_tensor);
     if (!cap.is_none() && ParseDlpackCapsule(cap, data_ptr, shape, device_id)) return true;
     return false;
 }
 
 }  // namespace pypto
 