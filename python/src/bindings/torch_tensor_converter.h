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
 * \file torch_tensor_converter.h
 * \brief Convert PyTorch tensors to device/on-board data (DeviceTensorData).
 */

#pragma once

#include "pybind_common.h"
#include "machine/runtime/device_launcher_binding.h"
#include "tilefwk/data_type.h"

#include <cstdint>
#include <vector>

namespace pypto {

// DLPack ABI structs (matches dlpack.h layout, no external dependency)
struct DLDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};
struct DLManagedTensor {
    struct DLTensor {
        void *data;
        int32_t device_type;
        int32_t device_id;
        int32_t ndim;
        DLDataType dtype;
        int64_t *shape;
        int64_t *strides;
        uint64_t byte_offset;
    } dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct DLManagedTensor *);
};

// DLPack dtype codes (DMLC base + PyTorch extensions)
constexpr uint8_t kDLInt = 0, kDLUInt = 1, kDLFloat = 2, kDLBfloat = 4;
constexpr uint8_t kDLComplex = 5, kDLBool = 6, kDLFloat8E5M2 = 7, kDLFloat8E4M3 = 8;

inline bool DlpackDtypeToDataType(uint8_t code, uint8_t bits, uint16_t lanes,
                                  npu::tile_fwk::DataType *out) {
    if (lanes != 1) return false;
    using DT = npu::tile_fwk::DataType;
    switch (code) {
        case kDLInt:
            switch (bits) {
                case 8: *out = DT::DT_INT8; return true;
                case 16: *out = DT::DT_INT16; return true;
                case 32: *out = DT::DT_INT32; return true;
                case 64: *out = DT::DT_INT64; return true;
                default: return false;
            }
        case kDLUInt:
            switch (bits) {
                case 8: *out = DT::DT_UINT8; return true;
                case 16: *out = DT::DT_UINT16; return true;
                case 32: *out = DT::DT_UINT32; return true;
                case 64: *out = DT::DT_UINT64; return true;
                default: return false;
            }
        case kDLFloat:
            switch (bits) {
                case 16: *out = DT::DT_FP16; return true;
                case 32: *out = DT::DT_FP32; return true;
                case 64: *out = DT::DT_DOUBLE; return true;
                default: return false;
            }
        case kDLBfloat:
            if (bits == 16) { *out = DT::DT_BF16; return true; }
            return false;
        case kDLBool:
            if (bits == 8) { *out = DT::DT_BOOL; return true; }
            return false;
        case kDLFloat8E5M2:
            if (bits == 8) { *out = DT::DT_FP8E5M2; return true; }
            return false;
        case kDLFloat8E4M3:
            if (bits == 8) { *out = DT::DT_FP8E4M3; return true; }
            return false;
        default:
            return false;
    }
}

py::object GetTorchToDlpack();

bool ParseDlpackCapsule(py::object cap, uintptr_t &data_ptr, std::vector<int64_t> &shape,
                       int *device_id = nullptr,
                       npu::tile_fwk::DataType *dtype_out = nullptr);

bool TryParseDlpack(py::object torch_tensor, uintptr_t &data_ptr, std::vector<int64_t> &shape,
                    int *device_id = nullptr, py::object to_dlpack = py::none(),
                    npu::tile_fwk::DataType *dtype_out = nullptr);

class TorchTensorConverter {
public:
    static void Convert(py::sequence tensors, py::sequence tensor_defs,
        std::vector<npu::tile_fwk::dynamic::DeviceTensorData> &tensors_data,
        std::vector<int> &device_ids);
};

class DeviceValidator {
public:
    static int ValidateAndGetDeviceId(const std::vector<int> &device_ids);
};

size_t ValidateInputs(py::sequence tensors, py::sequence tensor_defs);

}  // namespace pypto
