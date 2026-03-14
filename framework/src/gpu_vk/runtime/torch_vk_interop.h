#pragma once

#include <pybind11/pybind11.h>

#include "gpu_vk/common/vk_status.h"
#include "gpu_vk/runtime/vk_tensor_storage.h"

namespace py = pybind11;

namespace npu::tile_fwk::gpu_vk {

class TorchVkInterop {
public:
    VkStatus ImportFromTorchCpu(const py::object &tensor, VkTensorStorage &out) const;
    VkStatus ExportToTorchCpu(const VkTensorStorage &storage, py::object &outTensor) const;
    VkStatus TryImportExternalMemory(const py::object &tensor, VkTensorStorage &out) const;
};

} // namespace npu::tile_fwk::gpu_vk
