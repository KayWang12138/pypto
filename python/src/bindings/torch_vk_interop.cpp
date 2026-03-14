#include "pybind_common.h"

#include <string>

#include "gpu_vk/runtime/torch_vk_interop.h"

namespace pypto {

using npu::tile_fwk::gpu_vk::TorchVkInterop;
using npu::tile_fwk::gpu_vk::VkTensorStorage;
using npu::tile_fwk::gpu_vk::VkTensorStorageMode;

namespace {

const char *StorageModeToString(VkTensorStorageMode mode) {
    switch (mode) {
        case VkTensorStorageMode::CPU_STAGING:
            return "cpu_staging";
        case VkTensorStorageMode::VK_BUFFER:
            return "vk_buffer";
        case VkTensorStorageMode::EXTERNAL_MEMORY:
            return "external_memory";
        default:
            return "unknown";
    }
}

} // namespace

void BindTorchVkInterop(py::module &m) {
    py::enum_<VkTensorStorageMode>(m, "GpuVkTensorStorageMode")
        .value("CPU_STAGING", VkTensorStorageMode::CPU_STAGING)
        .value("VK_BUFFER", VkTensorStorageMode::VK_BUFFER)
        .value("EXTERNAL_MEMORY", VkTensorStorageMode::EXTERNAL_MEMORY);

    py::class_<VkTensorStorage>(m, "GpuVkTensorStorage")
        .def(py::init<>())
        .def_property_readonly("mode", [](const VkTensorStorage &storage) { return storage.mode; })
        .def_property_readonly("desc", [](const VkTensorStorage &storage) { return storage.desc; })
        .def_property_readonly("host_nbytes", [](const VkTensorStorage &storage) { return storage.hostData.size(); })
        .def_property_readonly("mode_name", [](const VkTensorStorage &storage) { return std::string(StorageModeToString(storage.mode)); });

    py::class_<TorchVkInterop>(m, "GpuVkTorchInterop")
        .def(py::init<>())
        .def(
            "import_from_torch_cpu",
            [](const TorchVkInterop &interop, const py::object &tensor) {
                VkTensorStorage storage;
                interop.ImportFromTorchCpu(tensor, storage);
                return storage;
            },
            py::arg("tensor"))
        .def(
            "export_to_torch_cpu",
            [](const TorchVkInterop &interop, const VkTensorStorage &storage) {
                py::object tensor;
                interop.ExportToTorchCpu(storage, tensor);
                return tensor;
            },
            py::arg("storage"))
        .def(
            "try_import_external_memory",
            [](const TorchVkInterop &interop, const py::object &tensor) {
                VkTensorStorage storage;
                interop.TryImportExternalMemory(tensor, storage);
                return storage;
            },
            py::arg("tensor"));
}

} // namespace pypto
