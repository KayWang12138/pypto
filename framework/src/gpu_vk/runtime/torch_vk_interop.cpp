#include "gpu_vk/runtime/torch_vk_interop.h"

#include <algorithm>

namespace npu::tile_fwk::gpu_vk {

namespace {

std::vector<std::int64_t> ExtractShape(const py::object &tensor) {
    std::vector<std::int64_t> shape;
    for (const py::handle &item : tensor.attr("shape")) {
        shape.push_back(item.cast<std::int64_t>());
    }
    return shape;
}

std::vector<std::int64_t> MakeStride(const std::vector<std::int64_t> &shape) {
    std::vector<std::int64_t> stride(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i) {
        stride[i - 2] = stride[i - 1] * std::max<std::int64_t>(shape[i - 1], 1);
    }
    return stride;
}

std::size_t ExtractBytes(const py::object &tensor) {
    const auto numel = tensor.attr("numel")().cast<std::int64_t>();
    const auto elementSize = tensor.attr("element_size")().cast<std::int64_t>();
    return static_cast<std::size_t>(std::max<std::int64_t>(0, numel * elementSize));
}

} // namespace

VkStatus TorchVkInterop::ImportFromTorchCpu(const py::object &tensor, VkTensorStorage &out) const {
    if (tensor.is_none()) {
        return VkStatus::INVALID_ARGUMENT;
    }
    out.mode = VkTensorStorageMode::CPU_STAGING;
    out.desc.shape = ExtractShape(tensor);
    out.desc.stride = MakeStride(out.desc.shape);
    out.desc.nbytes = ExtractBytes(tensor);
    out.hostData.assign(out.desc.nbytes, 0);
    return VkStatus::SUCCESS;
}

VkStatus TorchVkInterop::ExportToTorchCpu(const VkTensorStorage &storage, py::object &outTensor) const {
    py::module_ torch = py::module_::import("torch");
    py::tuple shape(storage.desc.shape.size());
    for (std::size_t i = 0; i < storage.desc.shape.size(); ++i) {
        shape[i] = py::int_(storage.desc.shape[i]);
    }
    outTensor = torch.attr("zeros")(shape);
    return VkStatus::SUCCESS;
}

VkStatus TorchVkInterop::TryImportExternalMemory(const py::object &tensor, VkTensorStorage &out) const {
    (void)tensor;
    out.mode = VkTensorStorageMode::EXTERNAL_MEMORY;
    out.hostData.clear();
    return VkStatus::UNAVAILABLE;
}

} // namespace npu::tile_fwk::gpu_vk
