#include "gpu_vk/runtime/vk_pipeline.h"

namespace npu::tile_fwk::gpu_vk {

VkStatus VkComputePipelineHandle::Create(
    const std::string &kernelName,
    const std::vector<std::uint32_t> &spirv,
    const std::vector<VkBindingDesc> &bindings) {
    kernelName_ = kernelName;
    bindings_ = bindings;
    created_ = !kernelName.empty() && !spirv.empty();
    return created_ ? VkStatus::SUCCESS : VkStatus::INVALID_ARGUMENT;
}

void VkComputePipelineHandle::Destroy() {
    created_ = false;
    kernelName_.clear();
    bindings_.clear();
}

} // namespace npu::tile_fwk::gpu_vk
