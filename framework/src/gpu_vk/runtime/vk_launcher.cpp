#include "gpu_vk/runtime/vk_launcher.h"

namespace npu::tile_fwk::gpu_vk {

VkStatus VkLauncher::Initialize(bool enableValidation) {
    validationEnabled_ = enableValidation;
    available_ = false;
    return VkStatus::UNAVAILABLE;
}

void VkLauncher::Destroy() {
    available_ = false;
}

VkStatus VkLauncher::RunElementwiseBinary(
    const std::string &opName,
    const void *input0,
    const void *input1,
    void *output,
    const VkTensorDesc &desc) {
    if (opName.empty() || input0 == nullptr || input1 == nullptr || output == nullptr || desc.nbytes == 0) {
        return VkStatus::INVALID_ARGUMENT;
    }
    return VkStatus::UNAVAILABLE;
}

VkStatus VkLauncher::RunElementwiseUnary(const std::string &opName, const void *input, void *output, const VkTensorDesc &desc) {
    if (opName.empty() || input == nullptr || output == nullptr || desc.nbytes == 0) {
        return VkStatus::INVALID_ARGUMENT;
    }
    return VkStatus::UNAVAILABLE;
}

} // namespace npu::tile_fwk::gpu_vk
