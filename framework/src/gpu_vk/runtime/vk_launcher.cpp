#include "gpu_vk/runtime/vk_launcher.h"

namespace npu::tile_fwk::gpu_vk {

VkStatus VkLauncher::Initialize(bool enableValidation) {
    Destroy();

    const VkStatus instanceStatus = instance_.Initialize(enableValidation);
    if (instanceStatus != VkStatus::kSuccess) {
        validationEnabled_ = false;
        available_ = false;
        return instanceStatus;
    }

    const VkStatus deviceStatus = device_.Initialize(instance_.NativeHandle());
    if (deviceStatus != VkStatus::kSuccess) {
        instance_.Destroy();
        validationEnabled_ = false;
        available_ = false;
        return deviceStatus;
    }

    validationEnabled_ = instance_.ValidationEnabled();
    available_ = true;
    return VkStatus::kSuccess;
}

void VkLauncher::Destroy() {
    available_ = false;
    validationEnabled_ = false;
    device_.Destroy();
    instance_.Destroy();
}

VkStatus VkLauncher::RunElementwiseBinary(
    const std::string &opName,
    const void *input0,
    const void *input1,
    void *output,
    const VkTensorDesc &desc) {
    if (opName.empty() || input0 == nullptr || input1 == nullptr || output == nullptr || desc.nbytes == 0) {
        return VkStatus::kInvalidArgument;
    }
    if (!available_) {
        return VkStatus::kUnavailable;
    }
    return VkStatus::kUnavailable;
}

VkStatus VkLauncher::RunElementwiseUnary(const std::string &opName, const void *input, void *output, const VkTensorDesc &desc) {
    if (opName.empty() || input == nullptr || output == nullptr || desc.nbytes == 0) {
        return VkStatus::kInvalidArgument;
    }
    if (!available_) {
        return VkStatus::kUnavailable;
    }
    return VkStatus::kUnavailable;
}

} // namespace npu::tile_fwk::gpu_vk
