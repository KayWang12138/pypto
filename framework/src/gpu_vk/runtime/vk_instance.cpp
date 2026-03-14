#include "gpu_vk/runtime/vk_instance.h"

namespace npu::tile_fwk::gpu_vk {

VkStatus VkInstanceContext::Initialize(bool enableValidation) {
    validationEnabled_ = enableValidation;
    available_ = false;
    nativeHandle_ = 0;
    return VkStatus::UNAVAILABLE;
}

void VkInstanceContext::Destroy() {
    available_ = false;
    nativeHandle_ = 0;
}

} // namespace npu::tile_fwk::gpu_vk
