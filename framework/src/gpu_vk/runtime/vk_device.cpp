#include "gpu_vk/runtime/vk_device.h"

namespace npu::tile_fwk::gpu_vk {

VkStatus VkDeviceContext::Initialize(std::uintptr_t instanceHandle) {
    instanceHandle_ = instanceHandle;
    available_ = false;
    nativeHandle_ = 0;
    computeQueueFamily_ = 0;
    return instanceHandle == 0 ? VkStatus::INVALID_ARGUMENT : VkStatus::UNAVAILABLE;
}

void VkDeviceContext::Destroy() {
    available_ = false;
    nativeHandle_ = 0;
    computeQueueFamily_ = 0;
}

} // namespace npu::tile_fwk::gpu_vk
