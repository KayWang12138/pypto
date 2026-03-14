#include "gpu_vk/runtime/vk_allocator.h"

namespace npu::tile_fwk::gpu_vk {

VkStatus VkAllocator::Initialize(std::uintptr_t deviceHandle, std::uintptr_t physicalDeviceHandle) {
    if (deviceHandle == 0 || physicalDeviceHandle == 0) {
        available_ = false;
        deviceHandle_ = 0;
        physicalDeviceHandle_ = 0;
        return VkStatus::kInvalidArgument;
    }
    deviceHandle_ = deviceHandle;
    physicalDeviceHandle_ = physicalDeviceHandle;
    available_ = true;
    return VkStatus::kSuccess;
}

VkStatus VkAllocator::AllocStorageBuffer(std::size_t bytes, VkBufferHandle &out) {
    if (!available_) {
        return VkStatus::kUnavailable;
    }
    return out.Allocate(deviceHandle_, physicalDeviceHandle_, bytes, VkBufferUsageKind::STORAGE);
}

VkStatus VkAllocator::AllocStagingBuffer(std::size_t bytes, VkBufferHandle &out) {
    if (!available_) {
        return VkStatus::kUnavailable;
    }
    return out.Allocate(deviceHandle_, physicalDeviceHandle_, bytes, VkBufferUsageKind::STAGING);
}

void VkAllocator::Destroy() {
    available_ = false;
    deviceHandle_ = 0;
    physicalDeviceHandle_ = 0;
}

} // namespace npu::tile_fwk::gpu_vk
