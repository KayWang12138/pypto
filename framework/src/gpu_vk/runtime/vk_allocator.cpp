#include "gpu_vk/runtime/vk_allocator.h"

namespace npu::tile_fwk::gpu_vk {

VkStatus VkAllocator::Initialize() {
    available_ = true;
    return VkStatus::SUCCESS;
}

VkStatus VkAllocator::AllocStorageBuffer(std::size_t bytes, VkBufferHandle &out) {
    if (!available_) {
        return VkStatus::UNAVAILABLE;
    }
    return out.Allocate(bytes, VkBufferUsageKind::STORAGE);
}

VkStatus VkAllocator::AllocStagingBuffer(std::size_t bytes, VkBufferHandle &out) {
    if (!available_) {
        return VkStatus::UNAVAILABLE;
    }
    return out.Allocate(bytes, VkBufferUsageKind::STAGING);
}

void VkAllocator::Destroy() {
    available_ = false;
}

} // namespace npu::tile_fwk::gpu_vk
