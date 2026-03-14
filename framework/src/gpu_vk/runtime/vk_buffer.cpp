#include "gpu_vk/runtime/vk_buffer.h"

#include <cstring>

namespace npu::tile_fwk::gpu_vk {

VkStatus VkBufferHandle::Allocate(std::size_t bytes, VkBufferUsageKind usage) {
    usage_ = usage;
    data_.assign(bytes, 0);
    return VkStatus::SUCCESS;
}

VkStatus VkBufferHandle::Upload(const void *src, std::size_t bytes) {
    if (src == nullptr) {
        return VkStatus::INVALID_ARGUMENT;
    }
    if (bytes > data_.size()) {
        return VkStatus::OUT_OF_MEMORY;
    }
    if (bytes != 0) {
        std::memcpy(data_.data(), src, bytes);
    }
    return VkStatus::SUCCESS;
}

VkStatus VkBufferHandle::Download(void *dst, std::size_t bytes) const {
    if (dst == nullptr) {
        return VkStatus::INVALID_ARGUMENT;
    }
    if (bytes > data_.size()) {
        return VkStatus::OUT_OF_MEMORY;
    }
    if (bytes != 0) {
        std::memcpy(dst, data_.data(), bytes);
    }
    return VkStatus::SUCCESS;
}

void VkBufferHandle::Destroy() {
    data_.clear();
}

} // namespace npu::tile_fwk::gpu_vk
