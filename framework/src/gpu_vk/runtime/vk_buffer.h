#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

enum class VkBufferUsageKind {
    STAGING = 0,
    STORAGE = 1,
};

class VkBufferHandle {
public:
    VkStatus Allocate(
        std::uintptr_t deviceHandle,
        std::uintptr_t physicalDeviceHandle,
        std::size_t bytes,
        VkBufferUsageKind usage);
    VkStatus Upload(const void *src, std::size_t bytes);
    VkStatus Download(void *dst, std::size_t bytes) const;
    void Destroy();

    std::size_t Size() const { return size_; }
    VkBufferUsageKind Usage() const { return usage_; }
    bool Allocated() const { return bufferHandle_ != 0; }
    bool HostVisible() const { return mappedPtr_ != nullptr; }
    std::uintptr_t NativeHandle() const { return bufferHandle_; }
    std::uintptr_t MemoryHandle() const { return memoryHandle_; }

private:
    VkBufferUsageKind usage_{VkBufferUsageKind::STAGING};
    std::size_t size_{0};
    std::uintptr_t deviceHandle_{0};
    std::uintptr_t bufferHandle_{0};
    std::uintptr_t memoryHandle_{0};
    void *mappedPtr_{nullptr};
};

} // namespace npu::tile_fwk::gpu_vk
