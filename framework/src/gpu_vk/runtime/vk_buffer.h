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
    VkStatus Allocate(std::size_t bytes, VkBufferUsageKind usage);
    VkStatus Upload(const void *src, std::size_t bytes);
    VkStatus Download(void *dst, std::size_t bytes) const;
    void Destroy();

    std::size_t Size() const { return data_.size(); }
    VkBufferUsageKind Usage() const { return usage_; }
    bool Allocated() const { return !data_.empty(); }

private:
    VkBufferUsageKind usage_{VkBufferUsageKind::STAGING};
    std::vector<std::uint8_t> data_;
};

} // namespace npu::tile_fwk::gpu_vk
