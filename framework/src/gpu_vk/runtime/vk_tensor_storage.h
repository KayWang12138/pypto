#pragma once

#include <vector>

#include "gpu_vk/runtime/vk_buffer.h"
#include "gpu_vk/runtime/vk_launcher.h"

namespace npu::tile_fwk::gpu_vk {

enum class VkTensorStorageMode {
    CPU_STAGING = 0,
    VK_BUFFER = 1,
    EXTERNAL_MEMORY = 2,
};

struct VkTensorStorage {
    VkTensorStorageMode mode{VkTensorStorageMode::CPU_STAGING};
    VkBufferHandle buffer;
    VkTensorDesc desc;
    std::vector<std::uint8_t> hostData;
    bool HasHostData() const { return !hostData.empty(); }
};

} // namespace npu::tile_fwk::gpu_vk
