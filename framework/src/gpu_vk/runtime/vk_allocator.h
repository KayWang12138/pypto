#pragma once

#include <cstddef>

#include "gpu_vk/common/vk_status.h"
#include "gpu_vk/runtime/vk_buffer.h"

namespace npu::tile_fwk::gpu_vk {

class VkAllocator {
public:
    VkStatus Initialize();
    VkStatus AllocStorageBuffer(std::size_t bytes, VkBufferHandle &out);
    VkStatus AllocStagingBuffer(std::size_t bytes, VkBufferHandle &out);
    void Destroy();

    bool Available() const { return available_; }

private:
    bool available_{false};
};

} // namespace npu::tile_fwk::gpu_vk
