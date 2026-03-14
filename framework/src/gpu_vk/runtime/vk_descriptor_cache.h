#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"

namespace npu::tile_fwk::gpu_vk {

class VkDescriptorCache {
public:
    std::size_t GetOrCreate(const std::vector<GpuVkBufferBinding> &bindings);
    std::size_t EntryCount() const;

private:
    std::vector<std::string> keys_;
};

} // namespace npu::tile_fwk::gpu_vk
