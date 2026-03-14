#include "gpu_vk/runtime/vk_descriptor_cache.h"

#include <sstream>

namespace npu::tile_fwk::gpu_vk {

namespace {

std::string MakeDescriptorKey(const std::vector<GpuVkBufferBinding> &bindings) {
    std::ostringstream builder;
    for (const auto &binding : bindings) {
        builder << binding.name << ":" << binding.binding << ":" << (binding.isOutput ? 1 : 0) << ";";
    }
    return builder.str();
}

} // namespace

std::size_t VkDescriptorCache::GetOrCreate(const std::vector<GpuVkBufferBinding> &bindings) {
    const std::string key = MakeDescriptorKey(bindings);
    for (std::size_t i = 0; i < keys_.size(); ++i) {
        if (keys_[i] == key) {
            return i;
        }
    }
    keys_.push_back(key);
    return keys_.size() - 1;
}

std::size_t VkDescriptorCache::EntryCount() const {
    return keys_.size();
}

} // namespace npu::tile_fwk::gpu_vk
