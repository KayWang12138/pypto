#include "gpu_vk/cache/shape_cache.h"

namespace npu::tile_fwk::gpu_vk {

bool ShapeCache::Find(const std::string &key, std::vector<std::int64_t> &shape) const {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }
    shape = it->second;
    return true;
}

void ShapeCache::Insert(const std::string &key, const std::vector<std::int64_t> &shape) {
    entries_[key] = shape;
}

std::size_t ShapeCache::EntryCount() const {
    return entries_.size();
}

} // namespace npu::tile_fwk::gpu_vk
