#include "gpu_vk/cache/artifact_cache.h"

namespace npu::tile_fwk::gpu_vk {

bool ArtifactCache::Find(const std::string &key, GpuVkArtifact &artifact) const {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }
    artifact = it->second;
    return true;
}

void ArtifactCache::Insert(const std::string &key, const GpuVkArtifact &artifact) {
    entries_[key] = artifact;
}

std::size_t ArtifactCache::EntryCount() const {
    return entries_.size();
}

} // namespace npu::tile_fwk::gpu_vk
