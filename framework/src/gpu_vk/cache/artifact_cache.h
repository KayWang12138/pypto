#pragma once

#include <string>
#include <unordered_map>

#include "gpu_vk/codegen/gpu_vk_artifact.h"

namespace npu::tile_fwk::gpu_vk {

class ArtifactCache {
public:
    bool Find(const std::string &key, GpuVkArtifact &artifact) const;
    void Insert(const std::string &key, const GpuVkArtifact &artifact);
    std::size_t EntryCount() const;

private:
    std::unordered_map<std::string, GpuVkArtifact> entries_;
};

} // namespace npu::tile_fwk::gpu_vk
