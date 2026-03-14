#include "gpu_vk/runtime/vk_pipeline_cache.h"

namespace npu::tile_fwk::gpu_vk {

bool VkPipelineCacheManager::Find(const PipelineCacheKey &key, VkComputePipelineHandle *&pipeline) {
    for (auto &entry : entries_) {
        if (IsSameKey(entry.key, key)) {
            ++hitCount_;
            pipeline = entry.pipeline.get();
            return true;
        }
    }
    pipeline = nullptr;
    return false;
}

void VkPipelineCacheManager::Insert(const PipelineCacheKey &key, std::unique_ptr<VkComputePipelineHandle> pipeline) {
    entries_.push_back(Entry{key, std::move(pipeline)});
}

std::size_t VkPipelineCacheManager::EntryCount() const {
    return entries_.size();
}

std::size_t VkPipelineCacheManager::HitCount() const {
    return hitCount_;
}

bool VkPipelineCacheManager::IsSameKey(const PipelineCacheKey &lhs, const PipelineCacheKey &rhs) {
    return lhs.kernelName == rhs.kernelName && lhs.shapeSignature == rhs.shapeSignature;
}

} // namespace npu::tile_fwk::gpu_vk
