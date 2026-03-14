#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gpu_vk/runtime/vk_pipeline.h"

namespace npu::tile_fwk::gpu_vk {

struct PipelineCacheKey {
    std::string kernelName;
    std::vector<std::int64_t> shapeSignature;
};

class VkPipelineCacheManager {
public:
    bool Find(const PipelineCacheKey &key, VkComputePipelineHandle *&pipeline);
    void Insert(const PipelineCacheKey &key, std::unique_ptr<VkComputePipelineHandle> pipeline);
    std::size_t EntryCount() const;
    std::size_t HitCount() const;

private:
    struct Entry {
        PipelineCacheKey key;
        std::unique_ptr<VkComputePipelineHandle> pipeline;
    };

    static bool IsSameKey(const PipelineCacheKey &lhs, const PipelineCacheKey &rhs);

    std::vector<Entry> entries_;
    std::size_t hitCount_{0};
};

} // namespace npu::tile_fwk::gpu_vk
