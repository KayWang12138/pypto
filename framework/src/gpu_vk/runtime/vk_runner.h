#pragma once

#include <string>
#include <vector>

#include "gpu_vk/cache/artifact_cache.h"
#include "gpu_vk/cache/shape_cache.h"
#include "gpu_vk/codegen/gpu_vk_artifact.h"
#include "gpu_vk/common/vk_status.h"
#include "gpu_vk/runtime/vk_allocator.h"
#include "gpu_vk/runtime/vk_descriptor_cache.h"
#include "gpu_vk/runtime/vk_executable.h"
#include "gpu_vk/runtime/vk_pipeline_cache.h"

namespace npu::tile_fwk::gpu_vk {

class VkRunner {
public:
    VkStatus Initialize(bool enableValidation);
    VkStatus Run(
        const GpuVkArtifact &artifact,
        const std::vector<VkTensorDesc> &inputs,
        const std::vector<VkTensorDesc> &outputs,
        const std::vector<const void *> &hostInputPtrs,
        const std::vector<void *> &hostOutputPtrs);
    void Destroy();

    bool Available() const { return available_; }
    bool EnableCache() const { return enableCache_; }
    void SetEnableCache(bool enableCache) { enableCache_ = enableCache; }
    std::size_t PipelineCacheHitCount() const { return pipelineCache_.HitCount(); }
    std::size_t PipelineCacheEntryCount() const { return pipelineCache_.EntryCount(); }

private:
    std::string BuildArtifactKey(const GpuVkArtifact &artifact, const std::vector<VkTensorDesc> &inputs) const;
    std::vector<std::int64_t> BuildShapeSignature(const std::vector<VkTensorDesc> &inputs) const;

    bool available_{false};
    bool validationEnabled_{false};
    bool enableCache_{true};
    VkAllocator allocator_;
    VkDescriptorCache descriptorCache_;
    VkPipelineCacheManager pipelineCache_;
    ShapeCache shapeCache_;
    ArtifactCache artifactCache_;
};

} // namespace npu::tile_fwk::gpu_vk
