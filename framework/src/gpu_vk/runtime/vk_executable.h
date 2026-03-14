#pragma once

#include <memory>

#include "gpu_vk/codegen/gpu_vk_artifact.h"
#include "gpu_vk/runtime/vk_pipeline.h"

namespace npu::tile_fwk::gpu_vk {

struct VkExecutable {
    GpuVkArtifact artifact;
    std::unique_ptr<VkComputePipelineHandle> pipeline;

    bool Ready() const;
};

} // namespace npu::tile_fwk::gpu_vk
