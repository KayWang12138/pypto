#pragma once

#include <cstddef>

#include "gpu_vk/passes/gpu_vk_pass.h"

namespace npu::tile_fwk::gpu_vk {

class OpFusionPass : public GpuVkPass {
public:
    const char *Name() const override;
    VkStatus Run(GpuVkTensorGraph &graph) override;
    std::size_t FusionGroupCount() const;

private:
    std::size_t fusionGroupCount_{0};
};

} // namespace npu::tile_fwk::gpu_vk
