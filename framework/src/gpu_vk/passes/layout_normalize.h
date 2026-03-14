#pragma once

#include "gpu_vk/passes/gpu_vk_pass.h"

namespace npu::tile_fwk::gpu_vk {

class LayoutNormalizePass : public GpuVkPass {
public:
    const char *Name() const override;
    VkStatus Run(GpuVkTensorGraph &graph) override;
};

} // namespace npu::tile_fwk::gpu_vk
