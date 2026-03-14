#pragma once

#include <vector>

#include "gpu_vk/ir/gpu_vk_tensor_ir.h"

namespace npu::tile_fwk::gpu_vk {

class FusionPlanner {
public:
    std::vector<std::vector<GpuVkOpKind>> BuildPlan(const GpuVkTensorGraph &graph) const;
};

} // namespace npu::tile_fwk::gpu_vk
