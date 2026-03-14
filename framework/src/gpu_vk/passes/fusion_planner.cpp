#include "gpu_vk/passes/fusion_planner.h"

namespace npu::tile_fwk::gpu_vk {

std::vector<std::vector<GpuVkOpKind>> FusionPlanner::BuildPlan(const GpuVkTensorGraph &graph) const {
    std::vector<std::vector<GpuVkOpKind>> plan;
    for (const auto &node : graph.Nodes()) {
        plan.push_back({node.op});
    }
    return plan;
}

} // namespace npu::tile_fwk::gpu_vk
