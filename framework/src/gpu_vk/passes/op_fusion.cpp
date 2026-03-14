#include "gpu_vk/passes/op_fusion.h"

#include <algorithm>

namespace npu::tile_fwk::gpu_vk {

const char *OpFusionPass::Name() const {
    return "op_fusion";
}

VkStatus OpFusionPass::Run(GpuVkTensorGraph &graph) {
    fusionGroupCount_ = std::max<std::size_t>(1, graph.NodeCount());
    return VkStatus::SUCCESS;
}

std::size_t OpFusionPass::FusionGroupCount() const {
    return fusionGroupCount_;
}

} // namespace npu::tile_fwk::gpu_vk
