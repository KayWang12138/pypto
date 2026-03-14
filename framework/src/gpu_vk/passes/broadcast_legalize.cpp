#include "gpu_vk/passes/broadcast_legalize.h"

namespace npu::tile_fwk::gpu_vk {

const char *BroadcastLegalizePass::Name() const {
    return "broadcast_legalize";
}

VkStatus BroadcastLegalizePass::Run(GpuVkTensorGraph &graph) {
    (void)graph;
    return VkStatus::kSuccess;
}

} // namespace npu::tile_fwk::gpu_vk
