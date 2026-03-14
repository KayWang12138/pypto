#include "gpu_vk/passes/constant_fold.h"

namespace npu::tile_fwk::gpu_vk {

const char *ConstantFoldPass::Name() const {
    return "constant_fold";
}

VkStatus ConstantFoldPass::Run(GpuVkTensorGraph &graph) {
    (void)graph;
    return VkStatus::SUCCESS;
}

} // namespace npu::tile_fwk::gpu_vk
