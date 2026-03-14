#include "gpu_vk/passes/gpu_vk_pass_manager.h"

#include "gpu_vk/passes/broadcast_legalize.h"
#include "gpu_vk/passes/constant_fold.h"
#include "gpu_vk/passes/dispatch_lowering.h"
#include "gpu_vk/passes/layout_normalize.h"
#include "gpu_vk/passes/op_fusion.h"
#include "gpu_vk/passes/tile_lowering.h"

namespace npu::tile_fwk::gpu_vk {

void GpuVkPassManager::RegisterDefaultPasses() {
    if (!tensorPasses_.empty()) {
        return;
    }
    tensorPasses_.push_back(std::make_unique<ConstantFoldPass>());
    tensorPasses_.push_back(std::make_unique<LayoutNormalizePass>());
    tensorPasses_.push_back(std::make_unique<BroadcastLegalizePass>());
    tensorPasses_.push_back(std::make_unique<OpFusionPass>());
}

VkStatus GpuVkPassManager::RunTensorPasses(GpuVkTensorGraph &graph) {
    RegisterDefaultPasses();
    for (const auto &pass : tensorPasses_) {
        const VkStatus status = pass->Run(graph);
        if (status != VkStatus::kSuccess) {
            return status;
        }
    }
    return VkStatus::kSuccess;
}

VkStatus GpuVkPassManager::LowerToDispatch(
    GpuVkTensorGraph &graph, GpuVkDispatchGraph &dispatchGraph, GpuVkShaderFunction &shaderFunction) {
    VkStatus status = RunTensorPasses(graph);
    if (status != VkStatus::kSuccess) {
        return status;
    }

    TileLoweringPass tileLowering;
    status = tileLowering.Run(graph, dispatchGraph);
    if (status != VkStatus::kSuccess) {
        return status;
    }

    DispatchLoweringPass dispatchLowering;
    return dispatchLowering.Run(graph, dispatchGraph, shaderFunction);
}

} // namespace npu::tile_fwk::gpu_vk
