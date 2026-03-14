#pragma once

#include <memory>
#include <vector>

#include "gpu_vk/common/vk_status.h"
#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"
#include "gpu_vk/ir/gpu_vk_shader_ir.h"
#include "gpu_vk/ir/gpu_vk_tensor_ir.h"
#include "gpu_vk/passes/gpu_vk_pass.h"

namespace npu::tile_fwk::gpu_vk {

class GpuVkPassManager {
public:
    void RegisterDefaultPasses();
    VkStatus RunTensorPasses(GpuVkTensorGraph &graph);
    VkStatus LowerToDispatch(GpuVkTensorGraph &graph, GpuVkDispatchGraph &dispatchGraph, GpuVkShaderFunction &shaderFunction);

private:
    std::vector<std::unique_ptr<GpuVkPass>> tensorPasses_;
};

} // namespace npu::tile_fwk::gpu_vk
