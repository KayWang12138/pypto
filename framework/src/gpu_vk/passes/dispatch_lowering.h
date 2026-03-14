#pragma once

#include "gpu_vk/common/vk_status.h"
#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"
#include "gpu_vk/ir/gpu_vk_shader_ir.h"
#include "gpu_vk/ir/gpu_vk_tensor_ir.h"

namespace npu::tile_fwk::gpu_vk {

class DispatchLoweringPass {
public:
    VkStatus Run(
        const GpuVkTensorGraph &graph, GpuVkDispatchGraph &dispatchGraph, GpuVkShaderFunction &shaderFunction) const;
};

} // namespace npu::tile_fwk::gpu_vk
