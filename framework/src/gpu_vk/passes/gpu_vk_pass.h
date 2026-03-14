#pragma once

#include "gpu_vk/common/vk_status.h"
#include "gpu_vk/ir/gpu_vk_tensor_ir.h"

namespace npu::tile_fwk::gpu_vk {

class GpuVkPass {
public:
    virtual ~GpuVkPass() = default;
    virtual const char *Name() const = 0;
    virtual VkStatus Run(GpuVkTensorGraph &graph) = 0;
};

} // namespace npu::tile_fwk::gpu_vk
