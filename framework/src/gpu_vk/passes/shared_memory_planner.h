#pragma once

#include <cstddef>

#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"

namespace npu::tile_fwk::gpu_vk {

class SharedMemoryPlanner {
public:
    std::size_t EstimateBytes(const GpuVkDispatchGraph &dispatchGraph) const;
};

} // namespace npu::tile_fwk::gpu_vk
