#include "gpu_vk/passes/shared_memory_planner.h"

namespace npu::tile_fwk::gpu_vk {

std::size_t SharedMemoryPlanner::EstimateBytes(const GpuVkDispatchGraph &dispatchGraph) const {
    return dispatchGraph.Bindings().size() * 256;
}

} // namespace npu::tile_fwk::gpu_vk
