#pragma once

#include <vector>

#include "gpu_vk/profile/vk_profiler.h"
#include "gpu_vk/runtime/vk_launcher.h"

namespace npu::tile_fwk::gpu_vk {

struct WorkgroupCandidate {
    std::uint32_t localX{1};
    std::uint32_t localY{1};
    std::uint32_t localZ{1};
};

class WorkgroupTuner {
public:
    std::vector<WorkgroupCandidate> GenerateCandidates(const VkTensorDesc &desc) const;
    WorkgroupCandidate SelectBest(const std::vector<VkKernelProfile> &profiles) const;
};

} // namespace npu::tile_fwk::gpu_vk
