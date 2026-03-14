#include "gpu_vk/profile/workgroup_tuner.h"

#include <limits>

namespace npu::tile_fwk::gpu_vk {

std::vector<WorkgroupCandidate> WorkgroupTuner::GenerateCandidates(const VkTensorDesc &desc) const {
    (void)desc;
    return {
        WorkgroupCandidate{32, 1, 1},
        WorkgroupCandidate{64, 1, 1},
        WorkgroupCandidate{128, 1, 1},
    };
}

WorkgroupCandidate WorkgroupTuner::SelectBest(const std::vector<VkKernelProfile> &profiles) const {
    VkKernelProfile bestProfile;
    bestProfile.dispatchNs = std::numeric_limits<std::uint64_t>::max();
    for (const auto &profile : profiles) {
        if (profile.dispatchNs < bestProfile.dispatchNs) {
            bestProfile = profile;
        }
    }
    return WorkgroupCandidate{bestProfile.localX, bestProfile.localY, bestProfile.localZ};
}

} // namespace npu::tile_fwk::gpu_vk
