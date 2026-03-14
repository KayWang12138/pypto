#include "gpu_vk/profile/vk_profiler.h"

namespace npu::tile_fwk::gpu_vk {

VkStatus VkProfiler::Begin() {
    begin_ = std::chrono::steady_clock::now();
    running_ = true;
    return VkStatus::kSuccess;
}

VkStatus VkProfiler::End() {
    if (!running_) {
        return VkStatus::kInvalidArgument;
    }
    end_ = std::chrono::steady_clock::now();
    running_ = false;
    profile_.dispatchNs =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - begin_).count());
    return VkStatus::kSuccess;
}

VkKernelProfile VkProfiler::Read() const {
    return profile_;
}

void VkProfiler::SetLaunchShape(
    std::uint32_t groupX,
    std::uint32_t groupY,
    std::uint32_t groupZ,
    std::uint32_t localX,
    std::uint32_t localY,
    std::uint32_t localZ) {
    profile_.groupX = groupX;
    profile_.groupY = groupY;
    profile_.groupZ = groupZ;
    profile_.localX = localX;
    profile_.localY = localY;
    profile_.localZ = localZ;
}

} // namespace npu::tile_fwk::gpu_vk
