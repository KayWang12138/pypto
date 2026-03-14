#pragma once

#include <chrono>
#include <cstdint>

#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

struct VkKernelProfile {
    std::uint64_t dispatchNs{0};
    std::uint32_t groupX{1};
    std::uint32_t groupY{1};
    std::uint32_t groupZ{1};
    std::uint32_t localX{1};
    std::uint32_t localY{1};
    std::uint32_t localZ{1};
};

class VkProfiler {
public:
    VkStatus Begin();
    VkStatus End();
    VkKernelProfile Read() const;
    void SetLaunchShape(
        std::uint32_t groupX, std::uint32_t groupY, std::uint32_t groupZ, std::uint32_t localX, std::uint32_t localY,
        std::uint32_t localZ);

private:
    std::chrono::steady_clock::time_point begin_;
    std::chrono::steady_clock::time_point end_;
    bool running_{false};
    VkKernelProfile profile_;
};

} // namespace npu::tile_fwk::gpu_vk
