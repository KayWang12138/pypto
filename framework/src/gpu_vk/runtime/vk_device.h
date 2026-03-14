#pragma once

#include <cstdint>

#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

class VkDeviceContext {
public:
    VkStatus Initialize(std::uintptr_t instanceHandle);
    void Destroy();

    bool Available() const { return available_; }
    std::uintptr_t InstanceHandle() const { return instanceHandle_; }
    std::uintptr_t NativeHandle() const { return nativeHandle_; }
    std::uint32_t ComputeQueueFamily() const { return computeQueueFamily_; }

private:
    bool available_{false};
    std::uintptr_t instanceHandle_{0};
    std::uintptr_t nativeHandle_{0};
    std::uint32_t computeQueueFamily_{0};
};

} // namespace npu::tile_fwk::gpu_vk
