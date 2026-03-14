#pragma once

#include <cstdint>

#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

class VkInstanceContext {
public:
    VkStatus Initialize(bool enableValidation);
    void Destroy();

    bool ValidationEnabled() const { return validationEnabled_; }
    std::uintptr_t NativeHandle() const { return nativeHandle_; }
    bool Available() const { return available_; }

private:
    bool validationEnabled_{false};
    bool available_{false};
    std::uintptr_t nativeHandle_{0};
};

} // namespace npu::tile_fwk::gpu_vk
