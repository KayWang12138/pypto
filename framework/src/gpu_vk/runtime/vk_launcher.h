#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_vk/common/vk_status.h"
#include "gpu_vk/runtime/vk_device.h"
#include "gpu_vk/runtime/vk_instance.h"

namespace npu::tile_fwk::gpu_vk {

struct VkTensorDesc {
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> stride;
    std::int32_t dtype{0};
    std::size_t nbytes{0};
};

class VkLauncher {
public:
    VkStatus Initialize(bool enableValidation);
    void Destroy();

    bool Available() const { return available_; }
    bool ValidationEnabled() const { return validationEnabled_; }

    VkStatus RunElementwiseBinary(
        const std::string &opName,
        const void *input0,
        const void *input1,
        void *output,
        const VkTensorDesc &desc);
    VkStatus RunElementwiseUnary(const std::string &opName, const void *input, void *output, const VkTensorDesc &desc);

private:
    bool available_{false};
    bool validationEnabled_{false};
    VkInstanceContext instance_;
    VkDeviceContext device_;
};

} // namespace npu::tile_fwk::gpu_vk
