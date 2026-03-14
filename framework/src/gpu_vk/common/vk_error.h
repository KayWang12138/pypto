#pragma once

#include <stdexcept>
#include <string>

#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

class VkError : public std::runtime_error {
public:
    VkError(VkStatus status, const std::string &message)
        : std::runtime_error(message), status_(status) {}

    VkStatus Status() const { return status_; }

private:
    VkStatus status_;
};

} // namespace npu::tile_fwk::gpu_vk
