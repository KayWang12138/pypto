#pragma once

namespace npu::tile_fwk::gpu_vk {

enum class VkStatus {
    SUCCESS = 0,
    UNAVAILABLE = 1,
    INVALID_ARGUMENT = 2,
    OUT_OF_MEMORY = 3,
    NOT_SUPPORTED = 4,
    INTERNAL_ERROR = 5,
};

const char *VkStatusToString(VkStatus status);

} // namespace npu::tile_fwk::gpu_vk
