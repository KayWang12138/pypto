#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

const char *VkStatusToString(VkStatus status) {
    switch (status) {
        case VkStatus::SUCCESS:
            return "SUCCESS";
        case VkStatus::UNAVAILABLE:
            return "UNAVAILABLE";
        case VkStatus::INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case VkStatus::OUT_OF_MEMORY:
            return "OUT_OF_MEMORY";
        case VkStatus::NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case VkStatus::INTERNAL_ERROR:
            return "INTERNAL_ERROR";
        default:
            return "UNKNOWN";
    }
}

} // namespace npu::tile_fwk::gpu_vk
