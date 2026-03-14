#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

const char *VkStatusToString(VkStatus status) {
    switch (status) {
        case VkStatus::kSuccess:
            return "SUCCESS";
        case VkStatus::kUnavailable:
            return "UNAVAILABLE";
        case VkStatus::kInvalidArgument:
            return "INVALID_ARGUMENT";
        case VkStatus::kOutOfMemory:
            return "OUT_OF_MEMORY";
        case VkStatus::kNotSupported:
            return "NOT_SUPPORTED";
        case VkStatus::kInternalError:
            return "INTERNAL_ERROR";
        case VkStatus::kDeviceNotFound:
            return "DEVICE_NOT_FOUND";
        case VkStatus::kToolNotFound:
            return "TOOL_NOT_FOUND";
        case VkStatus::kShaderCompileFailed:
            return "SHADER_COMPILE_FAILED";
        case VkStatus::kShaderModuleCreateFailed:
            return "SHADER_MODULE_CREATE_FAILED";
        case VkStatus::kDescriptorCreateFailed:
            return "DESCRIPTOR_CREATE_FAILED";
        case VkStatus::kPipelineLayoutCreateFailed:
            return "PIPELINE_LAYOUT_CREATE_FAILED";
        case VkStatus::kPipelineCreateFailed:
            return "PIPELINE_CREATE_FAILED";
        case VkStatus::kBufferCreateFailed:
            return "BUFFER_CREATE_FAILED";
        case VkStatus::kMemoryAllocFailed:
            return "MEMORY_ALLOC_FAILED";
        case VkStatus::kCommandRecordFailed:
            return "COMMAND_RECORD_FAILED";
        case VkStatus::kQueueSubmitFailed:
            return "QUEUE_SUBMIT_FAILED";
        case VkStatus::kDispatchFailed:
            return "DISPATCH_FAILED";
        case VkStatus::kDownloadFailed:
            return "DOWNLOAD_FAILED";
        default:
            return "UNKNOWN";
    }
}

} // namespace npu::tile_fwk::gpu_vk
