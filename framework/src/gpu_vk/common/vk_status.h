#pragma once

namespace npu::tile_fwk::gpu_vk {

enum class VkStatus {
    kSuccess = 0,
    kUnavailable = 1,
    kInvalidArgument = 2,
    kOutOfMemory = 3,
    kNotSupported = 4,
    kInternalError = 5,
    kDeviceNotFound = 6,
    kToolNotFound = 7,
    kShaderCompileFailed = 8,
    kShaderModuleCreateFailed = 9,
    kDescriptorCreateFailed = 10,
    kPipelineLayoutCreateFailed = 11,
    kPipelineCreateFailed = 12,
    kBufferCreateFailed = 13,
    kMemoryAllocFailed = 14,
    kCommandRecordFailed = 15,
    kQueueSubmitFailed = 16,
    kDispatchFailed = 17,
    kDownloadFailed = 18,
};

const char *VkStatusToString(VkStatus status);

} // namespace npu::tile_fwk::gpu_vk
