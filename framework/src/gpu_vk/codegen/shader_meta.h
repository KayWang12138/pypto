#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_vk/ir/gpu_vk_dispatch_ir.h"

namespace npu::tile_fwk::gpu_vk {

struct ShaderBindingMeta {
    std::uint32_t binding{0};
    std::string name;
    std::int32_t dtype{0};
    bool isOutput{false};
};

struct PushConstantMeta {
    std::string name;
    std::uint32_t offset{0};
    std::uint32_t size{0};
};

struct ShaderMeta {
    std::string kernelName;
    std::vector<ShaderBindingMeta> bindings;
    std::vector<PushConstantMeta> pushConstants;
    GpuVkDispatchParam dispatch;
};

} // namespace npu::tile_fwk::gpu_vk
