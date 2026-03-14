#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_vk/codegen/shader_meta.h"
#include "gpu_vk/ir/gpu_vk_shader_ir.h"

namespace npu::tile_fwk::gpu_vk {

struct GpuVkArtifact {
    ShaderMeta meta;
    GpuVkShaderFunction shaderFunction;
    std::string glsl;
    std::vector<std::uint32_t> spirv;
    std::string debugSummary;

    bool Empty() const;
};

} // namespace npu::tile_fwk::gpu_vk
