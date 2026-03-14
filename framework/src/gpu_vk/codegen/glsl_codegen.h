#pragma once

#include <string>

#include "gpu_vk/codegen/shader_meta.h"
#include "gpu_vk/ir/gpu_vk_shader_ir.h"

namespace npu::tile_fwk::gpu_vk {

class GlslComputeCodegen {
public:
    std::string Emit(const GpuVkShaderFunction &shaderFunction, const ShaderMeta &meta) const;
};

} // namespace npu::tile_fwk::gpu_vk
