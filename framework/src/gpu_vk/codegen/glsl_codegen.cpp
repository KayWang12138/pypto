#include "gpu_vk/codegen/glsl_codegen.h"

#include <sstream>

namespace npu::tile_fwk::gpu_vk {

std::string GlslComputeCodegen::Emit(const GpuVkShaderFunction &shaderFunction, const ShaderMeta &meta) const {
    std::ostringstream builder;
    builder << "#version 450\n";
    builder << "layout(local_size_x = " << meta.dispatch.localX << ", local_size_y = " << meta.dispatch.localY
            << ", local_size_z = " << meta.dispatch.localZ << ") in;\n";
    for (const auto &binding : meta.bindings) {
        builder << "layout(set = 0, binding = " << binding.binding << ") buffer " << binding.name << "Buffer { float "
                << binding.name << "[]; };\n";
    }
    builder << "void main() {\n";
    builder << "  uint idx = gl_GlobalInvocationID.x;\n";
    if (!shaderFunction.Ops().empty()) {
        builder << "  float value = " << shaderFunction.Ops().back().expr << ";\n";
    } else {
        builder << "  float value = 0.0;\n";
    }
    if (!meta.bindings.empty()) {
        builder << "  " << meta.bindings.back().name << "[idx] = value;\n";
    }
    builder << "}\n";
    return builder.str();
}

} // namespace npu::tile_fwk::gpu_vk
