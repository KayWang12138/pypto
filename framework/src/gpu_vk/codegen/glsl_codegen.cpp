#include "gpu_vk/codegen/glsl_codegen.h"

#include <sstream>

namespace npu::tile_fwk::gpu_vk {

std::string GlslComputeCodegen::Emit(const GpuVkShaderFunction &shaderFunction, const ShaderMeta &meta) const {
    std::ostringstream builder;
    builder << "#version 450\n";
    builder << "layout(local_size_x = " << meta.dispatch.localX << ", local_size_y = " << meta.dispatch.localY
            << ", local_size_z = " << meta.dispatch.localZ << ") in;\n";
    builder << "layout(push_constant) uniform PushConstants { uint numel; } pc;\n";
    for (const auto &binding : meta.bindings) {
        builder << "layout(set = 0, binding = " << binding.binding << ") buffer " << binding.name << "Buffer { float "
                << binding.name << "[]; };\n";
    }
    builder << "void main() {\n";
    builder << "  uint idx = gl_GlobalInvocationID.x;\n";
    builder << "  if (idx >= pc.numel) { return; }\n";
    if (!shaderFunction.Ops().empty()) {
        builder << "  float value = " << shaderFunction.Ops().back().expr << ";\n";
    } else {
        builder << "  float value = 0.0;\n";
    }
    std::string outputName;
    if (!shaderFunction.Ops().empty()) {
        outputName = shaderFunction.Ops().back().output;
    }
    if (outputName.empty() && !meta.bindings.empty()) {
        outputName = meta.bindings.back().name;
    }
    if (!outputName.empty()) {
        builder << "  " << outputName << "[idx] = value;\n";
    }
    builder << "}\n";
    return builder.str();
}

} // namespace npu::tile_fwk::gpu_vk
