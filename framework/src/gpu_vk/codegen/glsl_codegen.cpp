#include "gpu_vk/codegen/glsl_codegen.h"

#include <algorithm>
#include <sstream>

namespace npu::tile_fwk::gpu_vk {

namespace {

bool IsMatmulShader(const GpuVkShaderFunction &shaderFunction) {
    return !shaderFunction.Ops().empty() && shaderFunction.Ops().back().op == GpuVkOpKind::MATMUL;
}

const ShaderBindingMeta *FindOutputBinding(const ShaderMeta &meta) {
    for (const auto &binding : meta.bindings) {
        if (binding.isOutput) {
            return &binding;
        }
    }
    return meta.bindings.empty() ? nullptr : &meta.bindings.back();
}

void EmitBufferBindings(std::ostringstream &builder, const ShaderMeta &meta) {
    for (const auto &binding : meta.bindings) {
        builder << "layout(set = 0, binding = " << binding.binding << ") buffer " << binding.name << "Buffer { float "
                << binding.name << "[]; };\n";
    }
}

std::string EmitElementwiseShader(const GpuVkShaderFunction &shaderFunction, const ShaderMeta &meta) {
    std::ostringstream builder;
    builder << "#version 450\n";
    builder << "layout(local_size_x = " << meta.dispatch.localX << ", local_size_y = " << meta.dispatch.localY
            << ", local_size_z = " << meta.dispatch.localZ << ") in;\n";
    builder << "layout(push_constant) uniform PushConstants { uint numel; } pc;\n";
    EmitBufferBindings(builder, meta);
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

std::string EmitMatmulShader(const ShaderMeta &meta) {
    std::ostringstream builder;
    const std::uint32_t tileN = std::max<std::uint32_t>(1, meta.dispatch.localX);
    const std::uint32_t tileM = std::max<std::uint32_t>(1, meta.dispatch.localY);
    const std::uint32_t tileK = 16;
    builder << "#version 450\n";
    builder << "layout(local_size_x = " << tileN << ", local_size_y = " << tileM
            << ", local_size_z = " << meta.dispatch.localZ << ") in;\n";
    builder << "layout(push_constant) uniform PushConstants { uint B; uint M; uint N; uint K; uint lhsBatchStride; uint rhsBatchStride; } pc;\n";
    EmitBufferBindings(builder, meta);
    builder << "shared float Asub[" << tileM << "][" << tileK << "];\n";
    builder << "shared float Bsub[" << tileK << "][" << tileN << "];\n";

    const std::string lhsName = meta.bindings.size() > 0 ? meta.bindings[0].name : "input0";
    const std::string rhsName = meta.bindings.size() > 1 ? meta.bindings[1].name : "input1";
    const ShaderBindingMeta *outputBinding = FindOutputBinding(meta);
    const std::string outputName = outputBinding == nullptr ? "output0" : outputBinding->name;

    builder << "void main() {\n";
    builder << "  uint localCol = gl_LocalInvocationID.x;\n";
    builder << "  uint localRow = gl_LocalInvocationID.y;\n";
    builder << "  uint batch = gl_WorkGroupID.z;\n";
    builder << "  uint col = gl_WorkGroupID.x * " << tileN << "u + localCol;\n";
    builder << "  uint row = gl_WorkGroupID.y * " << tileM << "u + localRow;\n";
    builder << "  if (batch >= pc.B) { return; }\n";
    builder << "  uint aBase = batch * pc.lhsBatchStride;\n";
    builder << "  uint bBase = batch * pc.rhsBatchStride;\n";
    builder << "  uint cBase = batch * pc.M * pc.N;\n";
    builder << "  float acc = 0.0;\n";
    builder << "  uint tileCount = (pc.K + " << (tileK - 1) << "u) / " << tileK << "u;\n";
    builder << "  for (uint tile = 0; tile < tileCount; ++tile) {\n";
    builder << "    uint tiledACol = tile * " << tileK << "u + localCol;\n";
    builder << "    uint tiledBRow = tile * " << tileK << "u + localRow;\n";
    builder << "    Asub[localRow][localCol] = (row < pc.M && tiledACol < pc.K)\n";
    builder << "        ? " << lhsName << "[aBase + row * pc.K + tiledACol]\n";
    builder << "        : 0.0;\n";
    builder << "    Bsub[localRow][localCol] = (tiledBRow < pc.K && col < pc.N)\n";
    builder << "        ? " << rhsName << "[bBase + tiledBRow * pc.N + col]\n";
    builder << "        : 0.0;\n";
    builder << "    barrier();\n";
    builder << "    for (uint k = 0; k < " << tileK << "u; ++k) {\n";
    builder << "      acc += Asub[localRow][k] * Bsub[k][localCol];\n";
    builder << "    }\n";
    builder << "    barrier();\n";
    builder << "  }\n";
    builder << "  if (row < pc.M && col < pc.N) {\n";
    builder << "    " << outputName << "[cBase + row * pc.N + col] = acc;\n";
    builder << "  }\n";
    builder << "}\n";
    return builder.str();
}

} // namespace

std::string GlslComputeCodegen::Emit(const GpuVkShaderFunction &shaderFunction, const ShaderMeta &meta) const {
    if (IsMatmulShader(shaderFunction)) {
        return EmitMatmulShader(meta);
    }
    return EmitElementwiseShader(shaderFunction, meta);
}

} // namespace npu::tile_fwk::gpu_vk
