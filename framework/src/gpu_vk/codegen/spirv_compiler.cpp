#include "gpu_vk/codegen/spirv_compiler.h"

#include <cstddef>

namespace npu::tile_fwk::gpu_vk {

VkStatus SpirvCompiler::CompileGlslToSpirv(
    const std::string &glsl, const SpirvCompileOptions &options, std::vector<std::uint32_t> &spirvOut) const {
    if (glsl.empty()) {
        spirvOut.clear();
        return VkStatus::INVALID_ARGUMENT;
    }

    std::uint32_t hash = 2166136261u;
    for (unsigned char ch : glsl) {
        hash ^= static_cast<std::uint32_t>(ch);
        hash *= 16777619u;
    }

    spirvOut = {
        0x07230203u,
        static_cast<std::uint32_t>(glsl.size()),
        options.optimize ? 1u : 0u,
        options.debugInfo ? 1u : 0u,
        hash,
    };
    return VkStatus::SUCCESS;
}

} // namespace npu::tile_fwk::gpu_vk
