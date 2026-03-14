#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gpu_vk/common/vk_status.h"

namespace npu::tile_fwk::gpu_vk {

struct SpirvCompileOptions {
    bool optimize{true};
    bool debugInfo{false};
};

class SpirvCompiler {
public:
    VkStatus CompileGlslToSpirv(
        const std::string &glsl, const SpirvCompileOptions &options, std::vector<std::uint32_t> &spirvOut) const;
};

} // namespace npu::tile_fwk::gpu_vk
